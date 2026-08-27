"""Submit one checkpoint's whole evaluation as a three-stage SLURM chain.

    bridge (CPU, once)  ->  shards (array of independent single-GPU tasks)
                        ->  merge (CPU, afterany)

The shards are an ARRAY, deliberately. One job asking for eight GPUs waits until
eight are free at the same instant; an array of eight one-GPU tasks starts as
many as the pool can take right now and the rest join as cards free up. That is
also what lets several checkpoints' arrays queue together and flow through the
pool instead of blocking each other.

The merge depends on afterANY rather than afterok, so a dead shard still yields
a record, one that states its own partial coverage instead of quietly averaging
a smaller denominator.

Usage:
    python scripts_segvigen/submit_eval_chain.py --ckpt <path> --step 4530 \
        [--store ...] [--n_shards 8] [--val_loss 0.31] [--backfilled]
Prints one line: BRIDGE <id> ARRAY <id> MERGE <id>
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.dirname(HERE)
DEFAULT_STORE = "/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/live_eval"


def sbatch(args, script, env_export, dependency=None, extra=()):
    cmd = ["sbatch", "--parsable", f"--export=ALL,{env_export}"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd += list(extra) + [os.path.join(PROBE, script)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        raise SystemExit(f"sbatch {script} failed (rc={p.returncode}): {p.stderr.strip()[:400]}")
    return p.stdout.strip().split(";")[0].strip()


def submit_chain(ckpt, step, store=DEFAULT_STORE, n_shards=8, val_loss=None,
                 backfilled=False, draws=5, quiet=False):
    bridged = os.path.join(store, "bridged", f"step{step:07d}.pt")
    started = time.time()
    common = (f"LIVE_EVAL_STEP={step},LIVE_EVAL_STORE={store},"
              f"LIVE_EVAL_VAL_LOSS={val_loss},LIVE_EVAL_NSHARDS={n_shards}")

    bridge_id = sbatch(None, "live_eval_bridge.sbatch",
                       f"{common},LIGHTNING_CKPT={ckpt}")
    # No dependency on the bridge: the shard script polls for the bridged file
    # instead (see live_eval_shard.sbatch for why that was kept).
    array_id = sbatch(None, "live_eval_shard.sbatch",
                      f"{common},BRIDGED_CKPT={bridged},LIVE_EVAL_EXTRA=--draws {draws}",
                      extra=[f"--array=0-{n_shards - 1}"])
    merge_extra = "--backfilled" if backfilled else ""
    merge_id = sbatch(None, "live_eval_merge.sbatch",
                      f"{common},LIVE_EVAL_ARRAY_JOB={array_id},"
                      f"LIVE_EVAL_STARTED_AT={started:.0f},"
                      f"LIVE_EVAL_MERGE_EXTRA={merge_extra}",
                      dependency=f"afterany:{array_id}")
    # A CLAIM on this checkpoint, so any other submitter (the driver, a one-off
    # backlog drain, a person) can see it is already being evaluated and does not
    # queue a second array for the same weights. The claim names the jobs, so a
    # reader can tell a live claim from a stale one by asking SLURM.
    claim = {"step": step, "bridge": bridge_id, "array": array_id, "merge": merge_id,
             "n_shards": n_shards, "backfilled": backfilled,
             "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    claim_dir = os.path.join(store, "inflight")
    os.makedirs(claim_dir, exist_ok=True)
    tmp = os.path.join(claim_dir, f".step{step:07d}.json")
    with open(tmp, "w") as f:
        json.dump(claim, f, indent=1)
    os.replace(tmp, os.path.join(claim_dir, f"step{step:07d}.json"))

    if not quiet:
        print(f"BRIDGE {bridge_id} ARRAY {array_id} MERGE {merge_id}")
    return {"bridge": bridge_id, "array": array_id, "merge": merge_id,
            "step": step, "n_shards": n_shards, "started_at": started}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--n_shards", type=int, default=8)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--val_loss", default=None)
    ap.add_argument("--backfilled", action="store_true")
    args = ap.parse_args()
    submit_chain(args.ckpt, args.step, args.store, args.n_shards, args.val_loss,
                 args.backfilled, args.draws)


if __name__ == "__main__":
    sys.exit(main())
