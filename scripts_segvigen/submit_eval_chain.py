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
                 backfilled=False, draws=5, quiet=False, only_shards=None,
                 job_name=None, bridged_path=None, extra_args=""):
    """only_shards: re-run just these shard indices and merge again. This is the
    repair path for a checkpoint that lost a shard, which happens on a shared
    partition when another user's job preempts one; the merge keeps the shards
    that did land, so only the gap has to be recomputed."""
    bridged = bridged_path or os.path.join(store, "bridged", f"step{step:07d}.pt")
    started = time.time()
    jn = (["-J", job_name] if job_name else [])
    array_spec = (",".join(str(k) for k in sorted(only_shards)) if only_shards
                  else f"0-{n_shards - 1}")
    common = (f"LIVE_EVAL_STEP={step},LIVE_EVAL_STORE={store},"
              f"LIVE_EVAL_VAL_LOSS={val_loss},LIVE_EVAL_NSHARDS={n_shards}")

    bridge_id = None
    if not (only_shards and os.path.exists(bridged)):
        bridge_id = sbatch(None, "live_eval_bridge.sbatch",
                           f"{common},LIGHTNING_CKPT={ckpt}",
                           extra=jn and ["-J", f"{job_name}_bridge"] or [])
    # No dependency on the bridge: the shard script polls for the bridged file
    # instead (see live_eval_shard.sbatch for why that was kept).
    array_id = sbatch(None, "live_eval_shard.sbatch",
                      f"{common},BRIDGED_CKPT={bridged},"
                      f"LIVE_EVAL_EXTRA=--draws {draws} {extra_args}".strip(),
                      extra=[f"--array={array_spec}"] + (jn and ["-J", f"{job_name}_shard"] or []))
    merge_extra = "--backfilled" if backfilled else ""
    if only_shards:
        # Keep the shards already on disk; this run only fills the gap.
        merge_extra = (merge_extra + " --keep_shards").strip()
    merge_id = sbatch(None, "live_eval_merge.sbatch",
                      f"{common},LIVE_EVAL_ARRAY_JOB={array_id},"
                      f"LIVE_EVAL_STARTED_AT={started:.0f},"
                      f"LIVE_EVAL_MERGE_EXTRA={merge_extra}",
                      dependency=f"afterany:{array_id}",
                      extra=jn and ["-J", f"{job_name}_merge"] or [])
    # A CLAIM on this checkpoint, so any other submitter (the driver, a one-off
    # backlog drain, a person) can see it is already being evaluated and does not
    # queue a second array for the same weights. The claim names the jobs, so a
    # reader can tell a live claim from a stale one by asking SLURM.
    claim = {"step": step, "bridge": bridge_id, "array": array_id, "merge": merge_id,
             "n_shards": n_shards, "backfilled": backfilled, "repair": bool(only_shards),
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
    ap.add_argument("--only_shards", default=None,
                    help="comma-separated shard indices to re-run, for repairing a checkpoint "
                         "that lost one")
    args = ap.parse_args()
    only = ([int(x) for x in args.only_shards.split(",")] if args.only_shards else None)
    submit_chain(args.ckpt, args.step, args.store, args.n_shards, args.val_loss,
                 args.backfilled, args.draws, only_shards=only)


if __name__ == "__main__":
    sys.exit(main())
