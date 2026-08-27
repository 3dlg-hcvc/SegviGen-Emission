"""Watch the training node for new checkpoints and evaluate each one.

The standing loop behind board job live_eval_loop. It runs as a small CPU job
that outlives any chat session, and does only bookkeeping: the actual work is
one GPU job per checkpoint (live_eval_one.py).

Every poll it:
  1. lists step*.ckpt on the training node over ssh (the checkpoints live on
     that node's own local disk, so no shared filesystem can see them),
  2. treats a checkpoint as READY only once its size and modification time have
     been identical across two polls and it is at least settle_s old, so a file
     still being written is never copied,
  3. does nothing while an evaluation job is still in flight, one at a time,
  4. otherwise picks the next checkpoint to evaluate, copies it onto falas by
     running the copy ON the training node (falas is mounted there, so this is
     one hop, not a pull-then-push), and submits one GPU job for it.

Which checkpoint is next: when the loop has fallen behind by more than
--catch_up checkpoints, it jumps to the NEWEST one and records every checkpoint
it passed over as skipped, so the curve shows an honest gap instead of a
backlog it will never work through. Within a small backlog it works oldest
first, which is also what seeds the curve from the checkpoints that already
existed when the loop started.

Ending it: write a file named STOP in the store directory. The driver finishes
its current cycle, exits, and its sbatch wrapper does not chain a successor.

Usage (the sbatch wrapper is live_eval_driver.sbatch):
    python scripts_segvigen/live_eval_driver.py \
        --store /3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/live_eval \
        --max_hours 11.5
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.dirname(HERE)

TRAIN_NODE = "cs-venus-05"
TRAIN_PORT = "24"
CKPT_DIR = "/localscratch/xya120/segvigen-lightning/outputs/segvigen_256_bw_6gpu/ckpts"
DEFAULT_STORE = ("/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/live_eval")
EVAL_SBATCH = os.path.join(PROBE, "live_eval_one.sbatch")
DRIVER_SBATCH = os.path.join(PROBE, "live_eval_driver.sbatch")

# The training node's host key changed at some point and the stale entry in
# known_hosts turns every connection into a warning; a throwaway known-hosts
# file keeps key authentication working and the log readable.
SSH = ["ssh", "-p", TRAIN_PORT, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
       "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
       "-o", "ConnectTimeout=30"]

CKPT_RE = re.compile(r"^step(\d+)-([0-9.]+)\.ckpt$")


def log(msg):
    print(f"[driver {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, timeout=300):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def remote_list():
    """(name, size, mtime) for every step*.ckpt on the training node."""
    cmd = SSH + [TRAIN_NODE, f"stat -c '%n %s %Y' {shlex.quote(CKPT_DIR)}/step*.ckpt 2>/dev/null"]
    rc, out, err = run(cmd)
    if rc != 0 and not out:
        log(f"listing failed (rc={rc}): {err[:200]}")
        return None
    entries = []
    for line in out.splitlines():
        parts = line.rsplit(" ", 2)
        if len(parts) != 3:
            continue
        name = os.path.basename(parts[0])
        if CKPT_RE.match(name):
            entries.append((name, int(parts[1]), int(parts[2])))
    return entries


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"seen": {}, "processed": {}, "inflight": None, "cycles": 0}


def save_state(state, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)


def job_active(job_id):
    rc, out, _ = run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], timeout=60)
    return rc == 0 and out.strip() != ""


def write_skip_record(rec_dir, step, val_loss, ckpt_name, reason):
    path = os.path.join(rec_dir, f"step{step:07d}.json")
    if os.path.exists(path):
        return
    rec = {"step": step, "val_loss": val_loss, "status": "skipped", "reason": reason,
           "ckpt_name": ckpt_name, "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, path)


def stage(name, staging_dir):
    """Copy one checkpoint onto falas, running the copy on the training node.

    Written to a temporary name and renamed, so a copy interrupted halfway can
    never be picked up as a complete checkpoint.
    """
    src = f"{CKPT_DIR}/{name}"
    dst = os.path.join(staging_dir, name)
    tmp = os.path.join(staging_dir, f".staging.{name}")
    remote_cmd = (f"mkdir -p {shlex.quote(staging_dir)} && "
                  f"cp {shlex.quote(src)} {shlex.quote(tmp)} && "
                  f"mv {shlex.quote(tmp)} {shlex.quote(dst)} && "
                  f"stat -c %s {shlex.quote(dst)}")
    t0 = time.time()
    rc, out, err = run(SSH + [TRAIN_NODE, remote_cmd], timeout=1800)
    if rc != 0:
        log(f"staging {name} FAILED (rc={rc}): {err[:300]}")
        return None
    log(f"staged {name} ({int(out or 0) / 2**30:.1f} GB) in {time.time() - t0:.0f}s")
    return dst


def submit_eval(ckpt_path, step, val_loss, store, extra_args):
    export = (f"ALL,LIGHTNING_CKPT={ckpt_path},LIVE_EVAL_STEP={step},"
              f"LIVE_EVAL_VAL_LOSS={val_loss},LIVE_EVAL_STORE={store},"
              f"LIVE_EVAL_EXTRA={extra_args}")
    rc, out, err = run(["sbatch", "--parsable", f"--export={export}", EVAL_SBATCH], timeout=120)
    if rc != 0:
        log(f"sbatch FAILED (rc={rc}): {err[:300]}")
        return None
    job_id = out.split(";")[0].strip()
    log(f"submitted evaluation job {job_id} for step {step}")
    return job_id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--poll_s", type=int, default=180)
    ap.add_argument("--settle_s", type=int, default=120,
                    help="a checkpoint must be at least this old before it is copied")
    ap.add_argument("--catch_up", type=int, default=3,
                    help="work through a backlog up to this many checkpoints oldest first; "
                         "beyond it, jump to the newest and record the rest as skipped")
    ap.add_argument("--max_hours", type=float, default=11.5,
                    help="exit cleanly after this long so the wrapper can chain a successor")
    ap.add_argument("--backfill_dir", default=None,
                    help="a directory of already-copied checkpoints to seed the curve with")
    ap.add_argument("--eval_extra", default="",
                    help="extra arguments passed through to live_eval_one.py")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = ap.parse_args()

    store = args.store
    staging = os.path.join(store, "staging")
    rec_dir = os.path.join(store, "records")
    for d in (store, staging, rec_dir):
        os.makedirs(d, exist_ok=True)
    state_path = os.path.join(store, "driver_state.json")
    state = load_state(state_path)
    stop_path = os.path.join(store, "STOP")

    # Backfill: checkpoints already sitting on falas count as staged and ready.
    prestaged = {}
    if args.backfill_dir and os.path.isdir(args.backfill_dir):
        for name in sorted(os.listdir(args.backfill_dir)):
            if CKPT_RE.match(name):
                prestaged[name] = os.path.join(args.backfill_dir, name)
        log(f"backfill: {len(prestaged)} checkpoint(s) already on falas: {sorted(prestaged)}")

    t_start = time.time()
    log(f"store {store}; polling every {args.poll_s}s; giving up the job after {args.max_hours}h")

    while True:
        state["cycles"] = state.get("cycles", 0) + 1

        if os.path.exists(stop_path):
            log("STOP file present; exiting without chaining a successor")
            save_state(state, state_path)
            return 0

        # 1. is an evaluation still running?
        inflight = state.get("inflight")
        if inflight:
            if job_active(inflight["job"]):
                log(f"evaluation job {inflight['job']} (step {inflight['step']}) still running")
                save_state(state, state_path)
                if args.once:
                    return 0
                time.sleep(args.poll_s)
                continue
            step = inflight["step"]
            done = os.path.exists(os.path.join(rec_dir, f"step{step:07d}.json"))
            state["processed"][str(step)] = {
                "status": "done" if done else "failed",
                "job": inflight["job"], "ckpt_name": inflight["name"],
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            if not done:
                write_skip_record(rec_dir, step, inflight.get("val_loss"), inflight["name"],
                                  f"evaluation job {inflight['job']} ended without writing a result")
                log(f"evaluation job {inflight['job']} (step {step}) ended with no result; recorded as failed")
            else:
                log(f"evaluation job {inflight['job']} (step {step}) finished")
            state["inflight"] = None
            save_state(state, state_path)

        # 2. what is on the training node
        entries = remote_list()
        if entries is None:
            log("could not reach the training node this cycle; will retry")
            if args.once:
                return 1
            time.sleep(args.poll_s)
            continue

        now = time.time()
        ready = []
        seen = state["seen"]
        for name, size, mtime in entries:
            prev = seen.get(name)
            seen[name] = {"size": size, "mtime": mtime, "last_seen": now}
            stable = prev is not None and prev["size"] == size and prev["mtime"] == mtime
            if stable and (now - mtime) >= args.settle_s:
                ready.append(name)
        for name in prestaged:
            if name not in ready:
                ready.append(name)

        pending = []
        for name in ready:
            m = CKPT_RE.match(name)
            step = int(m.group(1))
            if str(step) in state["processed"]:
                continue
            if os.path.exists(os.path.join(rec_dir, f"step{step:07d}.json")):
                state["processed"][str(step)] = {"status": "done", "job": None, "ckpt_name": name}
                continue
            pending.append((step, float(m.group(2)), name))
        pending.sort()

        if not pending:
            log(f"nothing new ({len(entries)} checkpoint(s) on the node, all accounted for)")
        else:
            if len(pending) > args.catch_up:
                for step, val_loss, name in pending[:-1]:
                    write_skip_record(rec_dir, step, val_loss, name,
                                      f"a newer checkpoint (step {pending[-1][0]}) was already "
                                      "waiting when this one came up")
                    state["processed"][str(step)] = {"status": "skipped", "job": None, "ckpt_name": name}
                log(f"{len(pending)} checkpoints waiting, more than the {args.catch_up} this loop "
                    f"works through in order; jumping to step {pending[-1][0]} and recording "
                    f"{len(pending) - 1} as skipped")
                pending = pending[-1:]

            step, val_loss, name = pending[0]
            path = prestaged.get(name)
            from_backfill = path is not None
            if path is None:
                path = stage(name, staging)
            if path is None:
                state["processed"][str(step)] = {"status": "failed", "job": None, "ckpt_name": name}
                write_skip_record(rec_dir, step, val_loss, name, "the checkpoint could not be copied off the training node")
            else:
                # A backfilled checkpoint lives in the shared mirror directory,
                # not in this loop's own staging area, so the evaluation must not
                # delete it when it finishes.
                extra = args.eval_extra + (" --keep_ckpt" if from_backfill else "")
                job_id = submit_eval(path, step, val_loss, store, extra.strip())
                if job_id is None:
                    log("submission failed; leaving the checkpoint pending for the next cycle")
                else:
                    state["inflight"] = {"job": job_id, "step": step, "name": name,
                                         "val_loss": val_loss, "path": path,
                                         "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                    prestaged.pop(name, None)

        save_state(state, state_path)
        if args.once:
            return 0
        if (time.time() - t_start) > args.max_hours * 3600:
            log("reached the walltime this job budgeted; exiting so the wrapper can chain a successor")
            return 0
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
