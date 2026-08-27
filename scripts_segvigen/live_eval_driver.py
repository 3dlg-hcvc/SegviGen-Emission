"""Snapshot every training epoch and evaluate it.

The standing loop behind board job live_eval_loop. It runs as a small CPU job
that outlives any chat session and does only bookkeeping: the work itself is one
GPU job at a time (live_eval_one.py).

WHY IT SNAPSHOTS RATHER THAN WATCHING FOR NEW CHECKPOINT FILES (owner
correction, 2026-08-26): the training run keeps its best checkpoint only. On an
epoch whose validation loss is WORSE than the best so far, it writes no named
checkpoint at all, so a loop triggered by new files would silently drop exactly
the epochs a training curve has to show. Confirmed on this run within the hour:
epoch 6 finished at step 4530 with no file of its own, while the run's own
"last" pointer had already moved to it.

So the trigger is the run's last pointer. Every cycle the driver reads that
pointer's global_step out of the checkpoint's metadata, which costs a fraction
of a second and never touches the tensors (scripts_segvigen/ckpt_step.py), and
whenever the step has advanced it copies the file into the mirror under its own
name, step<NNNNNNN>_snapshot.ckpt. Every epoch therefore lands on the curve
whichever way the validation loss went, and the mirror accumulates the run's
full per-epoch history. Named best-checkpoint files, when they do appear, are
recorded as markers on the curve and are also snapshotted if the mirror does not
already hold that epoch; they are no longer the trigger.

DISK: one snapshot per epoch, about 10.5GB each, roughly every 25 minutes. That
is about 600GB a day of accumulation, deliberate and owner-endorsed (the
per-epoch history is worth having and can be pruned later). Nothing here ever
deletes a snapshot.

WHAT EVERY EPOCH GETS (owner decision, 2026-08-26): the whole validation split,
387 shapes, five draws each, sharded eight ways. There is no cheaper screening
tier any more. Ten shapes could not separate neighbouring epochs, and a curve
built on them would move with which ten shapes were drawn rather than with the
weights; the ten shapes survive only as the visualisation wall, where panels are
painted, and their IoU decides nothing.

The eight shards are an ARRAY of independent single-GPU tasks, never one job
asking for eight cards at once: a job wanting eight cards waits for eight to be
free at the same instant, while an array starts what the pool can take now and
the rest join as cards open, which also lets several checkpoints share the pool.
submit_eval_chain.py builds the chain; this driver only decides what to submit.

ONE CHECKPOINT AT A TIME, and its eight shards count as that one unit.
Priority for the slot:
  1. the NEWEST snapshot with no record at all,
  2. otherwise the OLDEST snapshot whose record says it was skipped.
Whenever a newer epoch arrives while older ones are still unevaluated, the
newest wins and the rest are recorded as skipped; since every snapshot is kept
on falas, nothing is lost, and the idle-slot backfill fills them in afterwards.
A backfilled point is marked as such: late, not missing.

Ending it: write a file named STOP in the store directory. The driver finishes
its current cycle, exits, and its sbatch wrapper does not chain a successor.

Usage (the sbatch wrapper is live_eval_driver.sbatch):
    python scripts_segvigen/live_eval_driver.py \
        --store /3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/live_eval
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
NODE_PROBE = "/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe"
DEFAULT_STORE = "/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/live_eval"
DEFAULT_MIRROR = ("/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/"
                  "segvigen_256_bw_6gpu_ckpts_mirror")
VAL_DIR = "/3dlg-jupiter-project/lightgen/segvigen_emissive/dataset_direct/val_72k"
sys.path.insert(0, HERE)
from submit_eval_chain import submit_chain   # noqa: E402
STEPS_PER_EPOCH = 755          # verified: the run's last pointer read 4530 at epoch 6

# The training node's host key changed at some point and the stale known_hosts
# entry turns every connection into a warning; a throwaway known-hosts file keeps
# key authentication working and the log readable.
SSH = ["ssh", "-p", TRAIN_PORT, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
       "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
       "-o", "ConnectTimeout=30"]

TOPK_RE = re.compile(r"^step(\d+)-([0-9.]+)\.ckpt$")
LAST_RE = re.compile(r"^last(-v\d+)?\.ckpt$")
SNAP_RE = re.compile(r"^step(\d+)_snapshot\.ckpt$")


def log(msg):
    print(f"[driver {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, timeout=300):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def epoch_of(step):
    return int(round(step / STEPS_PER_EPOCH))


# ------------------------------------------------------------- training node
def remote_list():
    """(name, size, mtime) for every .ckpt on the training node, or None."""
    cmd = SSH + [TRAIN_NODE, f"stat -c '%n %s %Y' {shlex.quote(CKPT_DIR)}/*.ckpt 2>/dev/null"]
    rc, out, err = run(cmd)
    if rc != 0 and not out:
        log(f"listing failed (rc={rc}): {err[:200]}")
        return None
    entries = []
    for line in out.splitlines():
        parts = line.rsplit(" ", 2)
        if len(parts) == 3:
            entries.append((os.path.basename(parts[0]), int(parts[1]), int(parts[2])))
    return entries


def remote_step(path):
    """The global_step inside a checkpoint on the training node, or None.

    Reads the checkpoint's pickled metadata only. A file that is still being
    written is not a readable archive yet, so a failure here doubles as the
    "not finished" signal and the caller simply tries again next cycle.
    """
    cmd = SSH + [TRAIN_NODE,
                 f"cd {shlex.quote(NODE_PROBE)} && /usr/bin/python3 "
                 f"scripts_segvigen/ckpt_step.py {shlex.quote(path)}"]
    rc, out, err = run(cmd, timeout=180)
    if rc != 0:
        return None, (err or out)[:160]
    try:
        return int(out.split()[0]), None
    except (ValueError, IndexError):
        return None, f"unparsable output {out[:80]!r}"


def snapshot(src_name, step, mirror):
    """Copy one checkpoint off the training node into the mirror as this
    epoch's snapshot. The copy runs ON the training node, which mounts the
    mirror, so it is one hop rather than a pull and a push.

    Written under a temporary name, VERIFIED by reading the step back out of the
    copy, and only then renamed. That verification is what makes a snapshot of a
    live, overwritable file safe: it proves the archive is complete and that the
    step did not move underneath the copy.
    """
    dst = os.path.join(mirror, f"step{step:07d}_snapshot.ckpt")
    tmp = os.path.join(mirror, f".snapshot.step{step:07d}.ckpt")
    src = f"{CKPT_DIR}/{src_name}"
    t0 = time.time()
    rc, out, err = run(SSH + [TRAIN_NODE,
                              f"mkdir -p {shlex.quote(mirror)} && "
                              f"cp {shlex.quote(src)} {shlex.quote(tmp)} && "
                              f"stat -c %s {shlex.quote(tmp)}"], timeout=2400)
    if rc != 0:
        log(f"snapshot of {src_name} at step {step} FAILED to copy (rc={rc}): {err[:200]}")
        run(SSH + [TRAIN_NODE, f"rm -f {shlex.quote(tmp)}"], timeout=120)
        return None
    copied_step, why = remote_step(tmp)
    if copied_step != step:
        log(f"snapshot of {src_name} REJECTED: the copy reads step {copied_step} "
            f"({why or 'no reason given'}), expected {step}; discarding it and retrying later")
        run(SSH + [TRAIN_NODE, f"rm -f {shlex.quote(tmp)}"], timeout=120)
        return None
    rc, _, err = run(SSH + [TRAIN_NODE, f"mv {shlex.quote(tmp)} {shlex.quote(dst)}"], timeout=300)
    if rc != 0:
        log(f"snapshot of {src_name} could not be renamed into place: {err[:200]}")
        return None
    log(f"snapshotted epoch {epoch_of(step)} (step {step}) from {src_name} "
        f"in {time.time() - t0:.0f}s -> {os.path.basename(dst)} "
        f"({int(out or 0) / 2**30:.1f} GB)")
    return dst


# ------------------------------------------------------------- store
DEFAULT_STATE = {"last_seen": {}, "processed": {}, "inflight": None, "cycles": 0,
                 "last_snapshot_step": None, "attempts": {}, "repairs": {}}


def load_state(path):
    """Missing keys are filled from the defaults rather than trusted to be
    there: this file outlives the code that wrote it, and an older driver's
    state must not crash a newer one on its first cycle."""
    state = dict(DEFAULT_STATE)
    if os.path.exists(path):
        try:
            with open(path) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                state.update({k: v for k, v in loaded.items() if k in DEFAULT_STATE})
            else:
                log("driver_state.json did not hold an object; starting a fresh one")
        except ValueError:
            log("driver_state.json was unreadable; starting a fresh one")
    return state


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def record_path(rec_dir, step, tier=None):
    """One record per checkpoint. `tier` is accepted and ignored so older
    callers (the one-off backlog drain script) keep working."""
    return os.path.join(rec_dir, f"step{step:07d}.json")


def read_record(rec_dir, step):
    path = record_path(rec_dir, step)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


def record_status(rec_dir, step):
    """'ok', 'skipped', 'failed', or None when the checkpoint has no record."""
    r = read_record(rec_dir, step)
    return None if r is None else r.get("status", "ok")


def missing_shards(rec_dir, step):
    """Shard indices a checkpoint's record says never arrived. A shared
    partition preempts a task now and then, and the merge records the gap
    rather than averaging a smaller denominator; this is what repairs it."""
    r = read_record(rec_dir, step)
    return list(r.get("missing_shards") or []) if r else []


def write_skip_record(rec_dir, step, tier_or_reason, reason=None, val_loss=None):
    """Record that an epoch was passed over. The snapshot itself is kept, so this
    is a note to the idle-slot backfill, not a loss."""
    if reason is None:
        reason = tier_or_reason
    path = record_path(rec_dir, step)
    if os.path.exists(path):
        return
    save_json({"step": step, "status": "skipped", "reason": reason, "val_loss": val_loss,
               "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, path)


def mirror_snapshots(mirror):
    """{step: path} for every checkpoint the mirror holds, snapshots and named
    best-checkpoint files alike."""
    out = {}
    if not os.path.isdir(mirror):
        return out
    for name in sorted(os.listdir(mirror)):
        m = SNAP_RE.match(name) or TOPK_RE.match(name)
        if m:
            out.setdefault(int(m.group(1)), os.path.join(mirror, name))
    return out


def update_best_markers(store, entries):
    """Named best-checkpoint files become markers on the curve, not triggers."""
    path = os.path.join(store, "best_markers.json")
    markers = {}
    if os.path.exists(path):
        try:
            markers = json.load(open(path))
        except ValueError:
            markers = {}
    changed = False
    for name, _size, _mtime in entries:
        m = TOPK_RE.match(name)
        if m and str(int(m.group(1))) not in markers:
            markers[str(int(m.group(1)))] = {"val_loss": float(m.group(2)), "file": name,
                                             "first_seen": time.strftime("%Y-%m-%d %H:%M:%S")}
            changed = True
    if changed:
        save_json(markers, path)
        log(f"best-checkpoint markers now cover steps {sorted(int(k) for k in markers)}")
    return markers


def live_claims(store):
    """Steps another submitter has already queued and that SLURM still knows
    about. Claims whose jobs are gone are cleared, so a crashed submitter cannot
    block a checkpoint forever."""
    claim_dir = os.path.join(store, "inflight")
    if not os.path.isdir(claim_dir):
        return set()
    live = set()
    for name in sorted(os.listdir(claim_dir)):
        if not name.startswith("step") or not name.endswith(".json"):
            continue
        path = os.path.join(claim_dir, name)
        try:
            c = json.load(open(path))
        except (ValueError, OSError):
            continue
        if job_active(c.get("array")) or job_active(c.get("merge")):
            live.add(int(c["step"]))
        else:
            try:
                os.remove(path)
            except OSError:
                pass
    return live


def job_active(job_id):
    if not job_id:
        return False
    rc, out, _ = run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], timeout=60)
    return rc == 0 and out.strip() != ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--mirror", default=DEFAULT_MIRROR,
                    help="where snapshots accumulate; this IS the run's per-epoch history")
    ap.add_argument("--poll_s", type=int, default=180)
    ap.add_argument("--settle_s", type=int, default=90,
                    help="the last pointer must be this old before it is read and copied")
    ap.add_argument("--n_shards", type=int, default=8,
                    help="how many independent single-GPU tasks share one checkpoint's split")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--max_retries", type=int, default=1,
                    help="how many times to re-run a checkpoint whose evaluation failed, "
                         "before leaving the failure on the record")
    ap.add_argument("--max_hours", type=float, default=11.5,
                    help="exit cleanly after this long so the wrapper can chain a successor")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    ap.add_argument("--no_submit", action="store_true",
                    help="do everything except submit jobs; for checking the loop's decisions")
    args = ap.parse_args()

    store, mirror = args.store, args.mirror
    rec_dir = os.path.join(store, "records")
    for d in (store, rec_dir, mirror):
        os.makedirs(d, exist_ok=True)
    state_path = os.path.join(store, "driver_state.json")
    state = load_state(state_path)
    stop_path = os.path.join(store, "STOP")

    t_start = time.time()
    log(f"store {store}")
    log(f"mirror {mirror} (one snapshot per epoch, about 10.5GB each, kept)")
    log(f"polling every {args.poll_s}s; {args.n_shards} shards per epoch; "
        f"giving up the job after {args.max_hours}h")

    while True:
        state["cycles"] = state.get("cycles", 0) + 1
        if os.path.exists(stop_path):
            log("STOP file present; exiting without chaining a successor")
            save_json(state, state_path)
            return 0

        # ---- 1. resolve anything still running
        #
        # A running evaluation must NOT stop this cycle here. Snapshotting has to
        # keep going while one runs: a full-validation job holds the slot for
        # about seventy minutes, three epochs pass in that time, and the run
        # overwrites its own last pointer, so an epoch missed during that window
        # is gone for good. Only SUBMISSION is one-at-a-time, and that is gated
        # further down, after the snapshots are taken.
        inflight = state.get("inflight")
        if inflight and (job_active(inflight["array"]) or job_active(inflight["merge"])):
            log(f"step {inflight['step']}: {inflight['n_shards']} shards "
                f"(array {inflight['array']}) still in flight; still watching for new epochs")
        elif inflight:
            step = inflight["step"]
            status = record_status(rec_dir, step)
            state["processed"][str(step)] = {
                "status": status or "failed", "array": inflight["array"],
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            if status is None:
                write_skip_record(rec_dir, step,
                                  f"the evaluation of this epoch (array {inflight['array']}) "
                                  "ended without writing a record")
                log(f"step {step}: the evaluation ended with no record; it will be retried "
                    "when the slot is idle")
            else:
                log(f"step {step}: evaluation finished, recorded as {status}")
            state["inflight"] = None
            save_json(state, state_path)

        # ---- 2. what the training node holds
        entries = remote_list()
        if entries is None:
            log("could not reach the training node this cycle; will retry")
            if args.once:
                return 1
            time.sleep(args.poll_s)
            continue
        update_best_markers(store, entries)

        # ---- 3. snapshot the run's last pointer if it has moved on
        lasts = [(n, s, m) for n, s, m in entries if LAST_RE.match(n)]
        snapshots = mirror_snapshots(mirror)
        now = time.time()
        if lasts:
            name, size, mtime = max(lasts, key=lambda e: e[2])   # the live pointer
            prev = state["last_seen"].get(name)
            state["last_seen"][name] = {"size": size, "mtime": mtime}
            stable = prev is not None and prev["size"] == size and prev["mtime"] == mtime
            if stable and (now - mtime) >= args.settle_s:
                step, why = remote_step(f"{CKPT_DIR}/{name}")
                if step is None:
                    log(f"{name} is not readable yet ({why}); trying again next cycle")
                elif step in snapshots:
                    pass                      # this epoch is already in the mirror
                else:
                    prev_step = state.get("last_snapshot_step")
                    if prev_step is not None:
                        # Epochs that came and went while nothing was watching cannot be
                        # recovered: the run overwrites its own last pointer. Say so on
                        # the curve instead of leaving an unexplained straight line.
                        missed = range(prev_step + STEPS_PER_EPOCH, step, STEPS_PER_EPOCH)
                        for ms in missed:
                            if ms not in snapshots:
                                write_skip_record(rec_dir, ms,
                                                  "the loop was not watching when this epoch was "
                                                  "written, and the run keeps no file for it")
                    if args.no_submit:
                        log(f"would snapshot epoch {epoch_of(step)} (step {step}) from {name} "
                            "(snapshotting is off)")
                        path = None
                    else:
                        path = snapshot(name, step, mirror)
                    if path:
                        state["last_snapshot_step"] = step
                        snapshots[step] = path
                        save_json(state, state_path)

        # ---- 4. a named best checkpoint whose epoch the mirror is missing
        for name, _s, _m in entries:
            m = TOPK_RE.match(name)
            if m and int(m.group(1)) not in snapshots:
                st = int(m.group(1))
                if args.no_submit:
                    log(f"would copy the named best checkpoint {name} (step {st}) into the "
                        "mirror (snapshotting is off)")
                    break
                p = snapshot(name, st, mirror)
                if p:
                    snapshots[st] = p
                break                          # at most one extra copy per cycle

        # ---- 5. decide what to run. One CHECKPOINT at a time; its shards count
        # as that one unit. Newest unevaluated epoch first, so the leading edge of
        # the curve is always current; everything it jumps over is recorded as
        # skipped and filled in later, because every snapshot is kept.
        if state.get("inflight"):
            save_json(state, state_path)
            if args.once:
                return 0
            time.sleep(args.poll_s)
            continue

        claimed = live_claims(store)
        if claimed:
            log(f"already being evaluated by another submitter: {sorted(claimed)}")
        statuses = {st: record_status(rec_dir, st) for st in snapshots}
        statuses = {st: v for st, v in statuses.items() if st not in claimed}
        never = sorted(st for st, v in statuses.items() if v is None)
        retryable = sorted(st for st, v in statuses.items() if v in ("skipped", "failed"))
        retryable = [st for st in retryable
                     if state.get("attempts", {}).get(str(st), 0) <= args.max_retries]

        incomplete = sorted(st for st in snapshots
                            if statuses.get(st) == "ok" and missing_shards(rec_dir, st)
                            and state.get("repairs", {}).get(str(st), 0) <= args.max_retries)

        choice, backfilled, repair = None, False, None
        if never:
            newest = never[-1]
            for st in never[:-1]:
                write_skip_record(rec_dir, st,
                                  f"a newer epoch (step {newest}) arrived before this one "
                                  "reached the front of the queue; it is kept on disk and "
                                  "will be filled in when the slot is idle")
                log(f"step {st} passed over in favour of step {newest}; queued for backfill")
            choice = newest
        elif incomplete:
            # Cheapest useful work: one checkpoint on the curve is standing on a
            # partial split. Re-run only the shards that went missing.
            choice = incomplete[0]
            repair = missing_shards(rec_dir, choice)
        elif retryable:
            # Idle slot: fill in the oldest epoch that was passed over, so the
            # curve becomes complete rather than permanently gapped.
            choice = retryable[0]
            backfilled = True

        if choice is None:
            log(f"nothing to run ({len(snapshots)} epoch(s) in the mirror, "
                f"{len(claimed)} in flight elsewhere, the rest evaluated)")
        elif args.no_submit:
            what = (f"re-run shards {repair} of" if repair
                    else f"evaluate across {args.n_shards} shards")
            log(f"would {what} step {choice}"
                + (" (backfill)" if backfilled else "") + " (submission is off)")
        else:
            markers = {}
            mpath = os.path.join(store, "best_markers.json")
            if os.path.exists(mpath):
                try:
                    markers = json.load(open(mpath))
                except ValueError:
                    markers = {}
            val_loss = markers.get(str(choice), {}).get("val_loss")
            if backfilled:
                # Clear the skip note so the merge writes the real record in its place.
                try:
                    os.remove(record_path(rec_dir, choice))
                except OSError:
                    pass
            try:
                r = submit_chain(snapshots[choice], choice, store=store,
                                 n_shards=args.n_shards, val_loss=val_loss,
                                 backfilled=backfilled, draws=args.draws, quiet=True,
                                 only_shards=repair)
            except SystemExit as e:
                log(f"could not submit step {choice}: {e}")
                r = None
            if r:
                state.setdefault("attempts", {})
                state.setdefault("repairs", {})
                if repair:
                    state["repairs"][str(choice)] = state["repairs"].get(str(choice), 0) + 1
                else:
                    state["attempts"][str(choice)] = state["attempts"].get(str(choice), 0) + 1
                state["inflight"] = {"step": choice, "array": r["array"], "merge": r["merge"],
                                     "bridge": r["bridge"], "n_shards": args.n_shards,
                                     "backfilled": backfilled,
                                     "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                what = (f"re-ran shards {repair}" if repair
                        else f"submitted {args.n_shards} shards")
                log(f"step {choice} (epoch {epoch_of(choice)}): {what} as array "
                    f"{r['array']}, merge {r['merge']}"
                    + (" [backfill]" if backfilled else "")
                    + (" [repair]" if repair else ""))

        save_json(state, state_path)
        if args.once:
            return 0
        if (time.time() - t_start) > args.max_hours * 3600:
            log("reached the walltime this job budgeted; exiting so the wrapper can chain a successor")
            return 0
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
