#!/bin/bash
# Snapshot the `_agentic` DiT's last.ckpt every EVERY steps while job 249207 trains on cs-venus-19.
#
# Why: configs/gen/emission_dit_pbr2emission_74k_v3_agentic.yaml keeps save_top_k=1 (best by
# val/loss) and overwrites last.ckpt every 2000 steps (~20 min at 0.55 s/step), so nothing of the
# training trajectory survives for validation-split checkpoint selection unless it is copied out
# inside that 20-minute window. Decision 2026-08-25 (Dongchen): keep a snapshot every 10k steps.
#
# How: the trainer logs `[ckpt-timing] SAVE_START step=N ...` before saving (best first, then last)
# and `[ckpt-timing] RESUME_AFTER_SAVE step=N gap=..s` once ALL saves for that step are complete
# (measured on 249207: step 6000, last.ckpt mtime 11:20 > best 11:19, gap 21.9 s). So after
# `RESUME_AFTER_SAVE step=N` with N % EVERY == 0, last.ckpt IS step N — cp it on the node's own
# NVMe first (seconds; leaves the overwrite window), then rsync to jupiter and sha256 both ends.
#
# Submit from the workstation with the script on stdin (no shared-home assumption, no --export):
#   ssh star2 'sbatch' < TRELLIS2/script/star2/dit_agentic_snapshot_last_venus19.sbatch
# CPU-only, 2 cores: no GPU, no partition (star2 rule), no --export (star2 kills those jobs).
#SBATCH -J dit_agentic_snap
#SBATCH -w cs-venus-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=14:00:00
#SBATCH --output=/localscratch/dya78/%x-%j.out
set -u

TRAIN_JOB=249207
RUN=/localscratch/dya78/TRELLIS.2-lightning/outputs/emission_dit_pbr2emission_74k_v3_agentic
LOG=/localscratch/dya78/emission_dit_pbr2emission_74k_v3_agentic_venus19-${TRAIN_JOB}.out
SNAP=$RUN/snaps
JUP=/3dlg-jupiter-project/lightgen/outputs/emission_dit_pbr2emission_74k_v3_agentic
EVERY=10000
MAX=50000
POLL=20          # seconds between log checks
SQUEUE_EVERY=30  # poll squeue only every POLL*SQUEUE_EVERY s (10 min) — never tight-loop it

mkdir -p "$SNAP" "$JUP/ckpts"
if [ -f "$RUN/config.yaml" ] && [ ! -f "$JUP/config.yaml" ]; then
    cp "$RUN/config.yaml" "$JUP/config.yaml" && echo "[snap] config.yaml -> jupiter"
fi
echo "[snap] start $(date '+%F %T') host=$(hostname) run=$RUN every=$EVERY max=$MAX"

snap_one() {  # $1 = step
    local S=$1 NAME T M SZ_BEFORE SZ_AFTER H_LOCAL H_JUP
    NAME=$(printf 'last-step%07d.ckpt' "$S")
    if [ -f "$SNAP/$NAME.jup.done" ]; then return 0; fi
    if [ ! -f "$SNAP/$NAME.done" ]; then
        # last.ckpt must be newer than this step's SAVE_START (guards a stale file after a requeue)
        T=$(grep -E "SAVE_START step=$S " "$LOG" | tail -1 | grep -oE 't=[0-9]+' | cut -d= -f2)
        M=$(stat -c %Y "$RUN/ckpts/last.ckpt" 2>/dev/null || echo 0)
        if [ -z "${T:-}" ] || [ "$M" -lt "$T" ]; then
            echo "[snap] step=$S: last.ckpt mtime=$M < save t=${T:-?}, waiting"; return 1
        fi
        SZ_BEFORE=$(stat -c %s "$RUN/ckpts/last.ckpt")
        echo "[snap] step=$S: cp last.ckpt ($SZ_BEFORE B, mtime $M) -> $SNAP/$NAME  $(date '+%T')"
        cp "$RUN/ckpts/last.ckpt" "$SNAP/$NAME.tmp" || { echo "[snap] cp FAILED"; rm -f "$SNAP/$NAME.tmp"; return 1; }
        # the source must be untouched across the copy (next overwrite is ~20 min out, but prove it):
        # same mtime and size as validated before the cp, and the copy is byte-complete.
        SZ_AFTER=$(stat -c %s "$RUN/ckpts/last.ckpt")
        M_AFTER=$(stat -c %Y "$RUN/ckpts/last.ckpt")
        if [ "$SZ_AFTER" != "$SZ_BEFORE" ] || [ "$M_AFTER" -ne "$M" ] \
           || [ "$(stat -c %s "$SNAP/$NAME.tmp")" != "$SZ_BEFORE" ]; then
            echo "[snap] step=$S: last.ckpt changed under the copy (size $SZ_BEFORE->$SZ_AFTER, mtime $M->$M_AFTER) — discarding"
            rm -f "$SNAP/$NAME.tmp"; return 1
        fi
        mv "$SNAP/$NAME.tmp" "$SNAP/$NAME"
        (cd "$SNAP" && sha256sum "$NAME" > "$NAME.sha256")
        touch "$SNAP/$NAME.done"
        echo "[snap] step=$S: local snapshot done $(date '+%T') $(cat "$SNAP/$NAME.sha256")"
    fi
    echo "[snap] step=$S: rsync -> jupiter $(date '+%T')"
    rsync -h --partial --inplace "$SNAP/$NAME" "$JUP/ckpts/$NAME" || { echo "[snap] rsync FAILED"; return 1; }
    H_LOCAL=$(cut -d' ' -f1 "$SNAP/$NAME.sha256")
    H_JUP=$(sha256sum "$JUP/ckpts/$NAME" | cut -d' ' -f1)
    if [ "$H_LOCAL" = "$H_JUP" ]; then
        echo "$H_JUP  $NAME" > "$SNAP/$NAME.jup.sha256"
        touch "$SNAP/$NAME.jup.done"
        echo "[snap] step=$S: jupiter copy VERIFIED $H_JUP $(date '+%T')"
    else
        echo "[snap] step=$S: SHA MISMATCH local=$H_LOCAL jupiter=$H_JUP — will retry"
        rm -f "$JUP/ckpts/$NAME"; return 1
    fi
}

i=0
while true; do
    # every completed save step so far (multiples of EVERY), oldest first — catches up after a hiccup
    for S in $(grep -oE 'RESUME_AFTER_SAVE step=[0-9]+' "$LOG" 2>/dev/null | grep -oE '[0-9]+$' | sort -un); do
        [ $((S % EVERY)) -eq 0 ] || continue
        snap_one "$S" || true
    done
    if [ -f "$SNAP/$(printf 'last-step%07d.ckpt' "$MAX").jup.done" ]; then
        echo "[snap] step $MAX on jupiter — done $(date '+%F %T')"; break
    fi
    i=$((i + 1))
    if [ $((i % SQUEUE_EVERY)) -eq 0 ]; then
        if ! squeue -j "$TRAIN_JOB" -h -o %T 2>/dev/null | grep -qE 'RUNNING|PENDING|REQUEUED|CONFIGURING'; then
            echo "[snap] training job $TRAIN_JOB is gone $(date '+%F %T'); one last pass then exit"
            sleep 60
            for S in $(grep -oE 'RESUME_AFTER_SAVE step=[0-9]+' "$LOG" 2>/dev/null | grep -oE '[0-9]+$' | sort -un); do
                [ $((S % EVERY)) -eq 0 ] || continue
                snap_one "$S" || true
            done
            break
        fi
    fi
    sleep "$POLL"
done
echo "[snap] exit $(date '+%F %T')"
