#!/bin/bash
# One eval-ready weights file per epoch, next to a running training job.
#   snapshot_sidecar.sh <output dir> <steps per epoch> [once]
# The trainer keeps only last.ckpt (10.5 GB, full state, rewritten every epoch) and the best-by-val-loss file, and
# checkpoint selection on the validation split needs every epoch. The paper run kept full states through an
# external mirror; this keeps what selection loads: scripts_segvigen/unrekey_for_eval.py's standalone file (raw
# weights, as the published step 23405 was bridged), 2.6 GB, as <output dir>/snapshots/step<N>.pt.
# A checkpoint is read only after its mtime has been still for 90 s (Lightning writes it in place), and only when
# its step is a whole number of epochs; `once` takes whatever the newest checkpoint holds (the end of a run).
set -u
OUT="${1:?output dir}"; EPOCH="${2:?steps per epoch}"; ONCE="${3:-}"
mkdir -p "$OUT/snapshots"
seen=""
while true; do
  ck=$(ls -t "$OUT"/ckpts/last*.ckpt 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    case "$ck" in */last.ckpt) ;; *) echo "WARNING $(date): newest checkpoint is $ck, not last.ckpt — a resume would load a stale last.ckpt" ;; esac
    m=$(stat -c %Y "$ck"); age=$(( $(date +%s) - m ))
    if [ "$m" != "$seen" ] && { [ "$age" -ge 90 ] || [ -n "$ONCE" ]; }; then
      if read -r step ep < <(python scripts_segvigen/ckpt_step.py "$ck" 2>/dev/null); then
        dst="$OUT/snapshots/step$(printf '%07d' "$step").pt"
        if [ -e "$dst" ]; then seen=$m
        elif [ -n "$ONCE" ] || [ $(( step % EPOCH )) -eq 0 ]; then
          echo "$(date) snapshot step=$step epoch=$ep from $ck"
          if python scripts_segvigen/unrekey_for_eval.py "$ck" "$dst.tmp" > "$dst.log" 2>&1 && [ "$(stat -c %Y "$ck")" = "$m" ]; then
            mv "$dst.tmp" "$dst"; seen=$m; echo "$(date) wrote $dst ($(stat -c %s "$dst") bytes)"
          else
            rm -f "$dst.tmp"; echo "$(date) snapshot of step $step failed or the checkpoint changed while it was read; will retry"
          fi
        else seen=$m
        fi
      else echo "$(date) $ck not readable yet"
      fi
    fi
  fi
  [ -n "$ONCE" ] && exit 0
  sleep 120
done
