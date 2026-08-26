#!/bin/bash
#SBATCH -J stage_data_256_local
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/stage_data_256_local_%j.log
# Split into phase a (emission+shape+pbr, ~13GB, minutes -- authorized to
# run immediately) and phase b (cond_fp16, 77GB -- only meaningful once
# job 257957's fp32->fp16 conversion is complete). CPU-only (gres:0) so it
# coexists with the 6-GPU training job (257933/257886 lineage) already
# running on this node.
#
# Submit phase a now:  sbatch --export=PHASE=a stage_data_256_local.sbatch
# Submit phase b later: sbatch --export=PHASE=b stage_data_256_local.sbatch
# Bare submission (e.g. the launcher's self-heal call) runs both:
#   sbatch stage_data_256_local.sbatch   # PHASE defaults to "all"
set -x
: "${PHASE:=all}"
bash /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/script/bw/stage_data_256_local.sh --phase "${PHASE}"
