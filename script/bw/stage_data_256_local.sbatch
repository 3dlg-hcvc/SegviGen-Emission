#!/bin/bash
#SBATCH -J stage_data_256_local
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/log/stage_data_256_local_%j.log
# PREP ONLY -- NOT authorized for submission until the owner's go (board
# lightning_port, 2026-08-26). CPU-only by design so it coexists with the
# 6-GPU training job (257933) already running on this node; --gres=gpu:0
# means it does not compete for a GPU slot.
set -x
bash /3dlg-falas/project/omages/lightgen/trellis2_lightning_probe/script/bw/stage_data_256_local.sh
