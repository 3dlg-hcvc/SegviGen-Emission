#!/usr/bin/env bash
# Smoke pilot: one AB+C pair on shard 0 of world_size=8, limited to 50 samples.
# Use this to validate Blender install, GLB visibility, and the tar handoff
# *before* fan-out via prep_74k_chain.sh.
set -euo pipefail

cd "$(dirname "$0")/../.."

RANK=0
WORLD_SIZE=8
LIMIT=50

AB_JOBID=$(./script/star2/prep_74k_stage_ab.sh ${RANK} ${WORLD_SIZE} ${LIMIT} | tail -n1)
echo "AB jobid: ${AB_JOBID}"

C_JOBID=$(./script/star2/prep_74k_stage_c.sh ${RANK} ${WORLD_SIZE} ${AB_JOBID} | tail -n1)
echo "C jobid:  ${C_JOBID}"

echo
echo "Smoke pilot submitted:"
echo "  AB: ${AB_JOBID}"
echo "  C : ${C_JOBID}  (depends on AB)"
echo
echo "Watch with:  ssh star2 squeue -u dya78"
