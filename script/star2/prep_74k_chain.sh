#!/usr/bin/env bash
# Submit all 16 sbatch jobs (8 AB + 8 C) for the full 74k fan-out.
# AB jobs are independent across shards; each C job depends only on its own AB.
set -euo pipefail

cd "$(dirname "$0")/../.."

WORLD_SIZE=8
echo "shard  AB_jobid    C_jobid"
echo "-----  ---------   ---------"
for RANK in $(seq 0 $((WORLD_SIZE-1))); do
    AB=$(./script/star2/prep_74k_stage_ab.sh ${RANK} ${WORLD_SIZE} | tail -n1)
    C=$(./script/star2/prep_74k_stage_c.sh ${RANK} ${WORLD_SIZE} ${AB} | tail -n1)
    printf "%5d  %-9s   %-9s\n" "${RANK}" "${AB}" "${C}"
done
