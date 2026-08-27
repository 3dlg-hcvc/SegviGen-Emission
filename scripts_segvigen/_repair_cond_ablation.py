"""Re-run the shards a conditioning-ablation arm lost, without redoing the rest.

The shared partition preempts tasks, and it preempted several of these. The
merge kept every shard that did land and the bridged checkpoint, so a repair
costs only the gap. Both arms must reach full coverage before the two are
compared: a paired delta over different shape subsets would not be a paired
delta at all.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from submit_eval_chain import submit_chain   # noqa: E402

ABL = "/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/cond_ablation"
MIR = ("/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/"
       "segvigen_256_bw_6gpu_ckpts_mirror")
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 24915
CKPT = f"{MIR}/step{STEP:07d}_snapshot.ckpt"
COMMON = f"--wall_sids_file {ABL}/val387.txt --wall_n 400"
ARMS = {"A": f"--arm A {COMMON}",
        "B": f"--arm B --cond_override {ABL}/black_cond.pth {COMMON}"}

any_work = False
for name, extra in ARMS.items():
    rec_path = f"{ABL}/arm{name}/records/step{STEP:07d}.json"
    if not os.path.exists(rec_path):
        print(f"arm {name}: no record yet, nothing to repair")
        continue
    miss = json.load(open(rec_path)).get("missing_shards") or []
    if not miss:
        print(f"arm {name}: complete, no repair needed")
        continue
    any_work = True
    r = submit_chain(CKPT, STEP, store=f"{ABL}/arm{name}", n_shards=8, draws=5, quiet=True,
                     only_shards=miss, job_name=f"cond_abl_{name}r",
                     bridged_path=f"{ABL}/bridged.pt", extra_args=extra)
    print(f"arm {name}: repairing shards {miss} -> array {r['array']}, merge {r['merge']}")
if not any_work:
    print("both arms are complete")
