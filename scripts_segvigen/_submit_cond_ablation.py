"""Submit the conditioning ablation: the same checkpoint, twice, one arm with the
normal image conditioning and one with the embedding of a plain black image used
for every shape. Seeds, sampler, shape set and shard layout are identical, so the
two arms differ in exactly one thing.

The arms live in their own stores, so their claim files and records cannot be
mistaken by the live loop's slot logic for work it should be doing.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from submit_eval_chain import submit_chain    # noqa: E402

ABL = "/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/cond_ablation"
MIRROR = ("/3dlg-falas/project/omages/lightgen/segvigen_emissive/outputs/"
          "segvigen_256_bw_6gpu_ckpts_mirror")
VAL387 = f"{ABL}/val387.txt"
BLACK = f"{ABL}/black_cond.pth"
SHARED_BRIDGE = f"{ABL}/bridged.pt"

step = int(sys.argv[1])
ckpt = f"{MIRROR}/step{step:07d}_snapshot.ckpt"
assert os.path.exists(ckpt), ckpt

# Paint panels for every shape in both arms: which 30 rows the page shows is
# decided from the delta distribution AFTER both arms land, so a second pass just
# to make pictures would be wasted GPU.
common_extra = f"--wall_sids_file {VAL387} --wall_n 400"
arms = [
    ("A", f"{ABL}/armA", f"--arm A {common_extra}"),
    ("B", f"{ABL}/armB", f"--arm B --cond_override {BLACK} {common_extra}"),
]
for name, store, extra in arms:
    r = submit_chain(ckpt, step, store=store, n_shards=8, val_loss=None,
                     draws=5, quiet=True, job_name=f"cond_abl_{name}",
                     bridged_path=SHARED_BRIDGE, extra_args=extra)
    print(f"arm {name}  bridge {r['bridge']}  array {r['array']}  merge {r['merge']}  "
          f"store {store}")
