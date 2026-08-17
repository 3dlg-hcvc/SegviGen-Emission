"""Regression + behaviour test for the PBR-latent concat added to
EmissionPbrFinetuneTrainer.

Why this is a unit test and not a 20-step training diff: train.py sets NO seed
(no seed_everything, no manual_seed anywhere), and the VAE encodes with
sample_posterior=True, so two runs of the same config differ by construction.
A loss-trace diff could not separate "the patch changed behaviour" from noise.
Here the seed is pinned around each call, so the comparison is exact.

The old implementation is loaded straight out of git (HEAD version of the file)
and run side by side with the working-tree one.

Usage:
    conda run --no-capture-output -n trellis2 \
        python data_toolkit/verify_albedo2emission_v3/test_trainer_concat.py
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile

import torch

REPO = "/local-scratch/localhome/dya78/code/lightgen/TRELLIS2"
sys.path.insert(0, REPO)

from trellis2.modules import sparse as sp  # noqa: E402

REL = "trellis2/trainers/vae/emission_pbr_finetune.py"
LAT, EMI = 32, 3
N_VOX, N_TOK = 500, 40


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class StubEncoder(torch.nn.Module):
    """Deterministic-given-a-seed stand-in for SparseUnetVaeEncoder."""

    def forward(self, x, sample_posterior=False, return_raw=False):
        # One token per 12.5 voxels, fixed — the trainer never inspects the mapping.
        mean = torch.randn(N_TOK, LAT, device=x.feats.device, dtype=x.feats.dtype)
        logvar = torch.zeros(N_TOK, LAT, device=x.feats.device, dtype=x.feats.dtype)
        z_feats = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample_posterior else mean
        coords = torch.zeros(N_TOK, 4, dtype=torch.int32, device=x.feats.device)
        coords[:, 1] = torch.arange(N_TOK, dtype=torch.int32, device=x.feats.device)
        z = sp.SparseTensor(z_feats, coords)
        return (z, mean, logvar) if return_raw else z


class StubDecoder(torch.nn.Module):
    """Records what it was handed, returns a 6-channel prediction."""

    def __init__(self):
        super().__init__()
        self.seen_channels = None
        self.seen_dtype = None
        self.seen_feats = None

    def forward(self, z):
        self.seen_channels = z.feats.shape[-1]
        self.seen_dtype = z.feats.dtype
        self.seen_feats = z.feats.detach().clone()
        out = torch.zeros(N_VOX, 6, device=z.feats.device, dtype=torch.float32)
        return sp.SparseTensor(out, torch.zeros(N_VOX, 4, dtype=torch.int32, device=z.feats.device))


def make_trainer(cls, decoder):
    t = cls.__new__(cls)
    t.loss_type = "l1"
    t.lambda_kl = 1e-7
    t.emission_channels = EMI
    t.training_models = {"encoder": StubEncoder(), "decoder": decoder}
    t.models = t.training_models
    return t


def make_batch(dtype=torch.float32):
    x_feats = torch.randn(N_VOX, 6, dtype=dtype)
    x_coords = torch.zeros(N_VOX, 4, dtype=torch.int32)
    x = sp.SparseTensor(x_feats, x_coords)
    pbr_coords = torch.zeros(N_TOK, 4, dtype=torch.int32)
    pbr_coords[:, 1] = torch.arange(N_TOK, dtype=torch.int32)
    pbr = sp.SparseTensor(torch.randn(N_TOK, LAT, dtype=torch.float32), pbr_coords)
    return x, pbr


def run(trainer, x, pbr_latent, seed=1234):
    torch.manual_seed(seed)
    if pbr_latent is None:
        terms, _ = trainer.training_losses(x)
    else:
        terms, _ = trainer.training_losses(x, pbr_latent=pbr_latent)
    return {k: float(v) for k, v in terms.items()}


def main() -> int:
    failures = []

    # ---- load the HEAD version of the trainer next to the working-tree one ----
    # It uses package-relative imports, which cannot resolve when the file is
    # loaded standalone, so rewrite them to absolute. Order matters: '...' first,
    # or the '..' rule would eat its leading two dots.
    old_src = subprocess.check_output(["git", "-C", REPO, "show", f"HEAD:{REL}"], text=True)
    old_src = re.sub(r"^from \.\.\.", "from trellis2.", old_src, flags=re.M)
    old_src = re.sub(r"^from \.\.", "from trellis2.trainers.", old_src, flags=re.M)
    with tempfile.TemporaryDirectory() as td:
        old_path = os.path.join(td, "emission_pbr_finetune_old.py")
        with open(old_path, "w") as fh:
            fh.write(old_src)
        old_mod = load_module(old_path, "epf_old")
    OldT = old_mod.EmissionPbrFinetuneTrainer

    # The working-tree version loads through the normal package path.
    from trellis2.trainers.vae.emission_pbr_finetune import EmissionPbrFinetuneTrainer as NewT

    assert OldT is not NewT, "old and new resolved to the same class"

    x, pbr = make_batch()

    # ---- 1. REGRESSION: with no pbr_latent, new == old, exactly ----
    dec_old, dec_new = StubDecoder(), StubDecoder()
    r_old = run(make_trainer(OldT, dec_old), x, None)
    r_new = run(make_trainer(NewT, dec_new), x, None)
    if r_old != r_new:
        failures.append(f"v2 path changed: old={r_old} new={r_new}")
    if dec_old.seen_channels != LAT or dec_new.seen_channels != LAT:
        failures.append(f"v2 path decoder width: old={dec_old.seen_channels} new={dec_new.seen_channels}")
    if dec_new.seen_feats is not None and not torch.equal(dec_old.seen_feats, dec_new.seen_feats):
        failures.append("v2 path fed the decoder different features")
    print(f"[1] v2 regression: old={r_old['loss']:.10f} new={r_new['loss']:.10f} "
          f"decoder width {dec_new.seen_channels}")

    # ---- 2. The OLD trainer silently swallows pbr_latent (why the patch is needed) ----
    dec_swallow = StubDecoder()
    old_t = make_trainer(OldT, dec_swallow)
    torch.manual_seed(1234)
    old_t.training_losses(x, pbr_latent=pbr)  # no error, no warning
    if dec_swallow.seen_channels != LAT:
        failures.append("expected the OLD trainer to ignore pbr_latent entirely")
    print(f"[2] old trainer given pbr_latent -> decoder still saw {dec_swallow.seen_channels} ch "
          f"(silently ignored — this is the trap the patch closes)")

    # ---- 3. NEW BEHAVIOUR: concat, emission first ----
    dec_cat = StubDecoder()
    new_t = make_trainer(NewT, dec_cat)
    run(new_t, x, pbr)
    if dec_cat.seen_channels != 2 * LAT:
        failures.append(f"expected decoder to see {2*LAT} ch, got {dec_cat.seen_channels}")
    elif not torch.equal(dec_cat.seen_feats[:, LAT:], pbr.feats.to(dec_cat.seen_feats.dtype)):
        failures.append("PBR half is not in columns 32:64 — concat order is wrong")
    print(f"[3] concat: decoder saw {dec_cat.seen_channels} ch, PBR occupies cols {LAT}:{2*LAT}")

    # ---- 4. DTYPE: fp32 npz latent must not promote an fp16 decoder ----
    dec_dt = StubDecoder()
    t_dt = make_trainer(NewT, dec_dt)
    x16, pbr32 = make_batch(dtype=torch.float16)
    run(t_dt, x16, pbr32)
    if dec_dt.seen_dtype != torch.float16:
        failures.append(f"dtype promoted to {dec_dt.seen_dtype}; the .to(z.feats.dtype) cast is not working")
    print(f"[4] dtype: z fp16 + pbr_latent fp32 -> decoder saw {dec_dt.seen_dtype}")

    # naive cat would have promoted — show the counterfactual so the test documents itself
    naive = torch.cat([torch.zeros(4, 2, dtype=torch.float16), torch.zeros(4, 2, dtype=torch.float32)], -1)
    print(f"    (an uncast torch.cat would have produced {naive.dtype})")

    # ---- 5. COORD GUARD fires on a mismatch ----
    bad_coords = pbr.coords.clone()
    bad_coords[0, 1] += 7
    bad = sp.SparseTensor(pbr.feats, bad_coords)
    try:
        run(make_trainer(NewT, StubDecoder()), x, bad)
        failures.append("coord guard did NOT fire on mismatched coords")
        print("[5] coord guard: NOT FIRED")
    except AssertionError:
        print("[5] coord guard: fired on mismatched coords, as intended")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: v2 path bit-identical; concat, dtype cast and coord guard all behave")
    return 0


if __name__ == "__main__":
    sys.exit(main())
