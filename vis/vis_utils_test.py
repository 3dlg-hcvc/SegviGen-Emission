"""Smoke tests for vis/vis_utils.py — run as a plain script."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vis.vis_utils import SampleRecord, load_samples

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "lightgen_test"))


def test_load_samples_from_parquet_first_3():
    recs = load_samples(
        data_root=DATA_ROOT,
        resolution=256,
        source=os.path.join(DATA_ROOT, "df_10samples.parquet"),
        select="first:3",
        glb_root=None,
        require=("emission_voxel_path",),
    )
    assert len(recs) == 3, f"expected 3, got {len(recs)}"
    for r in recs:
        assert isinstance(r, SampleRecord)
        assert os.path.exists(r.emission_voxel_path), r.emission_voxel_path


def test_load_samples_from_splits_json():
    splits = os.path.join(DATA_ROOT, "data_splits_10sample_as1k.json")
    if not os.path.exists(splits):
        print("skip: splits file missing")
        return
    recs = load_samples(
        data_root=DATA_ROOT,
        resolution=256,
        source=splits,
        select="split:val",
        require=("emission_voxel_path",),
    )
    assert len(recs) >= 1, "val split should have records"


def test_load_samples_sha_csv():
    recs_all = load_samples(
        data_root=DATA_ROOT,
        resolution=256,
        source=os.path.join(DATA_ROOT, "df_10samples.parquet"),
        select="all",
    )
    assert len(recs_all) == 10, f"expected 10 samples, got {len(recs_all)}"
    first_two = ",".join(r.sha for r in recs_all[:2])
    recs = load_samples(
        data_root=DATA_ROOT,
        resolution=256,
        source="ignored",
        select=f"sha:{first_two}",
    )
    assert len(recs) == 2, f"expected 2, got {len(recs)}"


def test_read_emission_voxel():
    recs = load_samples(
        data_root=DATA_ROOT,
        resolution=256,
        source=os.path.join(DATA_ROOT, "df_10samples.parquet"),
        select="first:1",
    )
    from vis.vis_utils import read_emission_voxel
    coords, feats, res = read_emission_voxel(recs[0].emission_voxel_path)
    assert coords.ndim == 2 and coords.shape[1] == 3, coords.shape
    assert feats.ndim == 2 and feats.shape[1] == 3, feats.shape
    assert feats.min() >= -1.0 and feats.max() <= 1.0, (feats.min(), feats.max())
    assert res == 256, f"expected 256, got {res}"
    assert coords.shape[0] == feats.shape[0]


def test_emission_to_srgb_u8():
    from vis.vis_utils import emission_to_srgb_u8
    import torch
    # black (-1, -1, -1) → 0; white (+1, +1, +1) → 255.
    black = emission_to_srgb_u8(torch.tensor([[-1.0, -1.0, -1.0]]))
    white = emission_to_srgb_u8(torch.tensor([[1.0, 1.0, 1.0]]))
    assert black.tolist() == [[0, 0, 0]], black.tolist()
    assert white.tolist() == [[255, 255, 255]], white.tolist()


def test_compute_emission_l1_zero_for_equal_inputs():
    from vis.vis_utils import compute_emission_l1
    import torch
    coords = torch.zeros((4, 3), dtype=torch.int32)
    feats = torch.tensor([[0.1, 0.2, 0.3]] * 4)
    assert compute_emission_l1(coords, feats, coords, feats) == 0.0


def test_resolve_experiment_run_dir():
    from vis.vis_utils import resolve_experiment
    # Any existing run dir works — pick the overfit one that's been around longest.
    run_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "emission_vae_10sample_overfit")
    if not os.path.isdir(run_dir):
        print("skip: run_dir missing")
        return
    h = resolve_experiment("vae", run_dir)
    assert h.label == "vae"
    assert os.path.exists(h.ckpt_path), h.ckpt_path
    assert h.trainer_name in ("EmissionVaeTrainer", "EmissionPbrFinetuneTrainer"), h.trainer_name


def test_resolve_experiment_ckpt_file():
    from vis.vis_utils import resolve_experiment
    run_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "emission_vae_10sample_overfit")
    if not os.path.isdir(run_dir):
        print("skip: run_dir missing")
        return
    ckpt = os.path.join(run_dir, "ckpts", "last.ckpt")
    if not os.path.exists(ckpt):
        print("skip: last.ckpt missing")
        return
    h = resolve_experiment("vae", ckpt)
    assert h.ckpt_path == os.path.abspath(ckpt)
    assert h.run_dir == os.path.abspath(run_dir)


if __name__ == "__main__":
    test_load_samples_from_parquet_first_3()
    print("ok: parquet first:3")
    test_load_samples_from_splits_json()
    print("ok: splits json split:val")
    test_load_samples_sha_csv()
    print("ok: sha csv")
    test_read_emission_voxel()
    print("ok: read_emission_voxel")
    test_emission_to_srgb_u8()
    print("ok: emission_to_srgb_u8")
    test_compute_emission_l1_zero_for_equal_inputs()
    print("ok: compute_emission_l1 zero")
    test_resolve_experiment_run_dir()
    print("ok: resolve_experiment run_dir")
    test_resolve_experiment_ckpt_file()
    print("ok: resolve_experiment ckpt file")
    print("all passed")
