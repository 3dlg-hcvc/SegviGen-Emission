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


if __name__ == "__main__":
    test_load_samples_from_parquet_first_3()
    print("ok: parquet first:3")
    test_load_samples_from_splits_json()
    print("ok: splits json split:val")
    test_load_samples_sha_csv()
    print("ok: sha csv")
    print("all passed")
