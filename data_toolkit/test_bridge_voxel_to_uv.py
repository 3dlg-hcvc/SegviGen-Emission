import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from bridge_voxel_to_uv import voxel_centers, somage_to_voxel_frame, resample_to_uv


def test_voxel_centers_range_and_formula():
    coords = np.array([[0, 0, 0], [255, 255, 255], [10, 20, 30]], dtype=np.int32)
    c = voxel_centers(coords, grid=256)
    assert np.allclose(c[0], [-0.5 + 0.5 / 256] * 3)
    assert np.allclose(c[1], [-0.5 + 255.5 / 256] * 3)
    assert c.min() > -0.5 and c.max() < 0.5


def test_somage_to_voxel_frame_is_half():
    pos = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    assert np.allclose(somage_to_voxel_frame(pos), [[-0.5, 0.0, 0.5]])


def test_resample_hits_caps_and_masks():
    # two voxels with known emission
    coords = np.array([[10, 20, 30], [100, 100, 100]], dtype=np.int32)
    emission = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    centers = voxel_centers(coords, grid=256)            # voxel frame
    # 2x2 UV grid; positions are in somage frame (= 2 * voxel frame)
    position_map = np.zeros((2, 2, 3), dtype=np.float32)
    position_map[0, 0] = centers[0] * 2.0                # exact hit -> voxel 0
    position_map[0, 1] = centers[1] * 2.0                # exact hit -> voxel 1
    position_map[1, 0] = centers[0] * 2.0 + 0.1          # 0.1 somage-frame offset (~0.087 L2 in voxel frame, >> cap)
    position_map[1, 1] = centers[0] * 2.0                # masked out below
    mask = np.array([[True, True], [True, False]])
    uv, miss = resample_to_uv(coords, emission, position_map, mask, grid=256, cap_voxels=1.5)
    assert np.allclose(uv[0, 0], [1.0, 0.0, 0.0])
    assert np.allclose(uv[0, 1], [0.0, 1.0, 0.0])
    assert np.allclose(uv[1, 0], [0.0, 0.0, 0.0])        # capped
    assert np.allclose(uv[1, 1], [0.0, 0.0, 0.0])        # masked
    assert abs(miss - 1.0 / 3.0) < 1e-6                  # 1 of 3 masked-True pixels capped


def test_resample_empty_voxels_all_miss():
    coords = np.zeros((0, 3), dtype=np.int32)
    emission = np.zeros((0, 3), dtype=np.float32)
    position_map = np.zeros((1, 2, 3), dtype=np.float32)
    mask = np.array([[True, True]])
    uv, miss = resample_to_uv(coords, emission, position_map, mask)
    assert miss == 1.0
    assert np.allclose(uv, 0.0)


import pandas as pd
import imageio.v2 as imageio
from bridge_voxel_to_uv import load_somage_geom, build_slim_metadata, bridge_one


def _write_fake_somage(path, centers_somage, H=2, W=2):
    pos01 = ((centers_somage + 1.0) / 2.0 * 65535).astype(np.uint16)  # encode [-1,1]->uint16
    position = np.zeros((H, W, 3), dtype=np.uint16)
    position[0, 0] = pos01[0]
    position[0, 1] = pos01[1]
    occupancy = np.zeros((H, W, 1), dtype=bool)
    occupancy[0, 0] = True
    occupancy[0, 1] = True
    emission_color = np.zeros((H, W, 3), dtype=np.uint8)
    np.savez(path, position=position, occupancy=occupancy, emission_color=emission_color)


def test_load_somage_geom(tmp_path):
    centers_somage = np.array([[-0.5, 0.0, 0.25], [0.1, -0.2, 0.3]], dtype=np.float32)
    ditem_dir = "000-000/" + "a" * 32
    somage_dir = tmp_path / "somages" / ditem_dir
    somage_dir.mkdir(parents=True)
    _write_fake_somage(str(somage_dir / "somage.npz"), centers_somage)
    meta = pd.DataFrame({"ditem_dir": [ditem_dir], "glb_1k_path": ["x.glb"]},
                        index=["a" * 32])
    pos, mask = load_somage_geom("a" * 32, meta, str(tmp_path / "somages"))
    assert mask.sum() == 2
    assert np.allclose(pos[0, 0], centers_somage[0], atol=1e-3)


def test_bridge_one_writes_aligned_png(tmp_path):
    coords = np.array([[10, 20, 30], [100, 100, 100]], dtype=np.int32)
    emission = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    centers = voxel_centers(coords, 256)
    sha = "b" * 32
    ditem_dir = "000-000/" + sha
    somage_dir = tmp_path / "somages" / ditem_dir
    somage_dir.mkdir(parents=True)
    _write_fake_somage(str(somage_dir / "somage.npz"), (centers * 2).astype(np.float32))
    npz_dir = tmp_path / "pred_voxels"; npz_dir.mkdir()
    np.savez(str(npz_dir / f"{sha}.npz"), coords=coords, pred=emission)
    meta = pd.DataFrame({"ditem_dir": [ditem_dir], "glb_1k_path": ["x.glb"]}, index=[sha])
    out_root = tmp_path / "uv_pred"
    miss = bridge_one(str(npz_dir / f"{sha}.npz"), "pred", meta,
                      str(tmp_path / "somages"), str(out_root), cap_voxels=1.5)
    png = out_root / sha / "pred_emission.png"
    assert png.exists()
    img = imageio.imread(str(png)).astype(np.float32) / 255.0
    assert img[0, 0, 0] > 0.9 and img[0, 1, 1] > 0.9
    assert miss == 0.0
