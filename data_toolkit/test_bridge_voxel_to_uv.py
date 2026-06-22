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
    position_map[1, 0] = centers[0] * 2.0 + 0.1          # 0.1 voxel-frame away -> beyond cap
    position_map[1, 1] = centers[0] * 2.0                # masked out below
    mask = np.array([[True, True], [True, False]])
    uv, miss = resample_to_uv(coords, emission, position_map, mask, grid=256, cap_voxels=1.5)
    assert np.allclose(uv[0, 0], [1.0, 0.0, 0.0])
    assert np.allclose(uv[0, 1], [0.0, 1.0, 0.0])
    assert np.allclose(uv[1, 0], [0.0, 0.0, 0.0])        # capped
    assert np.allclose(uv[1, 1], [0.0, 0.0, 0.0])        # masked
    assert abs(miss - 1.0 / 3.0) < 1e-6                  # 1 of 3 masked-True pixels capped
