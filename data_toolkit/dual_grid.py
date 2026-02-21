import argparse
import importlib
import os
import pickle
import sys
import time

import numpy as np
import o_voxel
import pandas as pd
import torch

try:
    import trimesh
except ImportError:
    trimesh = None
from functools import partial

from easydict import EasyDict as edict


def _rss_mb():
    try:
        with open("/proc/self/statm", "r") as f:
            rss_pages = int(f.readline().split()[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return rss_pages * page_size / (1024 * 1024)
    except Exception:
        return None


def _dbg(msg):
    rss = _rss_mb()
    if rss is None:
        print(f"[dual_grid] {time.strftime('%H:%M:%S')} {msg}")
    else:
        print(f"[dual_grid] {time.strftime('%H:%M:%S')} {msg} | RSS={rss:.1f} MB")


def _encode_ids_to_uint8(ids):
    if ids.size == 0:
        return np.zeros((0, 1), dtype=np.uint8)
    ids = np.where(ids < 0, 0, ids).astype(np.int64)
    max_id = int(ids.max())
    if max_id <= 255:
        return ids.astype(np.uint8).reshape(-1, 1)
    if max_id <= 65535:
        lo = (ids & 0xFF).astype(np.uint8)
        hi = ((ids >> 8) & 0xFF).astype(np.uint8)
        return np.stack([lo, hi], axis=1)
    b0 = (ids & 0xFF).astype(np.uint8)
    b1 = ((ids >> 8) & 0xFF).astype(np.uint8)
    b2 = ((ids >> 16) & 0xFF).astype(np.uint8)
    b3 = ((ids >> 24) & 0xFF).astype(np.uint8)
    return np.stack([b0, b1, b2, b3], axis=1)


def _encode_labels_to_uint8(labels):
    if not labels:
        return None
    max_len = max(len(label.encode("utf-8")) for label in labels)
    if max_len == 0:
        return None
    out = np.zeros((len(labels), max_len), dtype=np.uint8)
    for idx, label in enumerate(labels):
        encoded = label.encode("utf-8")
        if not encoded:
            continue
        out[idx, :len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return out


def _encode_ids_to_uint8_matrix(ids):
    if ids.size == 0:
        return np.zeros((ids.shape[0], 0), dtype=np.uint8)
    ids = np.where(ids < 0, 0, ids).astype(np.int64)
    max_id = int(ids.max())
    if max_id <= 255:
        return ids.astype(np.uint8)
    if max_id <= 65535:
        lo = (ids & 0xFF).astype(np.uint8)
        hi = ((ids >> 8) & 0xFF).astype(np.uint8)
        return np.stack([lo, hi], axis=2).reshape(ids.shape[0], -1)
    b0 = (ids & 0xFF).astype(np.uint8)
    b1 = ((ids >> 8) & 0xFF).astype(np.uint8)
    b2 = ((ids >> 16) & 0xFF).astype(np.uint8)
    b3 = ((ids >> 24) & 0xFF).astype(np.uint8)
    return np.stack([b0, b1, b2, b3], axis=2).reshape(ids.shape[0], -1)


def _plane_box_overlap(normal, vert, maxbox):
    vmin = np.empty(3, dtype=np.float32)
    vmax = np.empty(3, dtype=np.float32)
    for i in range(3):
        if normal[i] > 0.0:
            vmin[i] = -maxbox[i] - vert[i]
            vmax[i] = maxbox[i] - vert[i]
        else:
            vmin[i] = maxbox[i] - vert[i]
            vmax[i] = -maxbox[i] - vert[i]
    if np.dot(normal, vmin) > 0.0:
        return False
    return np.dot(normal, vmax) >= 0.0


def _tri_box_overlap(box_center, box_halfsize, triverts):
    v0 = triverts[0] - box_center
    v1 = triverts[1] - box_center
    v2 = triverts[2] - box_center

    for i in range(3):
        min_v = min(v0[i], v1[i], v2[i])
        max_v = max(v0[i], v1[i], v2[i])
        if min_v > box_halfsize[i] or max_v < -box_halfsize[i]:
            return False

    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    axes = (
        np.array([0.0, -e0[2], e0[1]]),
        np.array([e0[2], 0.0, -e0[0]]),
        np.array([-e0[1], e0[0], 0.0]),
        np.array([0.0, -e1[2], e1[1]]),
        np.array([e1[2], 0.0, -e1[0]]),
        np.array([-e1[1], e1[0], 0.0]),
        np.array([0.0, -e2[2], e2[1]]),
        np.array([e2[2], 0.0, -e2[0]]),
        np.array([-e2[1], e2[0], 0.0]),
    )
    for axis in axes:
        if np.allclose(axis, 0.0):
            continue
        p0 = np.dot(v0, axis)
        p1 = np.dot(v1, axis)
        p2 = np.dot(v2, axis)
        min_p = min(p0, p1, p2)
        max_p = max(p0, p1, p2)
        rad = (box_halfsize * np.abs(axis)).sum()
        if min_p > rad or max_p < -rad:
            return False

    normal = np.cross(e0, e1)
    if not _plane_box_overlap(normal, v0, box_halfsize):
        return False

    return True


def _pad_label_lists(label_lists):
    if not label_lists:
        return None, None
    max_len = max(len(labels) for labels in label_lists)
    if max_len == 0:
        return None, None
    out = np.full((len(label_lists), max_len), -1, dtype=np.int64)
    counts = np.zeros((len(label_lists),), dtype=np.int64)
    for i, labels in enumerate(label_lists):
        n = len(labels)
        counts[i] = n
        if n:
            out[i, :n] = labels
    return out, counts


def _dual_grid_mesh(file, metadatum, mesh_dump_root, root):
    sha256 = metadatum['sha256']
    try:
        _dbg(f"{sha256}: start processing")
        pack = {'sha256': sha256}
        data = None
        seg_map = None
        fine_map = None
        seg_parts = None
        fine_parts = None
        faces_np = None
        vertices_np = None
        for res in opt.resolution:
            need_process = False
            _dbg(f"{sha256} res={res}: begin")

            # check if already processed
            if os.path.exists(os.path.join(root, f'dual_grid_{res}', f'{sha256}.vxz')):
                try:
                    info = o_voxel.io.read_vxz_info(os.path.join(root, f'dual_grid_{res}', f'{sha256}.vxz'))
                    pack[f'dual_grid_converted_{res}'] = True
                    pack[f'dual_grid_size_{res}'] = info['num_voxel']
                except Exception as e:
                    print(f'Error reading {sha256}.vxz: {e}')
                    need_process = True
            else:
                need_process = True

            # process mesh
            if need_process:
                if data is None:
                    with open(os.path.join(mesh_dump_root, 'mesh_dumps', f'{sha256}.pickle'), 'rb') as f:
                        dump = pickle.load(f)
                    start = 0
                    vertices = []
                    faces = []
                    for obj in dump['objects']:
                        if obj['vertices'].size == 0 or obj['faces'].size == 0:
                            continue
                        vertices.append(obj['vertices'])
                        faces.append(obj['faces'] + start)
                        start += len(obj['vertices'])
                    vertices = torch.from_numpy(np.concatenate(vertices, axis=0)).float()
                    faces = torch.from_numpy(np.concatenate(faces, axis=0)).long()
                    vertices_min = vertices.min(dim=0)[0]
                    vertices_max = vertices.max(dim=0)[0]
                    center = (vertices_min + vertices_max) / 2
                    scale = 0.99999 / (vertices_max - vertices_min).max()
                    vertices = (vertices - center) * scale
                    assert torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5), 'vertices out of range'
                    data = {'vertices': vertices, 'faces': faces}

                    faces_np = faces.cpu().numpy()
                    vertices_np = vertices.cpu().numpy()
                    seg_map = dump.get('segmentation_map')
                    fine_map = dump.get('fine_segmentation_map')
                    seg_parts = dump.get('segmentation_parts', {}) or {}
                    fine_parts = dump.get('fine_segmentation_parts', {}) or {}
                    if isinstance(seg_parts, dict):
                        seg_parts = {int(k): v for k, v in seg_parts.items()}
                    if isinstance(fine_parts, dict):
                        fine_parts = {int(k): v for k, v in fine_parts.items()}
                    if seg_map is not None and fine_map is not None:
                        seg_map = np.asarray(seg_map, dtype=np.int32)
                        fine_map = np.asarray(fine_map, dtype=np.int32)
                        if seg_map.shape[0] != faces_np.shape[0] or fine_map.shape[0] != faces_np.shape[0]:
                            seg_map = None
                            fine_map = None

                    if metadatum.get('segmented_local_path') and (seg_map is None or fine_map is None):
                        raise ValueError('Segmentation data missing; run dump_mesh before dual_grid.')

                # Use single-pass voxelization with integrated label tracking if labels are available
                if seg_map is not None and fine_map is not None and faces_np is not None:
                    _dbg(f"{sha256} res={res}: voxelizing with integrated label tracking")
                    (voxel_indices, dual_vertices, intersected,
                     primary_coarse_t, primary_fine_t, coarse_ids_t, fine_ids_t) = (
                        o_voxel.convert.mesh_to_flexible_dual_grid_with_labels(
                            **data,
                            coarse_labels=torch.from_numpy(seg_map),
                            fine_labels=torch.from_numpy(fine_map),
                            grid_size=res,
                            aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
                            face_weight=1.0,
                            boundary_weight=0.2,
                            regularization_weight=1e-2,
                            timing=False,
                        )
                    )

                    dual_vertices = dual_vertices * res - voxel_indices
                    assert torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1+1e-3), 'dual_vertices out of range'
                    dual_vertices = torch.clamp(dual_vertices, 0, 1)
                    dual_vertices = (dual_vertices * 255).type(torch.uint8)
                    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

                    attr = {'vertices': dual_vertices, 'intersected': intersected}

                    primary_coarse = primary_coarse_t.cpu().numpy().astype(np.int64)
                    primary_fine = primary_fine_t.cpu().numpy().astype(np.int64)
                    coarse_ids = coarse_ids_t.cpu().numpy().astype(np.int64)
                    fine_ids = fine_ids_t.cpu().numpy().astype(np.int64)

                    attr['segmentation_id'] = torch.from_numpy(_encode_ids_to_uint8(primary_coarse))
                    attr['fine_segmentation_id'] = torch.from_numpy(_encode_ids_to_uint8(primary_fine))

                    if coarse_ids.shape[1] > 0:
                        attr['segmentation_ids'] = torch.from_numpy(_encode_ids_to_uint8_matrix(coarse_ids))
                        # Count non-(-1) entries per row
                        coarse_counts = (coarse_ids >= 0).sum(axis=1).astype(np.int64)
                        attr['segmentation_ids_count'] = torch.from_numpy(_encode_ids_to_uint8(coarse_counts))
                    if fine_ids.shape[1] > 0:
                        attr['fine_segmentation_ids'] = torch.from_numpy(_encode_ids_to_uint8_matrix(fine_ids))
                        fine_counts = (fine_ids >= 0).sum(axis=1).astype(np.int64)
                        attr['fine_segmentation_ids_count'] = torch.from_numpy(_encode_ids_to_uint8(fine_counts))

                    pair = np.stack([primary_fine, primary_coarse], axis=1)
                    unique_pairs, inverse = np.unique(pair, axis=0, return_inverse=True)
                    pair_labels = []
                    for fine_id, coarse_id in unique_pairs:
                        label = None
                        if fine_id >= 0:
                            part = fine_parts.get(int(fine_id))
                            if part:
                                label = part.get("label") or part.get("name")
                        if not label and coarse_id >= 0:
                            part = seg_parts.get(int(coarse_id))
                            if part:
                                label = part.get("label") or part.get("name")
                        pair_labels.append(label or "unknown")
                    labels = [pair_labels[i] for i in inverse]
                    label_bytes = _encode_labels_to_uint8(labels)
                    if label_bytes is not None:
                        attr['segmentation_label'] = torch.from_numpy(label_bytes)
                else:
                    # No labels - use original function
                    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
                        **data,
                        grid_size=res,
                        aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
                        face_weight=1.0,
                        boundary_weight=0.2,
                        regularization_weight=1e-2,
                        timing=False,
                    )
                    dual_vertices = dual_vertices * res - voxel_indices
                    assert torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1+1e-3), 'dual_vertices out of range'
                    dual_vertices = torch.clamp(dual_vertices, 0, 1)
                    dual_vertices = (dual_vertices * 255).type(torch.uint8)
                    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
                    attr = {'vertices': dual_vertices, 'intersected': intersected}

                o_voxel.io.write_vxz(
                    os.path.join(root, f'dual_grid_{res}', f'{sha256}.vxz'),
                    voxel_indices,
                    attr,
                )

                pack[f'dual_grid_converted_{res}'] = True
                pack[f'dual_grid_size_{res}'] = len(dual_vertices)

        return pack
    except Exception as e:
        print(f'Error processing {sha256}: {e}')
        return {'sha256': sha256, 'error': str(e)}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--mesh_dump_root', type=str, default=None,
                        help='Directory to load mesh dumps')
    parser.add_argument('--dual_grid_root', type=str, default=None,
                        help='Directory to save dual grids')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--resolution', type=str, default=256)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.resolution = [int(x) for x in opt.resolution.split(',')]
    opt.mesh_dump_root = opt.mesh_dump_root or opt.root
    opt.dual_grid_root = opt.dual_grid_root or opt.root

    for res in opt.resolution:
        os.makedirs(os.path.join(opt.dual_grid_root, f'dual_grid_{res}', 'new_records'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')).set_index('sha256'))
    for res in opt.resolution:
        if os.path.exists(os.path.join(opt.dual_grid_root, f'dual_grid_{res}', 'metadata.csv')):
            dual_grid_metadata = pd.read_csv(os.path.join(opt.dual_grid_root, f'dual_grid_{res}', 'metadata.csv')).set_index('sha256')
            dual_grid_metadata = dual_grid_metadata.rename(columns={'dual_grid_converted': f'dual_grid_converted_{res}', 'dual_grid_size': f'dual_grid_size_{res}'})
            metadata = metadata.combine_first(dual_grid_metadata)
    metadata = metadata.reset_index()
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['mesh_dumped'] == True]
        mask = np.zeros(len(metadata), dtype=bool)
        for res in opt.resolution:
            if f'dual_grid_converted_{res}' in metadata.columns:
                mask |= metadata[f'dual_grid_converted_{res}'] != True
            else:
                mask[:] = True
                break
        metadata = metadata[mask]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    # filter out objects that are already processed on disk
    for res in opt.resolution:
        dual_grid_dir = os.path.join(opt.dual_grid_root, f'dual_grid_{res}')
        if not os.path.exists(dual_grid_dir):
            continue
        sha256_list = [
            os.path.splitext(f)[0]
            for f in os.listdir(dual_grid_dir)
            if f.endswith('.vxz')
        ]
        if not sha256_list:
            continue
        metadata = metadata[~metadata['sha256'].isin(sha256_list)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_dual_grid_mesh, root=opt.dual_grid_root, mesh_dump_root=opt.mesh_dump_root)
    dual_grids = dataset_utils.foreach_instance(metadata, None, func, max_workers=opt.max_workers, no_file=True, desc='Dual griding')
    if 'error' in dual_grids.columns:
        errors = dual_grids[dual_grids[f'error'].notna()]
        with open('errors.txt', 'w') as f:
            f.write('\n'.join(errors['sha256'].tolist()))
    for res in opt.resolution:
        if f'dual_grid_converted_{res}' in dual_grids.columns:
            dual_grid_metadata = dual_grids[dual_grids[f'dual_grid_converted_{res}'] == True]
            if len(dual_grid_metadata) > 0:
                dual_grid_metadata = dual_grid_metadata[['sha256', f'dual_grid_converted_{res}', f'dual_grid_size_{res}']]
                dual_grid_metadata = dual_grid_metadata.rename(columns={f'dual_grid_converted_{res}': 'dual_grid_converted', f'dual_grid_size_{res}': 'dual_grid_size'})
                dual_grid_metadata.to_csv(os.path.join(opt.dual_grid_root, f'dual_grid_{res}', 'new_records', f'part_{opt.rank}.csv'), index=False)
    
