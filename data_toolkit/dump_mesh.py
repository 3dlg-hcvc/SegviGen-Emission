import os
import shutil
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
import tempfile
import pickle
from pathlib import Path

from datasets.common import _resolve_local_path


BLENDER_LINK = 'https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.5.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def _import_pygltftoolkit():
    import sys
    external_root = Path(__file__).resolve().parents[2]
    external_root_str = str(external_root)
    if external_root_str not in sys.path:
        sys.path.insert(0, external_root_str)
    import pygltftoolkit as pygltk
    return pygltk


def _dump_mesh_with_segmentation(segmented_path, sha256, output_path):
    pygltk = _import_pygltftoolkit()
    gltf = None
    vertices = faces = seg_map = fine_map = None
    segmentation_parts = fine_segmentation_parts = None
    try:
        gltf = pygltk.load(segmented_path, annotated=True)

        vertices = np.asarray(gltf.vertices, dtype=np.float32)
        faces = np.asarray(gltf.faces, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f'Expected triangular faces for {segmented_path}, got {faces.shape}')

        seg_map = getattr(gltf, "segmentation_map", None)
        if seg_map is None or len(seg_map) == 0:
            seg_map = np.full((faces.shape[0],), -1, dtype=np.int32)
        else:
            seg_map = np.asarray(seg_map, dtype=np.int32)
        fine_map = getattr(gltf, "fine_segmentation_map", None)
        if fine_map is None or len(fine_map) == 0:
            fine_map = np.full((faces.shape[0],), -1, dtype=np.int32)
        else:
            fine_map = np.asarray(fine_map, dtype=np.int32)

        if seg_map.shape[0] != faces.shape[0] or fine_map.shape[0] != faces.shape[0]:
            raise ValueError(
                f'Segmentation map size mismatch for {segmented_path}: '
                f'faces={faces.shape[0]} seg={seg_map.shape[0]} fine={fine_map.shape[0]}'
            )

        segmentation_parts = {}
        for pid, part in getattr(gltf, "segmentation_parts", {}).items():
            segmentation_parts[int(pid)] = {"name": part.name, "label": part.label}
        fine_segmentation_parts = {}
        for pid, part in getattr(gltf, "fine_segmentation_parts", {}).items():
            fine_segmentation_parts[int(pid)] = {"name": part.name, "label": part.label}

        output = {
            "objects": [{"vertices": vertices, "faces": faces}],
            "segmentation_map": seg_map,
            "fine_segmentation_map": fine_map,
            "segmentation_parts": segmentation_parts,
            "fine_segmentation_parts": fine_segmentation_parts,
            "mesh_normalized": False,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(output, f)
    finally:
        try:
            del gltf, vertices, faces, seg_map, fine_map, segmentation_parts, fine_segmentation_parts
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass


def _dump_mesh(file_path, metadatum, root, download_root=None):
    sha256 = metadatum['sha256']
    # print(f'[dump_mesh] start {sha256} -> {file_path}', flush=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = os.path.join(tmp_dir, f'{sha256}.pickle')
        output_path = os.path.join(root, 'mesh_dumps', f'{sha256}.pickle')
        segmented_path = None
        if metadatum.get('segmented_local_path'):
            segmented_path = _resolve_local_path(metadatum.get('segmented_local_path'), download_root)
        if not segmented_path and file_path.endswith('.glb'):
            candidate = file_path[:-4] + '.segmented.glb'
            if os.path.exists(candidate):
                segmented_path = candidate

        if segmented_path and os.path.exists(segmented_path):
            _dump_mesh_with_segmentation(segmented_path, sha256, temp_path)
        else:
            if not os.path.exists(BLENDER_PATH):
                _install_blender()
            args = [
                BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'dump_mesh.py'),
                '--',
                '--object', os.path.expanduser(file_path),
                '--output_path', os.path.expanduser(temp_path)
            ]
            if file_path.endswith('.blend'):
                args.insert(1, file_path)

            call(args, stdout=DEVNULL, stderr=DEVNULL)
        
        if os.path.exists(temp_path):
            shutil.move(temp_path, output_path)
            return {'sha256': sha256, 'mesh_dumped': True}
        else:
            if os.path.exists(temp_path + '_error.txt'):
                with open(temp_path + '_error.txt', 'r') as f:
                    error_msg = f.read()
                raise ValueError(f'Failed to dump mesh. File {file_path}. Error message: {error_msg}')
            else:
                raise ValueError(f'Failed to dump mesh. File {file_path}.')

if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--download_root', type=str, default=None,
                        help='Directory to save the downloaded files')
    parser.add_argument('--mesh_dump_root', type=str, default=None,
                        help='Directory to save the mesh dumps')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    parser.add_argument('--max_objects', type=int, default=None,
                        help='Optional cap on number of objects processed per run')
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.download_root = opt.download_root or opt.root
    opt.mesh_dump_root = opt.mesh_dump_root or opt.root

    os.makedirs(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'new_records'), exist_ok=True)
    
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.download_root, 'raw', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.download_root, 'raw', 'metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'mesh_dumped' in metadata.columns:
            metadata = metadata[metadata['mesh_dumped'] != True]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    if opt.max_objects is not None:
        metadata = metadata[:opt.max_objects]
    records = []

    # filter out objects that are already processed
    sha256_list = os.listdir(os.path.join(opt.mesh_dump_root, 'mesh_dumps'))
    sha256_list = [os.path.splitext(f)[0] for f in sha256_list if f.endswith('.pickle')]
    for sha256 in sha256_list:
        records.append({'sha256': sha256, 'mesh_dumped': True})
    print(f'Found {len(sha256_list)} dumped mesh')
    metadata = metadata[~metadata['sha256'].isin(sha256_list)]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_dump_mesh, root=opt.mesh_dump_root, download_root=opt.download_root)
    mesh_dumped = dataset_utils.foreach_instance(metadata, opt.download_root, func, max_workers=opt.max_workers, desc='Dumping mesh')
    mesh_dumped = pd.concat([mesh_dumped, pd.DataFrame.from_records(records)])
    mesh_dumped.to_csv(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'new_records', f'part_{opt.rank}.csv'), index=False)
