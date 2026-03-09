"""
Encode emission voxels into latent representations using a trained emission VAE encoder.

Usage (analogous to encode_pbr_latent.py):

    python data_toolkit/encode_emission_latent.py TexVerse \\
        --root /data/texverse \\
        --resolution 256 \\
        --enc_model emission_vae \\
        --model_root /runs/emission_vae \\
        --ckpt best

Or using a pretrained encoder:

    python data_toolkit/encode_emission_latent.py TexVerse \\
        --root /data/texverse \\
        --resolution 256 \\
        --enc_pretrained /runs/emission_vae/ckpts/encoder_best.pt

Input:  emission_voxels_{resolution}/{sha256}.vxz
          attrs: emissive (3 channels, uint8, range [0, 255])
Output: emission_latents/{latent_name}/{sha256}.npz
          feats : float32 [N_tokens, C_latent]
          coords: uint8   [N_tokens, 3]
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
import torch
import numpy as np
import pandas as pd
import o_voxel
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis2.models as models
import trellis2.modules.sparse as sp

torch.set_grad_enabled(False)


def is_valid_sparse_tensor(tensor):
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()


def clear_cuda_error():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    import importlib
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser(
        description='Encode emission voxels into latent representations.'
    )
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory that contains metadata.csv and voxel subdirs')
    parser.add_argument('--emission_voxel_root', type=str, default=None,
                        help='Directory with emission_voxels_{res}/ subfolders (default: --root)')
    parser.add_argument('--emission_latent_root', type=str, default=None,
                        help='Directory to save emission latents (default: --root)')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Skip objects below this aesthetic score')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Voxel grid resolution that was used during voxelization')
    parser.add_argument('--enc_pretrained', type=str, default=None,
                        help='HuggingFace / local path to a pretrained emission encoder')
    parser.add_argument('--model_root', type=str, default=None,
                        help='Root directory for local model configs')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Name of the encoder model config (relative to --model_root)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint name (e.g. "best") when using --enc_model')
    parser.add_argument('--instances', type=str, default=None,
                        help='Comma-separated sha256s or path to a file with one sha256 per line')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    dataset_utils.add_args(parser)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.emission_voxel_root = opt.emission_voxel_root or opt.root
    opt.emission_latent_root = opt.emission_latent_root or opt.root

    # ---------------------------------------------------------------------------
    # Load encoder
    # ---------------------------------------------------------------------------
    if opt.enc_model is None:
        assert opt.enc_pretrained is not None, \
            "Either --enc_pretrained or --enc_model must be specified"
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}_{opt.resolution}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.enc_model.split("/")[-1]}_{opt.ckpt}_{opt.resolution}'
        cfg = edict(json.load(
            open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')
        ))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        encoder.eval()
        print(f'Loaded encoder from {ckpt_path}')

    save_dir = os.path.join(opt.emission_latent_root, 'emission_latents', latent_name)
    os.makedirs(os.path.join(save_dir, 'new_records'), exist_ok=True)

    # ---------------------------------------------------------------------------
    # Build sample list from metadata.csv
    # ---------------------------------------------------------------------------
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError(f'metadata.csv not found in {opt.root}')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')

    for extra_csv in [
        os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv'),
        os.path.join(opt.emission_voxel_root, f'emission_voxels_{opt.resolution}', 'metadata.csv'),
        os.path.join(save_dir, 'metadata.csv'),
    ]:
        if os.path.exists(extra_csv):
            metadata = metadata.combine_first(
                pd.read_csv(extra_csv).set_index('sha256')
            )

    metadata = metadata.reset_index()

    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        # Only process rows that have been emission-voxelized
        if 'emission_voxelized' in metadata.columns:
            metadata = metadata[metadata['emission_voxelized'] == True]
        # Skip already-encoded rows
        if 'emission_latent_encoded' in metadata.columns:
            metadata = metadata[metadata['emission_latent_encoded'] != True]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances) as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end   = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    # ---------------------------------------------------------------------------
    # Skip already-processed files
    # ---------------------------------------------------------------------------
    records = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex, \
         tqdm(total=len(metadata), desc='Filtering existing files') as pbar:
        def _check(sha256):
            path = os.path.join(save_dir, f'{sha256}.npz')
            if os.path.exists(path):
                coords = np.load(path)['coords']
                records.append({
                    'sha256': sha256,
                    'emission_latent_encoded': True,
                    'emission_latent_tokens': coords.shape[0],
                })
            pbar.update()
        ex.map(_check, metadata['sha256'].values)

    existing = {r['sha256'] for r in records}
    print(f'Already encoded: {len(existing)} objects')
    metadata = metadata[~metadata['sha256'].isin(existing)]
    print(f'Remaining to encode: {len(metadata)} objects')

    # ---------------------------------------------------------------------------
    # Encode
    # ---------------------------------------------------------------------------
    sha256s   = list(metadata['sha256'].values)
    load_queue = Queue(maxsize=32)

    with ThreadPoolExecutor(max_workers=32) as loader_ex, \
         ThreadPoolExecutor(max_workers=32) as saver_ex:

        def _load(sha256):
            try:
                coords, attr = o_voxel.io.read_vxz(
                    os.path.join(opt.emission_voxel_root,
                                 f'emission_voxels_{opt.resolution}',
                                 f'{sha256}.vxz'),
                    num_threads=4,
                )
                # emissive: uint8 [N, 3], scale to [-1, 1]
                feats = attr['emissive'].float() / 255.0 * 2.0 - 1.0
                x = sp.SparseTensor(
                    feats,
                    torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
                )
                load_queue.put((sha256, x))
            except Exception as exc:
                print(f'[Loader Error] {sha256}: {exc}')
                load_queue.put((sha256, None))

        loader_ex.map(_load, sha256s)

        def _save(sha256, pack):
            np.savez_compressed(os.path.join(save_dir, f'{sha256}.npz'), **pack)
            records.append({
                'sha256': sha256,
                'emission_latent_encoded': True,
                'emission_latent_tokens': pack['coords'].shape[0],
            })

        for _ in tqdm(range(len(sha256s)), desc='Encoding emission latents'):
            sha256, voxels = load_queue.get()
            if voxels is None:
                continue
            if not is_valid_sparse_tensor(voxels):
                print(f'[Skip] {sha256}: NaN/Inf in input voxels')
                continue
            try:
                z = encoder(voxels.cuda())
                torch.cuda.synchronize()
                if not torch.isfinite(z.feats).all():
                    print(f'[Skip] {sha256}: Non-finite latent')
                    clear_cuda_error()
                    continue
                pack = {
                    'feats':  z.feats.cpu().numpy().astype(np.float32),
                    'coords': z.coords[:, 1:].cpu().numpy().astype(np.uint8),
                }
                saver_ex.submit(_save, sha256, pack)
            except Exception as exc:
                print(f'[Error] {sha256}: {exc}')
                clear_cuda_error()

        saver_ex.shutdown(wait=True)

    # Save record CSV
    pd.DataFrame.from_records(records).to_csv(
        os.path.join(save_dir, 'new_records', f'part_{opt.rank}.csv'),
        index=False,
    )
    print(f'Saved {len(records)} records to {save_dir}/new_records/part_{opt.rank}.csv')
