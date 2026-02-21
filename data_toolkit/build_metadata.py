import argparse
import importlib
import os
import shutil
import sys
import time

import pandas as pd

try:
    import o_voxel
except ImportError:
    o_voxel = None
from easydict import EasyDict as edict


def update_metadata(path, opt, converted_column=None, size_column=None):
    """Update metadata from records, with fallback to listing .vxz files if no records exist."""
    if not os.path.exists(path):
        return None
    timestamp = str(int(time.time()))
    try:
        os.makedirs(os.path.join(path, 'merged_records'), exist_ok=True)
        os.makedirs(os.path.join(path, 'new_records'), exist_ok=True)
    except PermissionError as e:
        metadata_path = os.path.join(path, 'metadata.csv')
        print(f"Skipping records merge for read-only path: {path}")
        if os.path.exists(metadata_path):
            return pd.read_csv(metadata_path)
        return None
    if opt.from_merged_records:
        merged_dir = os.path.join(path, 'merged_records')
        if not os.path.exists(merged_dir):
            metadata_path = os.path.join(path, 'metadata.csv')
            if os.path.exists(metadata_path):
                return pd.read_csv(metadata_path)
            return None
        df_files = [f for f in os.listdir(merged_dir) if f.endswith('.csv')]
        df_files = [f for f in df_files if int(f.split('_')[0]) >= opt.record_start]
    else:
        new_dir = os.path.join(path, 'new_records')
        if not os.path.exists(new_dir):
            metadata_path = os.path.join(path, 'metadata.csv')
            if os.path.exists(metadata_path):
                return pd.read_csv(metadata_path)
            return None
        df_files = [f for f in os.listdir(new_dir) if f.startswith('part_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df = pd.read_csv(os.path.join(path, 'new_records', f))
            if len(df) > 0:  # Only add non-empty dataframes
                df_parts.append(df)
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    if len(df_parts) > 0:
        if os.path.exists(os.path.join(path, 'metadata.csv')):
            metadata = pd.read_csv(os.path.join(path, 'metadata.csv'))
        else:
            columns = df_parts[0].columns
            metadata = pd.DataFrame(columns=columns)
        metadata.set_index('sha256', inplace=True)
        for df_part in df_parts:
            if 'sha256' in df_part.columns:
                df_part.set_index('sha256', inplace=True)
                metadata = df_part.combine_first(metadata)
        metadata.to_csv(os.path.join(path, 'metadata.csv'))
        for f in df_files:
            shutil.move(os.path.join(path, 'new_records', f), os.path.join(path, 'merged_records', f'{timestamp}_{f}'))
        return metadata
    else:
        # Fallback: if no valid records but we have converted_column, try building from .vxz files
        if converted_column is not None:
            metadata = _build_metadata_from_vxz_files(path, converted_column, size_column)
            if metadata is not None:
                print(f"Built metadata from {len(metadata)} .vxz files in {path}")
                metadata.to_csv(os.path.join(path, 'metadata.csv'), index=False)
                return metadata
        if os.path.exists(os.path.join(path, 'metadata.csv')):
            return pd.read_csv(os.path.join(path, 'metadata.csv'))
    return None


def _count_downloaded_assets(metadata, downloaded_metadata, download_root):
    if downloaded_metadata is not None and 'local_path' in downloaded_metadata.columns:
        return downloaded_metadata['local_path'].count()
    if 'local_path' in metadata.columns:
        return metadata['local_path'].count()

    file_col = None
    if 'file_identifier' in metadata.columns:
        file_col = 'file_identifier'
    elif 'fileIdentifier' in metadata.columns:
        file_col = 'fileIdentifier'

    if file_col is None:
        return 0

    raw_root = os.path.join(download_root, 'raw')
    count = 0
    for value in metadata[file_col].astype(str):
        if '://' in value or os.path.isabs(value):
            continue
        full_path = os.path.join(raw_root, value)
        if os.path.exists(full_path):
            count += 1
    return count


def _ensure_local_path(metadata, download_root):
    file_col = None
    if 'file_identifier' in metadata.columns:
        file_col = 'file_identifier'
    elif 'fileIdentifier' in metadata.columns:
        file_col = 'fileIdentifier'
    if file_col is None:
        return metadata, False

    raw_root = os.path.join(download_root, 'raw')
    if not os.path.exists(raw_root):
        return metadata, False

    metadata = metadata.copy()
    updated = False
    if 'local_path' not in metadata.columns:
        metadata['local_path'] = pd.NA
        updated = True

    needs = metadata['local_path'].isna()
    if not needs.any():
        return metadata, updated

    values = metadata.loc[needs, file_col].astype(str).tolist()
    new_values = []
    for value in values:
        if '://' in value or os.path.isabs(value):
            new_values.append(pd.NA)
            continue
        rel_path = os.path.join('raw', value)
        if os.path.exists(os.path.join(download_root, rel_path)):
            new_values.append(rel_path)
        else:
            new_values.append(pd.NA)

    if any(not pd.isna(v) for v in new_values):
        metadata.loc[needs, 'local_path'] = new_values
        updated = True
    return metadata, updated


def _build_metadata_from_vxz_files(directory, converted_column, size_column=None):
    """Fallback: build metadata by listing .vxz files without opening them."""
    if not os.path.exists(directory):
        return None
    vxz_files = [f for f in os.listdir(directory) if f.endswith('.vxz')]
    if not vxz_files:
        return None
    rows = []
    for filename in vxz_files:
        sha256 = os.path.splitext(filename)[0]
        row = {
            'sha256': sha256,
            converted_column: True,
        }
        if size_column:
            row[size_column] = pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--download_root', type=str, default=None,
                        help='Directory to save the downloaded files')
    parser.add_argument('--thumbnail_root', type=str, default=None,
                        help='Directory to save the thumbnail files')
    parser.add_argument('--render_cond_root', type=str, default=None,
                        help='Directory to save the render condition files')
    parser.add_argument('--mesh_dump_root', type=str, default=None,
                        help='Directory to save the mesh files')
    parser.add_argument('--pbr_dump_root', type=str, default=None,
                        help='Directory to save the pbr files')
    parser.add_argument('--dual_grid_root', type=str, default=None,
                        help='Directory to save the dual grid files')
    parser.add_argument('--pbr_voxel_root', type=str, default=None,
                        help='Directory to save the pbr voxel files')
    parser.add_argument('--ss_latent_root', type=str, default=None,
                        help='Directory to save the sparse structure latent files')
    parser.add_argument('--shape_latent_root', type=str, default=None,
                        help='Directory to save the shape latent files')
    parser.add_argument('--pbr_latent_root', type=str, default=None,
                        help='Directory to save the pbr latent files')
    parser.add_argument('--field', type=str, default='all',
                        help='Fields to process, separated by commas')
    parser.add_argument('--from_merged_records', action='store_true',
                        help='Build metadata from merged records')
    parser.add_argument('--record_start', type=int)
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild metadata from scratch, ignore existing metadata.')
    dataset_utils.add_args(parser)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.download_root = opt.download_root or opt.root
    opt.thumbnail_root = opt.thumbnail_root or opt.root
    opt.render_cond_root = opt.render_cond_root or opt.root
    opt.mesh_dump_root = opt.mesh_dump_root or opt.root
    opt.pbr_dump_root = opt.pbr_dump_root or opt.root
    opt.dual_grid_root = opt.dual_grid_root or opt.root
    opt.pbr_voxel_root = opt.pbr_voxel_root or opt.root
    opt.ss_latent_root = opt.ss_latent_root or opt.root
    opt.shape_latent_root = opt.shape_latent_root or opt.root
    opt.pbr_latent_root = opt.pbr_latent_root or opt.root

    os.makedirs(opt.root, exist_ok=True)

    opt.field = opt.field.split(',')
    
    # get file list
    if os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        print('Loading previous metadata...')
        metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv'))
    else:
        metadata = dataset_utils.get_metadata(**opt)
        metadata.to_csv(os.path.join(opt.root, 'metadata.csv'), index=False)
    metadata, metadata_updated = _ensure_local_path(metadata, opt.download_root)
    if metadata_updated:
        metadata.to_csv(os.path.join(opt.root, 'metadata.csv'), index=False)
    
    # merge downloaded
    downloaded_metadata = update_metadata(os.path.join(opt.download_root, 'raw'), opt)

    # merge thumbnails
    thumbnail_metadata = update_metadata(os.path.join(opt.thumbnail_root, 'thumbnails'), opt)
    
    # merge aesthetic scores
    aesthetic_score_metadata = update_metadata(os.path.join(opt.root, 'aesthetic_scores'), opt)
    
    # merge render conditions
    render_cond_metadata = update_metadata(os.path.join(opt.render_cond_root, 'renders_cond'), opt)

    # merge mesh dumped
    mesh_dumped_metadata = update_metadata(os.path.join(opt.mesh_dump_root, 'mesh_dumps'), opt)
        
    # merge pbr dumped
    pbr_dumped_metadata = update_metadata(os.path.join(opt.pbr_dump_root, 'pbr_dumps'), opt)
    
    # merge asset stats
    asset_stats_metadata = update_metadata(os.path.join(opt.root, 'asset_stats'), opt)
        
    # merge dual grid
    dual_grid_resolutions = []
    for dir in os.listdir(opt.dual_grid_root):
        if os.path.isdir(os.path.join(opt.dual_grid_root, dir)) and dir.startswith('dual_grid_'):
            dual_grid_resolutions.append(int(dir.split('_')[-1]))
    dual_grid_metadata = {}
    for res in dual_grid_resolutions:
        dual_grid_dir = os.path.join(opt.dual_grid_root, f'dual_grid_{res}')
        dual_grid_metadata[res] = update_metadata(dual_grid_dir, opt, 
                                                   converted_column='dual_grid_converted',
                                                   size_column='dual_grid_size')
    
    # merge pbr voxelized
    pbr_voxel_resolutions = []
    for dir in os.listdir(opt.pbr_voxel_root):
        if os.path.isdir(os.path.join(opt.pbr_voxel_root, dir)) and dir.startswith('pbr_voxels_'):
            pbr_voxel_resolutions.append(int(dir.split('_')[-1]))
    pbr_voxel_metadata = {}
    for res in pbr_voxel_resolutions:
        pbr_voxel_metadata[res] = update_metadata(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_{res}'), opt,
                                                   converted_column='pbr_voxelized',
                                                   size_column='num_pbr_voxels')
        
    # merge ss latents
    ss_latent_models = []
    if os.path.exists(os.path.join(opt.ss_latent_root, 'ss_latents')):
        ss_latent_models = os.listdir(os.path.join(opt.ss_latent_root, 'ss_latents'))
    ss_latent_metadata = {}
    for model in ss_latent_models:
        ss_latent_metadata[model] = update_metadata(os.path.join(opt.ss_latent_root, f'ss_latents/{model}'), opt)
        
    # merge shape latents
    shape_latent_models = []
    if os.path.exists(os.path.join(opt.shape_latent_root, 'shape_latents')):
        shape_latent_models = os.listdir(os.path.join(opt.shape_latent_root, 'shape_latents'))
    shape_latent_metadata = {}
    for model in shape_latent_models:
        shape_latent_metadata[model] = update_metadata(os.path.join(opt.shape_latent_root, f'shape_latents/{model}'), opt)
        
    # merge pbr latents
    pbr_latent_models = []
    if os.path.exists(os.path.join(opt.pbr_latent_root, 'pbr_latents')):
        pbr_latent_models = os.listdir(os.path.join(opt.pbr_latent_root, 'pbr_latents'))
    pbr_latent_metadata = {}
    for model in pbr_latent_models:
        pbr_latent_metadata[model] = update_metadata(os.path.join(opt.pbr_latent_root, f'pbr_latents/{model}'), opt)

    # statistics
    num_downloaded = _count_downloaded_assets(metadata, downloaded_metadata, opt.download_root)
    with open(os.path.join(opt.root, 'statistics.txt'), 'w') as f:
        f.write('Statistics:\n')
        f.write(f'  - Number of assets: {len(metadata)}\n')
        f.write(f'  - Number of assets downloaded: {num_downloaded}\n')
        if thumbnail_metadata is not None:
            f.write(f'  - Number of assets with thumbnails: {thumbnail_metadata["thumbnailed"].sum()}\n')
        if aesthetic_score_metadata is not None:
            f.write(f'  - Number of assets with aesthetic scores: {aesthetic_score_metadata["aesthetic_score"].count()}\n')
        if render_cond_metadata is not None:
            f.write(f'  - Number of assets with render conditions: {render_cond_metadata["cond_rendered"].count()}\n')
        if mesh_dumped_metadata is not None:
            f.write(f'  - Number of assets with mesh dumped: {mesh_dumped_metadata["mesh_dumped"].sum()}\n')
        if pbr_dumped_metadata is not None:
            f.write(f'  - Number of assets with PBR dumped: {pbr_dumped_metadata["pbr_dumped"].sum()}\n')
        if asset_stats_metadata is not None:
            f.write(f'  - Number of assets with asset stats: {len(asset_stats_metadata)}\n')
        if len(dual_grid_resolutions) != 0:
            f.write(f'  - Number of assets with dual grid:\n')
            for res in dual_grid_resolutions:
                if dual_grid_metadata[res] is not None:
                    f.write(f'    - {res}: {dual_grid_metadata[res]["dual_grid_converted"].sum()}\n')
        if len(pbr_voxel_resolutions) != 0:
            f.write(f'  - Number of assets with PBR voxelization:\n')
            for res in pbr_voxel_resolutions:
                if pbr_voxel_metadata[res] is not None:
                    f.write(f'    - {res}: {pbr_voxel_metadata[res]["pbr_voxelized"].sum()}\n')
        if len(ss_latent_models) != 0:
            f.write(f'  - Number of assets with sparse structure latents:\n')
            for model in ss_latent_models:
                if ss_latent_metadata[model] is not None:
                    f.write(f'    - {model}: {ss_latent_metadata[model]["ss_latent_encoded"].sum()}\n')
        if len(shape_latent_models) != 0:
            f.write(f'  - Number of assets with shape latents:\n')
            for model in shape_latent_models:
                if shape_latent_metadata[model] is not None:
                    f.write(f'    - {model}: {shape_latent_metadata[model]["shape_latent_encoded"].sum()}\n')
        if len(pbr_latent_models) != 0:
            f.write(f'  - Number of assets with PBR latents:\n')
            for model in pbr_latent_models:
                if pbr_latent_metadata[model] is not None:
                    f.write(f'    - {model}: {pbr_latent_metadata[model]["pbr_latent_encoded"].sum()}\n')
        
    with open(os.path.join(opt.root, 'statistics.txt'), 'r') as f:
        print(f.read())