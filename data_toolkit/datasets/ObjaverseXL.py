import argparse

import pandas as pd

from .common import ensure_sha256_column, foreach_instance, load_hf_csv, download_objaverse_xl


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["sketchfab", "github"],
        help="ObjaverseXL source to use (sketchfab or github)",
    )
    parser.add_argument(
        "--metadata_repo",
        type=str,
        default=None,
        help="Optional HuggingFace dataset repo_id containing metadata CSVs "
        "(defaults to TRELLIS-500K repo).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of parallel processes for downloading (Objaverse-XL backend).",
    )


def get_metadata(*, source: str, metadata_repo: str = None, **kwargs) -> pd.DataFrame:
    filename = f"ObjaverseXL_{source}.csv"
    df = load_hf_csv(filename, repo_id=metadata_repo) if metadata_repo else load_hf_csv(filename)
    df = ensure_sha256_column(df)
    if df.index.name == "sha256":
        df = df.reset_index()
    return df


def download(
    metadata: pd.DataFrame,
    *,
    source: str,
    download_root: str,
    check_only: bool = False,
    processes: int = None,
    **kwargs,
) -> pd.DataFrame:
    md = metadata.copy()
    if "source" not in md.columns:
        md["source"] = source
    else:
        md["source"] = md["source"].fillna(source)
    return download_objaverse_xl(
        md,
        download_root=download_root,
        check_only=check_only,
        processes=processes,
    )


__all__ = ["add_args", "get_metadata", "foreach_instance", "download"]
