import argparse

import pandas as pd

from .common import ensure_sha256_column, foreach_instance, load_hf_csv, download_objaverse_xl


def add_args(parser: argparse.ArgumentParser):
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


def get_metadata(*, metadata_repo: str = None, **kwargs) -> pd.DataFrame:
    df = (
        load_hf_csv("SketchfabPicked.csv", repo_id=metadata_repo)
        if metadata_repo
        else load_hf_csv("SketchfabPicked.csv")
    )
    df = ensure_sha256_column(df)
    if df.index.name == "sha256":
        df = df.reset_index()
    return df


def download(
    metadata: pd.DataFrame,
    *,
    download_root: str,
    check_only: bool = False,
    processes: int = None,
    **kwargs,
) -> pd.DataFrame:
    return download_objaverse_xl(
        metadata,
        download_root=download_root,
        check_only=check_only,
        processes=processes,
    )


__all__ = ["add_args", "get_metadata", "foreach_instance", "download"]
