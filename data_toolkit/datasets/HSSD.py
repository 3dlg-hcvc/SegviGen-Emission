import argparse

import pandas as pd

from .common import ensure_sha256_column, foreach_instance, load_hf_csv


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--metadata_repo",
        type=str,
        default=None,
        help="Optional HuggingFace dataset repo_id containing metadata CSVs "
        "(defaults to TRELLIS-500K repo).",
    )
    parser.add_argument(
        "--hssd_repo",
        type=str,
        default="hssd/hssd-hab",
        help="HuggingFace dataset repo_id containing HSSD object files.",
    )


def get_metadata(*, metadata_repo: str = None, **kwargs) -> pd.DataFrame:
    df = load_hf_csv("HSSD.csv", repo_id=metadata_repo) if metadata_repo else load_hf_csv("HSSD.csv")
    df = ensure_sha256_column(df)
    if df.index.name == "sha256":
        df = df.reset_index()
    return df


def download(
    metadata: pd.DataFrame,
    *,
    download_root: str,
    check_only: bool = False,
    hssd_repo: str = "hssd/hssd-hab",
    **kwargs,
) -> pd.DataFrame:
    import os
    import shutil
    import tempfile
    from huggingface_hub import hf_hub_download

    records = []
    raw_root = os.path.join(download_root, "raw")
    os.makedirs(raw_root, exist_ok=True)

    for row in metadata.to_dict(orient="records"):
        sha256 = row.get("sha256")
        file_identifier = row.get("file_identifier") or row.get("fileIdentifier")
        if not sha256 or not file_identifier:
            continue

        rel_path = os.path.join("raw", file_identifier)
        out_path = os.path.join(download_root, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            records.append({"sha256": sha256, "local_path": rel_path})
            continue
        if check_only:
            continue

        try:
            cached_path = hf_hub_download(
                repo_id=hssd_repo,
                filename=str(file_identifier),
                repo_type="dataset",
            )
            with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(out_path)) as tmp:
                tmp_path = tmp.name
            shutil.copy2(cached_path, tmp_path)
            os.replace(tmp_path, out_path)
            records.append({"sha256": sha256, "local_path": rel_path})
        except Exception as e:
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            records.append({"sha256": sha256, "error": str(e)})

    return pd.DataFrame.from_records(records)


__all__ = ["add_args", "get_metadata", "foreach_instance", "download"]
import argparse

import pandas as pd

from .common import ensure_sha256_column, foreach_instance, load_hf_csv


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--metadata_repo",
        type=str,
        default=None,
        help="Optional HuggingFace dataset repo_id containing metadata CSVs "
        "(defaults to TRELLIS-500K repo).",
    )
    parser.add_argument(
        "--hssd_repo",
        type=str,
        default="hssd/hssd-hab",
        help="HuggingFace dataset repo_id containing HSSD object files.",
    )


def get_metadata(*, metadata_repo: str = None, **kwargs) -> pd.DataFrame:
    df = load_hf_csv("HSSD.csv", repo_id=metadata_repo) if metadata_repo else load_hf_csv("HSSD.csv")
    df = ensure_sha256_column(df)
    if df.index.name == "sha256":
        df = df.reset_index()
    return df


def download(
    metadata: pd.DataFrame,
    *,
    download_root: str,
    check_only: bool = False,
    hssd_repo: str = "hssd/hssd-hab",
    **kwargs,
) -> pd.DataFrame:
    import os
    import shutil
    import tempfile
    from huggingface_hub import hf_hub_download

    records = []
    raw_root = os.path.join(download_root, "raw")
    os.makedirs(raw_root, exist_ok=True)

    for row in metadata.to_dict(orient="records"):
        sha256 = row.get("sha256")
        file_identifier = row.get("file_identifier") or row.get("fileIdentifier")
        if not sha256 or not file_identifier:
            continue

        rel_path = os.path.join("raw", file_identifier)
        out_path = os.path.join(download_root, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            records.append({"sha256": sha256, "local_path": rel_path})
            continue
        if check_only:
            continue

        try:
            cached_path = hf_hub_download(
                repo_id=hssd_repo,
                filename=str(file_identifier),
                repo_type="dataset",
            )
            with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(out_path)) as tmp:
                tmp_path = tmp.name
            shutil.copy2(cached_path, tmp_path)
            os.replace(tmp_path, out_path)
            records.append({"sha256": sha256, "local_path": rel_path})
        except Exception as e:
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            records.append({"sha256": sha256, "error": str(e)})

    return pd.DataFrame.from_records(records)


__all__ = ["add_args", "get_metadata", "foreach_instance", "download"]
