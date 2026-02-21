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
        "--abo_base_url",
        type=str,
        default="https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/3dmodels/original",
        help="Base URL prefix for ABO 3D models (public S3).",
    )


def get_metadata(*, metadata_repo: str = None, **kwargs) -> pd.DataFrame:
    df = load_hf_csv("ABO.csv", repo_id=metadata_repo) if metadata_repo else load_hf_csv("ABO.csv")
    df = ensure_sha256_column(df)
    if df.index.name == "sha256":
        df = df.reset_index()
    return df


def download(
    metadata: pd.DataFrame,
    *,
    download_root: str,
    check_only: bool = False,
    abo_base_url: str = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/3dmodels/original",
    **kwargs,
) -> pd.DataFrame:
    """
    Download ABO .glb files into <download_root>/raw/<file_identifier>.

    Expects metadata to contain:
      - sha256
      - file_identifier (e.g. "3/B07YBH2SR3.glb")
    """
    import os
    import urllib.request
    import tempfile

    records = []
    raw_root = os.path.join(download_root, "raw")
    os.makedirs(raw_root, exist_ok=True)

    for row in metadata.to_dict(orient="records"):
        sha256 = row.get("sha256")
        file_identifier = row.get("file_identifier")
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

        url = abo_base_url.rstrip("/") + "/" + str(file_identifier).lstrip("/")
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(out_path)) as tmp:
                tmp_path = tmp.name
            urllib.request.urlretrieve(url, tmp_path)
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
