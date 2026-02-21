from typing import Callable, Dict, Optional

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


HF_TRELLIS500K_REPO_ID = "JeffreyXiang/TRELLIS-500K"


def load_hf_csv(filename: str, *, repo_id: str = HF_TRELLIS500K_REPO_ID) -> pd.DataFrame:
    """
    Load a CSV from a HuggingFace *dataset* repo via huggingface_hub cache.
    """
    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return pd.read_csv(csv_path)


def ensure_sha256_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column naming so downstream scripts always have a 'sha256' column.
    """
    if "sha256" in df.columns:
        return df
    # Common alternates seen in some manifests
    for alt in ("sha256sum", "sha256_sum", "sha256_hash", "hash_sha256"):
        if alt in df.columns:
            return df.rename(columns={alt: "sha256"})
    raise ValueError("Metadata is missing required 'sha256' column")


def normalize_objaverse_xl_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Objaverse-XL-style metadata columns for downloading.
    """
    df = ensure_sha256_column(df.copy())
    if "fileIdentifier" not in df.columns and "file_identifier" in df.columns:
        df = df.rename(columns={"file_identifier": "fileIdentifier"})
    if "source" not in df.columns and "fileIdentifier" in df.columns:
        df["source"] = df["fileIdentifier"].astype(str).str.split("/", n=1).str[0]
    return df


def download_objaverse_xl(
    metadata: pd.DataFrame,
    *,
    download_root: str,
    check_only: bool = False,
    processes: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Download objects using objaverse.xl downloaders.
    """
    import tempfile
    import urllib.request
    from urllib.parse import quote, urlparse

    import objaverse.xl as oxl

    records = []
    raw_root = os.path.join(download_root, "raw")
    os.makedirs(raw_root, exist_ok=True)

    df = normalize_objaverse_xl_metadata(metadata)
    required = {"fileIdentifier", "source", "sha256"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for Objaverse-XL download: {sorted(missing)}")

    df = df[list(required)]

    def github_raw_info(file_identifier: str) -> Optional[Dict[str, str]]:
        if not file_identifier:
            return None
        try:
            parsed = urlparse(str(file_identifier))
            if parsed.netloc != "github.com":
                return None
            parts = parsed.path.strip("/").split("/")
            if len(parts) < 5:
                return None
            org, repo, mode, commit_hash = parts[0], parts[1], parts[2], parts[3]
            if mode not in {"blob", "raw", "tree"}:
                return None
            rel_parts = parts[4:]
            rel_path = os.path.join("raw", "github", org, repo, commit_hash, *rel_parts)
            raw_path = "/".join(rel_parts)
            raw_url = (
                f"https://raw.githubusercontent.com/{org}/{repo}/{commit_hash}/"
                f"{quote(raw_path, safe='/')}"
            )
            return {"raw_url": raw_url, "rel_path": rel_path}
        except Exception:
            return None

    github_df = df[df["source"] == "github"].copy()
    other_df = df[df["source"] != "github"].copy()

    if check_only:
        for row in github_df.to_dict(orient="records"):
            sha256 = row.get("sha256")
            file_identifier = row.get("fileIdentifier")
            if not file_identifier or not sha256:
                continue
            info = github_raw_info(file_identifier)
            if not info:
                continue
            out_path = os.path.join(download_root, info["rel_path"])
            if os.path.exists(out_path):
                records.append({"sha256": sha256, "local_path": info["rel_path"]})

        for row in other_df.to_dict(orient="records"):
            file_identifier = row.get("fileIdentifier")
            sha256 = row.get("sha256")
            if not file_identifier or not sha256:
                continue
            expected_path = os.path.join(raw_root, str(file_identifier))
            if os.path.exists(expected_path):
                rel_path = os.path.relpath(expected_path, download_root)
                records.append({"sha256": sha256, "local_path": rel_path})

        return pd.DataFrame.from_records(records)

    paths = {}

    if len(other_df) > 0:
        paths.update(
            oxl.download_objects(
                objects=other_df,
                download_dir=raw_root,
                processes=processes,
            )
        )

    for row in github_df.to_dict(orient="records"):
        sha256 = row.get("sha256")
        file_identifier = row.get("fileIdentifier")
        if not file_identifier or not sha256:
            continue
        info = github_raw_info(file_identifier)
        if not info:
            records.append({"sha256": sha256, "error": "invalid_github_url"})
            continue
        out_path = os.path.join(download_root, info["rel_path"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            records.append({"sha256": sha256, "local_path": info["rel_path"]})
            continue
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(out_path)) as tmp:
                tmp_path = tmp.name
            urllib.request.urlretrieve(info["raw_url"], tmp_path)
            os.replace(tmp_path, out_path)
            records.append({"sha256": sha256, "local_path": info["rel_path"]})
        except Exception as e:
            try:
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            records.append({"sha256": sha256, "error": str(e)})

    for row in other_df.to_dict(orient="records"):
        file_identifier = row.get("fileIdentifier")
        sha256 = row.get("sha256")
        if not file_identifier or not sha256:
            continue
        local_path = paths.get(file_identifier)
        if local_path:
            if os.path.isabs(local_path) and local_path.startswith(download_root):
                local_path = os.path.relpath(local_path, download_root)
            records.append({"sha256": sha256, "local_path": local_path})
        else:
            records.append({"sha256": sha256, "error": "missing"})

    return pd.DataFrame.from_records(records)


def _resolve_local_path(local_path: str, download_root: Optional[str]) -> Optional[str]:
    if local_path is None or (isinstance(local_path, float) and pd.isna(local_path)):
        return None
    p = os.path.expanduser(str(local_path))
    if os.path.isabs(p):
        return p
    if download_root:
        base = os.path.abspath(download_root)
        c1 = os.path.join(base, p)
        if os.path.exists(c1):
            return c1
        c2 = os.path.join(base, "raw", p)
        if os.path.exists(c2):
            return c2
    if os.path.exists(p):
        return os.path.abspath(p)
    return p


def foreach_instance(
    metadata: pd.DataFrame,
    download_root: Optional[str],
    func: Callable,
    *,
    max_workers: int = 0,
    desc: Optional[str] = None,
    no_file: bool = False,
) -> pd.DataFrame:
    """
    Apply func(file_path, metadatum, ...) over instances described by `metadata`.

    This is used by downstream processing scripts (dump_mesh/dump_pbr/render_cond/...).
    When `no_file=True`, file_path will be passed as None.
    """
    if len(metadata) == 0:
        return pd.DataFrame()

    def run_one(row: Dict) -> Optional[Dict]:
        metadatum = dict(row)
        file_path = None
        if not no_file:
            lp = metadatum.get("local_path")
            file_path = _resolve_local_path(lp, download_root)
            if not file_path or not os.path.exists(file_path):
                return {
                    "sha256": metadatum.get("sha256"),
                    "error": "missing_file:%s" % file_path,
                }
        try:
            return func(file_path, metadatum)
        except Exception as e:
            return {
                "sha256": metadatum.get("sha256"),
                "error": str(e),
            }

    records = []
    if max_workers and max_workers > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(run_one, row) for row in metadata.to_dict(orient="records")]
            for fut in as_completed(futures):
                out = fut.result()
                if out is not None:
                    records.append(out)
    else:
        for row in metadata.to_dict(orient="records"):
            out = run_one(row)
            if out is not None:
                records.append(out)

    return pd.DataFrame.from_records(records)


def download(*args, **kwargs):
    """
    Downloading is dataset/source-specific and intentionally not implemented here.
    The data_toolkit pipeline can still be used for metadata merging and processing
    if local files are already present and `local_path` is populated.
    """
    raise NotImplementedError(
        "download() is not implemented in this minimal datasets shim. "
        "Provide local files + local_path, or implement dataset-specific download()."
    )
