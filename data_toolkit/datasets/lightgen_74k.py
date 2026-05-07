"""Minimal dataset module for the 74k emissive subset.

Used by data_toolkit/voxelize_pbr.py (and similar scripts) which dynamically
import `datasets.<name>` from sys.argv[1]. The 74k subset's GLBs already live
on /3dlg-falas, so download/metadata-fetch helpers from `common` are not
needed here — only `foreach_instance` is exposed.
"""
from __future__ import annotations
import argparse

from .common import foreach_instance


def add_args(parser: argparse.ArgumentParser):
    """No dataset-specific CLI flags; the 74k metadata.csv is already on disk."""
    pass


__all__ = ["add_args", "foreach_instance"]
