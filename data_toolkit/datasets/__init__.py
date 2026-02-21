"""
Dataset metadata loaders for TRELLIS data_toolkit.

The toolkit scripts (e.g. build_metadata.py) dynamically import:
  importlib.import_module(f"datasets.{SUBSET}")

So this package must live next to the toolkit scripts as:
  data_toolkit/datasets/<SUBSET>.py

Each subset module is expected to expose at least:
  - add_args(parser)
  - get_metadata(**kwargs) -> pandas.DataFrame with a 'sha256' column

Some processing scripts also call:
  - foreach_instance(...)
  - download(...)
"""
