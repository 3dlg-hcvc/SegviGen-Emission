"""Coverage hook for the coverage run (step 9): with COVERAGE_PROCESS_START set to an rc file, every
Python process (including dataloader workers and subprocesses) records line coverage. Loaded through
PYTHONPATH; the code under test never imports it. The training check has the same hook in its own
sitecustomize (Python loads only the first one on the path)."""
import os

if os.environ.get("COVERAGE_PROCESS_START"):
    import coverage

    coverage.process_startup()
