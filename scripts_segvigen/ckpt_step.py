"""Read a lightning checkpoint's global_step without loading its tensors.

The live evaluation loop has to ask "has the running checkpoint advanced
another epoch?" every few minutes, against a 10.5GB file on the training
node's local disk. Loading that file to answer costs minutes of that node's
memory bandwidth while it is training on six GPUs, so this reads the step out
of the checkpoint's metadata instead.

A torch checkpoint is a zip archive whose `data.pkl` member holds the pickled
object graph; every tensor's bytes live in SEPARATE members that the pickle
refers to by persistent id. Unpickling `data.pkl` with a persistent_load that
hands back a placeholder therefore reads a few hundred kilobytes and touches
none of the tensor data, and the top-level dict's own scalars (global_step,
epoch, the callback state) come through as real values.

Anything the pickle names that is not a plain container is replaced by a
permissive stub, so this never imports torch, lightning, or any project class,
and cannot execute code from the checkpoint.

Usage:
    python scripts_segvigen/ckpt_step.py /path/to/last.ckpt
prints one line: "<global_step> <epoch>", or exits non-zero with the reason.
"""
import pickle
import sys
import zipfile


class _Stub:
    """Stands in for any class the pickle names. Permissive on purpose: pickle
    may call it, set state on it, or append to it, and none of that should
    raise, because the only fields this module reads are plain scalars."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Stub()

    def __setstate__(self, state):
        pass

    def append(self, *a, **k):
        pass

    def extend(self, *a, **k):
        pass

    def __setitem__(self, *a, **k):
        pass


_SAFE = {
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("builtins", "dict"), ("builtins", "list"), ("builtins", "set"),
    ("builtins", "tuple"), ("builtins", "frozenset"), ("builtins", "bytearray"),
    ("builtins", "complex"),
}


class _MetaUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        return _Stub()

    def find_class(self, module, name):
        if (module, name) in _SAFE:
            return super().find_class(module, name)
        return _Stub


def read_meta(path):
    """Return the checkpoint's top-level dict with tensors replaced by stubs."""
    with zipfile.ZipFile(path) as z:
        member = next((n for n in z.namelist() if n.endswith("data.pkl")), None)
        if member is None:
            raise ValueError(f"{path}: no data.pkl member; not a torch zip checkpoint")
        with z.open(member) as f:
            return _MetaUnpickler(f).load()


def global_step(path):
    """The optimizer step this checkpoint was written at, or None."""
    meta = read_meta(path)
    if not isinstance(meta, dict):
        return None
    v = meta.get("global_step")
    return int(v) if isinstance(v, int) else None


def step_and_epoch(path):
    meta = read_meta(path)
    if not isinstance(meta, dict):
        return None, None
    gs, ep = meta.get("global_step"), meta.get("epoch")
    return (int(gs) if isinstance(gs, int) else None,
            int(ep) if isinstance(ep, int) else None)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        gs, ep = step_and_epoch(sys.argv[1])
    except Exception as e:
        sys.exit(f"ERROR {type(e).__name__}: {e}")
    if gs is None:
        sys.exit("ERROR no global_step in this checkpoint")
    print(f"{gs} {ep if ep is not None else -1}")
