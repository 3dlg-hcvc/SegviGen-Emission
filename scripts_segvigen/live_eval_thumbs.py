"""Fast voxel-projection thumbnails for the live per-checkpoint evaluation loop.

Why a projection and not a render: this loop runs once per training checkpoint,
roughly every 25 minutes, and must never become the slow step. The box-render
bake pipeline (bpy_rebake plus a Cycles render) costs minutes per panel and needs
its own job; one checkpoint here needs 70 panels (10 shapes x [geometry, ground
truth, 5 draws]). This module paints the decoded voxels straight into an image
with numpy, at a few tens of milliseconds a panel, with no Blender, no unwrap,
and no bake step that can silently produce a plausible black panel.

What it gives up: shading, the object's own colour, and any sub-voxel detail.
What it keeps, which is what this loop is for: where the model puts emission on
the object, next to where the ground truth puts it, at a glance.

Geometry is drawn in neutral grey and emissive voxels in warm orange, on a
transparent background so the same file reads on a light or a dark page.

Every panel of one shape shares a ShapeView: the same camera, the same
normalisation, and the same depth ordering, computed once. That is what makes
seven panels of a shape cost barely more than one, and it is also what makes the
panels comparable cell by cell. The camera is fixed across shapes as well, so no
panel is ever flattered by a per-panel view.
"""
import numpy as np

# Fixed camera for every panel this loop ever draws. Azimuth/elevation in
# degrees; UP_AXIS is the index of the voxel axis that points up (1, verified
# against real decoded shapes: it stands cakes, swords and cars upright).
AZIMUTH_DEG = 35.0
ELEVATION_DEG = 22.0
UP_AXIS = 1

SS = 2                             # supersample factor, area-downsampled at the end
SPLAT = 2                          # voxel footprint in supersampled pixels
GEOM_RGB = (0.545, 0.561, 0.588)   # neutral grey, readable on light and dark
EMIT_RGB = (1.0, 0.478, 0.176)     # warm orange
SHADE_LO, SHADE_HI = 0.55, 1.15    # depth shading range, near = bright
EMIT_SHADE_FLOOR = 0.85            # a lit region stays lit even at the back


def _rotate(coords, up_axis=None, az_deg=None, el_deg=None):
    """(N,3) voxel coords -> (screen_x, screen_y, depth), depth larger = nearer.

    The camera constants are read here, not bound as default arguments, so a
    caller that overrides the module globals actually changes the view (an
    earlier version bound them at def time and silently ignored overrides).
    """
    up_axis = UP_AXIS if up_axis is None else up_axis
    az_deg = AZIMUTH_DEG if az_deg is None else az_deg
    el_deg = ELEVATION_DEG if el_deg is None else el_deg
    u = coords[:, up_axis].astype(np.float32)
    others = [i for i in range(3) if i != up_axis]
    a = coords[:, others[0]].astype(np.float32)
    b = coords[:, others[1]].astype(np.float32)

    az, el = np.radians(az_deg), np.radians(el_deg)
    sx = a * np.cos(az) - b * np.sin(az)
    horiz = a * np.sin(az) + b * np.cos(az)
    sy = -(u * np.cos(el) - horiz * np.sin(el))   # image y grows downward
    depth = u * np.sin(el) + horiz * np.cos(el)
    return sx, sy, depth


class ShapeView:
    """One shape's projection, reusable for every panel of that shape."""

    def __init__(self, coords, size=192, margin_frac=0.06):
        coords = np.asarray(coords)
        self.size = size
        self.big = size * SS
        self.empty = coords.size == 0
        if self.empty:
            return

        sx, sy, depth = _rotate(coords)
        lo_x, hi_x = float(sx.min()), float(sx.max())
        lo_y, hi_y = float(sy.min()), float(sy.max())
        span = max(hi_x - lo_x, hi_y - lo_y, 1e-6)
        scale = self.big * (1.0 - 2.0 * margin_frac) / span
        cx, cy = 0.5 * (lo_x + hi_x), 0.5 * (lo_y + hi_y)

        bx = (sx - cx) * scale + self.big * 0.5
        by = (sy - cy) * scale + self.big * 0.5

        # Each voxel covers a SPLAT x SPLAT block. Expanding first and sorting
        # once keeps the painter's algorithm exact: a single far-to-near order
        # over every pixel any voxel touches, so the nearest writer always wins.
        # (Assigning one offset at a time would let a far voxel's second pixel
        # overwrite a near voxel's first.)
        n = coords.shape[0]
        offs = [(dx, dy) for dy in range(SPLAT) for dx in range(SPLAT)]
        px = np.concatenate([bx + dx for dx, _ in offs]).astype(np.int32)
        py = np.concatenate([by + dy for _, dy in offs]).astype(np.int32)
        np.clip(px, 0, self.big - 1, out=px)
        np.clip(py, 0, self.big - 1, out=py)
        dep = np.tile(depth, len(offs))
        idx = np.tile(np.arange(n, dtype=np.int32), len(offs))

        order = np.argsort(dep, kind="stable")
        self.px, self.py, self.idx = px[order], py[order], idx[order]
        dep = dep[order]
        t = (dep - dep[0]) / max(float(dep[-1] - dep[0]), 1e-6)
        self.shade = (SHADE_LO + t * (SHADE_HI - SHADE_LO)).astype(np.float32)
        self.n_voxels = n

    def render(self, emissive=None):
        """Paint one panel. Returns uint8 RGBA (size, size, 4)."""
        size = self.size
        if self.empty:
            return np.zeros((size, size, 4), dtype=np.uint8)
        if emissive is None:
            em = np.zeros(self.idx.shape[0], dtype=bool)
        else:
            em = np.asarray(emissive).astype(bool)[self.idx]

        base = np.where(em[:, None],
                        np.array(EMIT_RGB, dtype=np.float32)[None, :],
                        np.array(GEOM_RGB, dtype=np.float32)[None, :])
        sh = np.where(em, np.maximum(self.shade, EMIT_SHADE_FLOOR), self.shade)[:, None]
        rgb = np.clip(base * sh, 0.0, 1.0)

        canvas = np.zeros((self.big, self.big, 4), dtype=np.float32)
        canvas[self.py, self.px, :3] = rgb
        canvas[self.py, self.px, 3] = 1.0

        # Area downsample: averages colour AND coverage, which antialiases the
        # silhouette against the transparent background.
        canvas = canvas.reshape(size, SS, size, SS, 4).mean(axis=(1, 3))
        a = canvas[:, :, 3:4]
        with np.errstate(invalid="ignore", divide="ignore"):
            rgb_out = np.where(a > 0, canvas[:, :, :3] / np.maximum(a, 1e-6), 0.0)
        out = np.concatenate([np.clip(rgb_out, 0, 1), np.clip(a, 0, 1)], axis=2)
        return (out * 255.0 + 0.5).astype(np.uint8)


def render_panel(coords, emissive=None, size=192, view=None):
    """One-shot convenience: build a ShapeView and render a single panel."""
    view = view or ShapeView(coords, size=size)
    return view.render(emissive)


def save_png(arr, path):
    """Write a uint8 RGBA array as a PNG. Pillow when it is available (it is, in
    the evaluation environment); a zlib fallback keeps this module usable
    anywhere, so a thumbnail is never lost to a missing image library."""
    try:
        from PIL import Image
        Image.fromarray(arr, mode="RGBA").save(path, optimize=True)
        return
    except ImportError:
        pass
    import struct
    import zlib
    h, w = arr.shape[:2]
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
