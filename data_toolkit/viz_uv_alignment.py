#!/usr/bin/env python3
"""Single-file HTML viewer comparing, per object, TEXGen's UV (somage GT emission)
vs the o-voxel emission resampled into UV via the bridge. Sorted by ceiling IoU so
both clean alignment and failure modes (boundary bleed) are visible. Run from the
TRELLIS2 repo root (needs the eval outputs + somages on /cs/3dlg-falas)."""
import os, io, base64, json, numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from bridge_voxel_to_uv import voxel_centers

E = "outputs/emission_dit_74k/eval"
SOM = "/cs/3dlg-falas/project/omages/datasets/TexVerse/lightgen/somages_corresp_dc80k"
N = 16
OUT = "visualizations/uv_alignment_texgen_vs_ovoxel.html"

meta = pd.read_parquet(f"{E}/metadata_74k.parquet")
g = pd.read_csv("../evaluation/eval_emission_dit74k/gt_emission_full/eval_results.csv").set_index("shape_id")

pool = [(s, r.iou_2d) for s, r in g.iterrows()
        if os.path.exists(f"{E}/pred_voxels/{s}.npz") and np.isfinite(r.iou_2d)]
pool.sort(key=lambda x: x[1])
picks = [pool[i] for i in np.linspace(0, len(pool) - 1, N).astype(int)]


def boost(x):
    m = float(x.max()) or 1.0
    return np.clip(x / m, 0, 1)


def panel_b64(sha):
    raw = np.load(os.path.join(SOM, meta.loc[sha, "ditem_dir"], "somage.npz"))
    occ = raw["occupancy"]; occ = (occ[..., 0] if occ.ndim == 3 else occ).astype(bool)
    gt = raw["emission_color"].astype(np.float32) / 255
    pos = raw["position"].astype(np.float32) / 65535 * 2 - 1
    d = np.load(f"{E}/pred_voxels/{sha}.npz")
    vemit = np.clip(d["gt_emission"].astype(np.float32), 0, 1)
    tree = cKDTree(voxel_centers(d["coords"], 256))
    uv = np.zeros_like(gt)
    dist, idx = tree.query(pos[occ] / 2.0, k=1)
    vv = vemit[idx].copy(); vv[dist > 1.5 / 256] = 0
    uv[occ] = vv
    gm = (gt.max(-1) > 0) & occ; bm = (uv.max(-1) > 0) & occ
    ov = np.zeros_like(gt); ov[..., 0] = gm; ov[..., 1] = bm
    union = (gm | bm).sum()
    iou = (gm & bm).sum() / union if union else float("nan")
    mae = np.abs(gt - uv)[(gm | bm)].mean() if union else 0.0
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.7), facecolor="#0b0e14")
    for a in ax:
        a.set_facecolor("#0b0e14"); a.set_xticks([]); a.set_yticks([])
    ax[0].imshow(boost(gt * occ[..., None])); ax[0].set_title("TEXGen UV  (somage GT emission × occupancy)", color="#cdd6f4", fontsize=12)
    ax[1].imshow(boost(uv)); ax[1].set_title("o-voxel → UV  (bridge)", color="#cdd6f4", fontsize=12)
    ax[2].imshow(ov); ax[2].set_title("overlay  R=TEXGen  G=o-voxel  (yellow=agree)", color="#cdd6f4", fontsize=12)
    fig.suptitle(f"{sha}    mask IoU={iou:.3f}   MAE(union)={mae:.3f}", color="#f5c2e7", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, facecolor="#0b0e14", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), iou


imgs, opts = {}, []
for sha, ceil in picks:
    b64, iou = panel_b64(sha)
    imgs[sha] = b64
    opts.append((sha, iou))
    print(f"  {sha[:16]} IoU={iou:.3f}", flush=True)

optlines = "\n".join(
    f'<option value="{s}">{i+1:02d}.  IoU={io:.3f}   {s[:16]}…</option>'
    for i, (s, io) in enumerate(opts))
data_js = "var IMGS=" + json.dumps(imgs) + ";"

html = """<!doctype html><html><head><meta charset=utf-8>
<title>TEXGen UV vs o-voxel UV</title>
<style>
 body{background:#0b0e14;color:#cdd6f4;font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 .sub{color:#7f849c;font-size:13px;margin-bottom:18px;max-width:1150px;line-height:1.5}
 select{background:#1e2030;color:#cdd6f4;border:1px solid #313244;border-radius:8px;padding:8px 12px;font-size:14px;min-width:420px}
 .nav{display:flex;gap:10px;align-items:center;margin-bottom:16px}
 button{background:#1e2030;color:#cdd6f4;border:1px solid #313244;border-radius:8px;padding:8px 14px;font-size:14px;cursor:pointer}
 button:hover{background:#313244}
 img{max-width:100%;border:1px solid #313244;border-radius:10px}
 .legend{color:#7f849c;font-size:12px;margin-top:10px;max-width:1150px;line-height:1.5}
 code{color:#a6e3a1}
</style></head><body>
<h1>TEXGen UV &nbsp;vs&nbsp; o-voxel → UV bridge &nbsp;—&nbsp; emission alignment (same object)</h1>
<div class=sub>Left = the <b>somage GT emission</b> (the UV layout TEXGen's evaluator compares against). Middle = the <b>o-voxel emission resampled into UV</b> via the position map (<code>p_voxel = p_somage/2</code>, nearest voxel). Right = mask overlay; <b>yellow</b> = both agree (aligned), red = TEXGen-only, green = o-voxel-only (boundary / thin-geometry bleed). Objects sorted by ceiling IoU. Emission boosted to per-panel max for visibility. Use ←/→ keys.</div>
<div class=nav>
 <button onclick="step(-1)">← prev</button>
 <select id=sel onchange="show(this.value)">__OPTS__</select>
 <button onclick="step(1)">next →</button>
</div>
<div><img id=img></div>
<div class=legend>The emissive UV island coincides (yellow) wherever aligned — the bridge places emission in the correct TEXGen UV chart. Green specks/lines = nearest-voxel bleed at UV-chart boundaries on thin geometry; the value gap (uniform UV-bake vs textured voxel-bake) shows as MAE. Both effects apply equally to pred / gt_recon / gt_emission, so they cancel in the DiT-vs-ceiling comparison.</div>
<script>__DATA__
var KEYS=Object.keys(IMGS),cur=0;
function show(k){cur=KEYS.indexOf(k);document.getElementById("img").src="data:image/png;base64,"+IMGS[k];document.getElementById("sel").value=k;}
function step(d){cur=(cur+d+KEYS.length)%KEYS.length;show(KEYS[cur]);}
document.addEventListener("keydown",function(e){if(e.key=="ArrowLeft")step(-1);if(e.key=="ArrowRight")step(1);});
show(KEYS[0]);
</script></body></html>"""
html = html.replace("__OPTS__", optlines).replace("__DATA__", data_js)

os.makedirs("visualizations", exist_ok=True)
open(OUT, "w").write(html)
print(f"\nHTML -> TRELLIS2/{OUT}  ({len(html)/1e6:.2f} MB, {len(imgs)} objects)")
