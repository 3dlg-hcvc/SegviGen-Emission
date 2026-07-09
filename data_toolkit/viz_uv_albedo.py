#!/usr/bin/env python3
"""Albedo alignment diagnostic: bridge the DENSE base_color voxels into UV and
compare to the somage 'color' (albedo) map for the same objects. Albedo covers
the whole surface, so a frame/bridge misalignment smears the entire texture (a
much more sensitive alignment test than sparse emission). Same 16 objects as the
emission viewer (sorted by emission ceiling IoU). Runs in the trellis2 env
(needs o_voxel to read the pbr .vxz extracted from the LOCAL part tars).
"""
import os, io, base64, json, tarfile, tempfile, numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import o_voxel

E = "outputs/emission_dit_74k/eval"
SOM = "/cs/3dlg-falas/project/omages/datasets/TexVerse/lightgen/somages_corresp_dc80k"
TARDIR = "/local-scratch/localhome/dya78/lightgen_74k_staging/voxels"
OUT = "visualizations/uv_albedo_alignment.html"
N = 16

meta = pd.read_parquet(f"{E}/metadata_74k.parquet")
g = pd.read_csv("../evaluation/eval_emission_dit74k/gt_emission_full/eval_results.csv").set_index("shape_id")
pool = [(s, r.iou_2d) for s, r in g.iterrows()
        if os.path.exists(f"{E}/pred_voxels/{s}.npz") and np.isfinite(r.iou_2d)]
pool.sort(key=lambda x: x[1])
picks = [pool[i][0] for i in np.linspace(0, len(pool) - 1, N).astype(int)]
emis_iou = {pool[i][0]: pool[i][1] for i in np.linspace(0, len(pool) - 1, N).astype(int)}
want = set(picks)

# extract the needed pbr voxels from the local part tars
tmp = tempfile.mkdtemp()
prefix = "pbr_voxels_256/"
got = {}
for t in sorted(os.listdir(TARDIR)):
    if not t.endswith(".tar") or len(got) == len(want):
        continue
    with tarfile.open(os.path.join(TARDIR, t)) as tf:
        for m in tf:
            if m.name.startswith(prefix) and m.name.endswith(".vxz"):
                sha = m.name[len(prefix):-4]
                if sha in want and sha not in got:
                    m.name = os.path.basename(m.name)
                    tf.extract(m, tmp)
                    got[sha] = os.path.join(tmp, m.name)
print(f"extracted {len(got)}/{len(want)} pbr voxels", flush=True)


def voxel_centers(coords, grid=256):
    return -0.5 + (coords.astype(np.float64) + 0.5) / grid


def psnr(a, b, m):
    mse = (((a - b) ** 2).sum(-1)[m]).mean() / 3.0
    return float("inf") if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))


def panel_b64(sha):
    raw = np.load(os.path.join(SOM, meta.loc[sha, "ditem_dir"], "somage.npz"))
    occ = raw["occupancy"]; occ = (occ[..., 0] if occ.ndim == 3 else occ).astype(bool)
    col = raw["color"].astype(np.float32) / 255            # somage albedo
    pos = raw["position"].astype(np.float32) / 65535 * 2 - 1
    coords, attr = o_voxel.io.read_vxz(got[sha], num_threads=2)
    coords = coords.int().cpu().numpy()
    bc = np.clip(attr["base_color"].float().cpu().numpy() / 255.0, 0, 1)   # voxel albedo
    tree = cKDTree(voxel_centers(coords, 256))
    uv = np.zeros_like(col)
    dist, idx = tree.query(pos[occ] / 2.0, k=1)
    vv = bc[idx].copy(); vv[dist > 1.5 / 256] = 0
    uv[occ] = vv
    p = psnr(col, uv, occ); mae = np.abs(col - uv)[occ].mean()
    om = occ[..., None]
    col_m = np.clip(col * om, 0, 1)          # mask somage albedo by occupancy (drop background atlas)
    uv_m = np.clip(uv, 0, 1)                  # bridge is already occupancy-masked
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.7), facecolor="#0b0e14")
    for a in ax:
        a.set_facecolor("#0b0e14"); a.set_xticks([]); a.set_yticks([])
    ax[0].imshow(col_m); ax[0].set_title("TEXGen UV albedo  (somage 'color' × occupancy)", color="#cdd6f4", fontsize=12)
    ax[1].imshow(uv_m); ax[1].set_title("o-voxel base_color → UV  (bridge)", color="#cdd6f4", fontsize=12)
    ax[2].imshow(np.clip(np.abs(col_m - uv_m) * 2, 0, 1)); ax[2].set_title("|diff| ×2  (over surface)", color="#cdd6f4", fontsize=12)
    fig.suptitle(f"{sha}    albedo PSNR={p:.1f} dB   MAE={mae:.3f}   (emission ceil IoU={emis_iou[sha]:.3f})",
                 color="#f5c2e7", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=96, facecolor="#0b0e14", bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), p, mae


imgs, opts = {}, []
for sha in picks:
    if sha not in got:
        continue
    b64, p, mae = panel_b64(sha)
    imgs[sha] = b64
    opts.append((sha, p, mae))
    print(f"  {sha[:16]}  albedo PSNR={p:.1f}dB  MAE={mae:.3f}", flush=True)

optlines = "\n".join(
    f'<option value="{s}">{i+1:02d}.  albedo PSNR={p:.1f}dB   MAE={mae:.3f}   {s[:14]}…</option>'
    for i, (s, p, mae) in enumerate(opts))
data_js = "var IMGS=" + json.dumps(imgs) + ";"
html = """<!doctype html><html><head><meta charset=utf-8><title>Albedo UV alignment</title>
<style>
 body{background:#0b0e14;color:#cdd6f4;font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 .sub{color:#7f849c;font-size:13px;margin-bottom:18px;max-width:1150px;line-height:1.5}
 select{background:#1e2030;color:#cdd6f4;border:1px solid #313244;border-radius:8px;padding:8px 12px;font-size:14px;min-width:480px}
 .nav{display:flex;gap:10px;align-items:center;margin-bottom:16px}
 button{background:#1e2030;color:#cdd6f4;border:1px solid #313244;border-radius:8px;padding:8px 14px;font-size:14px;cursor:pointer}
 button:hover{background:#313244} img{max-width:100%;border:1px solid #313244;border-radius:10px}
 .legend{color:#7f849c;font-size:12px;margin-top:10px;max-width:1150px;line-height:1.5} code{color:#a6e3a1}
</style></head><body>
<h1>Albedo (base_color) UV alignment &nbsp;—&nbsp; o-voxel → UV bridge vs TEXGen UV</h1>
<div class=sub>Dense diagnostic: albedo covers the whole surface, so a frame/bridge misalignment smears the <b>entire</b> texture (unlike sparse emission). Left = somage <code>color</code> (albedo). Middle = <b>base_color voxels resampled into UV</b> via the same bridge. Right = |diff|×2. High PSNR + a near-black diff = the bridge is spatially correct. Same objects as the emission viewer (emission ceiling IoU shown for cross-reference). Use ←/→.</div>
<div class=nav><button onclick="step(-1)">← prev</button>
 <select id=sel onchange="show(this.value)">__OPTS__</select>
 <button onclick="step(1)">next →</button></div>
<div><img id=img></div>
<div class=legend>If albedo aligns cleanly on a shape whose <i>emission</i> looked bad, the emission problem is emission-specific (sparsity, dim values, voxel-vs-UV bake gap, or the model) — not the bridge. If albedo is <i>also</i> smeared/shifted, that shape has a real geometry/UV issue (e.g. degenerate UV or a voxelization mismatch).</div>
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
