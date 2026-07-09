#!/usr/bin/env python3
"""Scalable folder viewer (index.html + p/<sha>.png, lazy-loaded) comparing the
o-voxel -> UV bridge against the somage GT, for many objects.

  --mode emission : somage emission (masked) | o-voxel emission bridge | mask overlay
  --mode albedo   : somage albedo (masked)   | o-voxel base_color bridge (srgb2lin matched) | |diff|

Albedo reads pbr voxels extracted from the LOCAL part tars; emission reads the
pred_voxels npz produced by the eval. Run in the trellis2 env (o_voxel). Deploy
the produced folder with:  cp -r visualizations/gallery_<mode> /project/3dlg-hcvc/lightgen/www/
"""
import os, sys, argparse, io, base64, tarfile, tempfile, numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

E = "outputs/emission_dit_74k/eval"
SOM = "/cs/3dlg-falas/project/omages/datasets/TexVerse/lightgen/somages_corresp_dc80k"
VDIR = "/local-scratch/localhome/dya78/lightgen_74k_staging/voxels"

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["emission", "albedo", "colormatch"], required=True)
ap.add_argument("--n", type=int, default=60)
args = ap.parse_args()

meta = pd.read_parquet(f"{E}/metadata_74k.parquet")
g = pd.read_csv(f"../evaluation/eval_emission_dit74k/gt_emission_full/eval_results.csv").set_index("shape_id")
pool = [(s, r.iou_2d) for s, r in g.iterrows()
        if os.path.exists(f"{E}/pred_voxels/{s}.npz") and np.isfinite(r.iou_2d)]
pool.sort(key=lambda x: x[1])
picks = [pool[i] for i in np.unique(np.linspace(0, len(pool) - 1, args.n).astype(int))]
shas = [s for s, _ in picks]


def voxel_centers(c):
    return -0.5 + (c.astype(np.float64) + 0.5) / 256


def srgb2lin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def boost(x):
    m = float(x.max()) or 1.0
    return np.clip(x / m, 0, 1)


def psnr(a, b, m):
    e = (((a - b) ** 2).sum(-1)[m]).mean() / 3.0
    return 99.0 if e <= 0 else 20 * np.log10(1 / np.sqrt(e))


# albedo / colormatch: extract needed pbr voxels from local tars
pbr_path = {}
if args.mode in ("albedo", "colormatch"):
    tmp = tempfile.mkdtemp(); want = set(shas)
    for t in sorted(os.listdir(VDIR)):
        if not t.endswith(".tar") or len(pbr_path) == len(want):
            continue
        with tarfile.open(os.path.join(VDIR, t)) as tf:
            for m in tf:
                if m.name.startswith("pbr_voxels_256/") and m.name.endswith(".vxz"):
                    s = m.name[len("pbr_voxels_256/"):-4]
                    if s in want and s not in pbr_path:
                        m.name = s + ".vxz"; tf.extract(m, tmp); pbr_path[s] = os.path.join(tmp, m.name)
    import o_voxel
    print(f"extracted {len(pbr_path)}/{len(want)} pbr voxels", flush=True)


def render(sha):
    raw = np.load(os.path.join(SOM, meta.loc[sha, "ditem_dir"], "somage.npz"))
    occ = raw["occupancy"]; occ = (occ[..., 0] if occ.ndim == 3 else occ).astype(bool)
    pos = raw["position"].astype(np.float32) / 65535 * 2 - 1
    om = occ[..., None]
    nc = 4 if args.mode == "colormatch" else 3
    fig, ax = plt.subplots(1, nc, figsize=(4.5 * nc, 4.7), facecolor="#0b0e14")
    for a in ax:
        a.set_facecolor("#0b0e14"); a.set_xticks([]); a.set_yticks([])
    if args.mode == "colormatch":
        col = raw["color"].astype(np.float32) / 255
        coords, attr = o_voxel.io.read_vxz(pbr_path[sha], num_threads=2)
        coords = coords.int().cpu().numpy()
        bc = np.clip(attr["base_color"].float().cpu().numpy() / 255.0, 0, 1)
        tree = cKDTree(voxel_centers(coords))
        uv = np.zeros_like(col); dist, idx = tree.query(pos[occ] / 2.0, k=1)
        vv = bc[idx].copy(); vv[dist > 1.5 / 256] = 0; uv[occ] = vv
        cm = np.clip(col * om, 0, 1)
        uv_m = srgb2lin(uv) * om
        p_raw = psnr(cm, uv * om, occ); p_m = psnr(cm, uv_m, occ)
        ax[0].imshow(cm); ax[0].set_title("somage albedo (target, linear)", color="#cdd6f4", fontsize=11)
        ax[1].imshow(np.clip(uv * om, 0, 1)); ax[1].set_title(f"bridge raw (sRGB)  PSNR={p_raw:.1f}", color="#cdd6f4", fontsize=11)
        ax[2].imshow(np.clip(uv_m, 0, 1)); ax[2].set_title(f"bridge srgb2lin  PSNR={p_m:.1f}", color="#a6e3a1", fontsize=11)
        ax[3].imshow(np.clip(np.abs(cm - uv_m) * 3, 0, 1)); ax[3].set_title("|diff| matched ×3", color="#cdd6f4", fontsize=11)
        label = f"raw={p_raw:.1f} → matched={p_m:.1f} dB  (+{p_m-p_raw:.1f})"
    elif args.mode == "emission":
        gt = raw["emission_color"].astype(np.float32) / 255
        d = np.load(f"{E}/pred_voxels/{sha}.npz")
        ve = np.clip(d["gt_emission"].astype(np.float32), 0, 1)
        tree = cKDTree(voxel_centers(d["coords"]))
        uv = np.zeros_like(gt); dist, idx = tree.query(pos[occ] / 2.0, k=1)
        vv = ve[idx].copy(); vv[dist > 1.5 / 256] = 0; uv[occ] = vv
        gm = (gt.max(-1) > 0) & occ; bm = (uv.max(-1) > 0) & occ
        u = (gm | bm).sum(); iou = (gm & bm).sum() / u if u else float("nan")
        ov = np.zeros_like(gt); ov[..., 0] = gm; ov[..., 1] = bm
        ax[0].imshow(boost(gt * om)); ax[0].set_title("TEXGen UV  (somage emission × occ)", color="#cdd6f4", fontsize=11)
        ax[1].imshow(boost(uv)); ax[1].set_title("o-voxel emission → UV  (bridge)", color="#cdd6f4", fontsize=11)
        ax[2].imshow(ov); ax[2].set_title("overlay  R=TEXGen G=o-voxel (yellow=agree)", color="#cdd6f4", fontsize=11)
        label = f"IoU={iou:.3f}"
    else:
        col = raw["color"].astype(np.float32) / 255
        coords, attr = o_voxel.io.read_vxz(pbr_path[sha], num_threads=2)
        coords = coords.int().cpu().numpy()
        bc = np.clip(attr["base_color"].float().cpu().numpy() / 255.0, 0, 1)
        tree = cKDTree(voxel_centers(coords))
        uv = np.zeros_like(col); dist, idx = tree.query(pos[occ] / 2.0, k=1)
        vv = bc[idx].copy(); vv[dist > 1.5 / 256] = 0; uv[occ] = vv
        uv = srgb2lin(uv) * om          # color-space match (sRGB voxel -> linear) + mask
        cm = np.clip(col * om, 0, 1)
        p = psnr(cm, uv, occ)
        ax[0].imshow(cm); ax[0].set_title("TEXGen UV albedo  (somage 'color' × occ)", color="#cdd6f4", fontsize=11)
        ax[1].imshow(np.clip(uv, 0, 1)); ax[1].set_title("o-voxel base_color → UV  (srgb2lin matched)", color="#cdd6f4", fontsize=11)
        ax[2].imshow(np.clip(np.abs(cm - uv) * 3, 0, 1)); ax[2].set_title("|diff| ×3", color="#cdd6f4", fontsize=11)
        label = f"PSNR={p:.1f}dB"
    fig.suptitle(f"{sha}    {label}", color="#f5c2e7", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, label


outdir = f"visualizations/gallery_{args.mode}"
pdir = os.path.join(outdir, "p")
os.makedirs(pdir, exist_ok=True)
entries = []
for i, sha in enumerate(shas):
    if args.mode == "albedo" and sha not in pbr_path:
        continue
    try:
        fig, label = render(sha)
    except Exception as e:
        print(f"  skip {sha[:12]}: {e}", flush=True); continue
    fig.savefig(os.path.join(pdir, f"{sha}.png"), dpi=92, facecolor="#0b0e14", bbox_inches="tight")
    plt.close(fig)
    entries.append((sha, label))
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(shas)}", flush=True)

title = {"emission": "Emission", "albedo": "Albedo (sRGB-matched)",
         "colormatch": "Albedo color-space match (sRGB → linear)"}[args.mode]
metric = "IoU" if args.mode == "emission" else "PSNR"
opt = "\n".join(f'<option value="{s}">{i+1:02d}.  {lab}   {s[:14]}…</option>' for i, (s, lab) in enumerate(entries))
js_list = "var L=" + str([s for s, _ in entries]) + ";"
html = """<!doctype html><html><head><meta charset=utf-8><title>__TITLE__ UV gallery</title>
<style>
 body{background:#0b0e14;color:#cdd6f4;font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px}
 h1{font-size:18px;font-weight:600;margin:0 0 4px} .sub{color:#7f849c;font-size:13px;margin-bottom:16px;max-width:1150px;line-height:1.5}
 select{background:#1e2030;color:#cdd6f4;border:1px solid #313244;border-radius:8px;padding:8px 12px;font-size:14px;min-width:460px}
 .nav{display:flex;gap:10px;align-items:center;margin-bottom:14px} code{color:#a6e3a1}
 button{background:#1e2030;color:#cdd6f4;border:1px solid #313244;border-radius:8px;padding:8px 14px;font-size:14px;cursor:pointer} button:hover{background:#313244}
 img{max-width:100%;border:1px solid #313244;border-radius:10px} .c{color:#7f849c;font-size:12px;margin-top:6px}
</style></head><body>
<h1>__TITLE__ — o-voxel → UV bridge vs TEXGen UV (__N__ objects, sorted by emission IoU)</h1>
<div class=sub>somage GT masked by occupancy. __SUBNOTE__ Use the dropdown or ←/→ keys.</div>
<div class=nav><button onclick="step(-1)">← prev</button>
 <select id=sel onchange="show(this.selectedIndex)">__OPT__</select>
 <button onclick="step(1)">next →</button><span class=c id=cnt></span></div>
<div><img id=img></div>
<script>__LIST__
var i=0;function show(k){i=(k+L.length)%L.length;document.getElementById("img").src="p/"+L[i]+".png";document.getElementById("sel").selectedIndex=i;document.getElementById("cnt").textContent=(i+1)+" / "+L.length;}
function step(d){show(i+d);}
document.addEventListener("keydown",function(e){if(e.key=="ArrowLeft")step(-1);if(e.key=="ArrowRight")step(1);});
show(0);
</script></body></html>"""
subnote = {"emission": "Right = mask overlay (yellow=agree).",
           "albedo": "Middle bridge is sRGB→linear matched to the linear somage albedo; right = |diff|×3.",
           "colormatch": "Panels: somage albedo target (linear) | raw bridge (voxel base_color is sRGB → too bright) | bridge after srgb2lin (matched) | |diff|×3 after match. PSNR jumps once the color spaces agree."}[args.mode]
html = (html.replace("__TITLE__", title).replace("__N__", str(len(entries)))
        .replace("__SUBNOTE__", subnote).replace("__OPT__", opt).replace("__LIST__", js_list))
open(os.path.join(outdir, "index.html"), "w").write(html)
print(f"\n{len(entries)} panels -> {outdir}/  (index.html + p/*.png)")
