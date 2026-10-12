"""The thumbnail -> DINOv3 conditioning of SegviGen, for route 3 of build_cond_tree.py.

Five functions copied verbatim from the original SegviGen tree (QhelDIV/SegviGen-emissive at a67a69c, the
code that built every existing conditioning file): preprocess_image, get_cond and img_to_cond from
data_toolkit/img_to_cond.py, preprocess_image_repair and img_to_cond_repair from
emissive/data_prep/build_cond_thumbnail.py (one change: img_to_cond_repair takes get_cond from this module
instead of importing img_to_cond). The caller passes this repo's BiRefNet (trellis2.pipelines.rembg) and
DinoV3FeatureExtractor (trellis2.modules.image_feature_extractor): the extractor is byte-identical to that
tree's, and BiRefNet differs only in how it loads.

That code is under the MIT License:

    MIT License

    Copyright (c) 2026 Lin Li

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
import numpy as np
import torch
from PIL import Image


def preprocess_image(rembg_model, input):
    if input.mode != "RGB":
        bg = Image.new("RGB", input.size, (255, 255, 255))
        bg.paste(input, mask=input.split()[3])
        input = bg
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(input.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
    if has_alpha:
        output = input
    else:
        input = input.convert('RGB')
        output = rembg_model(input)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output

def get_cond(image_cond_model, image):
    image_cond_model.image_size = 512
    cond = image_cond_model(image)
    neg_cond = torch.zeros_like(cond)
    return {'cond': cond.cpu(), 'neg_cond': neg_cond.cpu()}

def img_to_cond(rembg_model, image_cond_model, image_path, save_path):
    image = Image.open(image_path)
    image = preprocess_image(rembg_model, image)
    cond_dict = get_cond(image_cond_model, [image])
    torch.save(cond_dict, save_path)

def preprocess_image_repair(rembg_model, image):
    """DEVIATION FROM PATH A, applied ONLY under --repair, only to the shapes whose
    thumbnails Path A's preprocess_image cannot process at all. Two guards, both
    measured against real failures on the TexVerse thumbnails; nothing else changes,
    and the guards are no-ops on any image the unmodified function already handles.

    Guard 1 (mode). Path A's images are RGB or RGBA renders, so `input.split()[3]`
    is always the alpha band there. Some TexVerse thumbnails are 1-band grayscale
    ('L', 1024x640) -- these are TexVerse's PLACEHOLDER image, a generic cube icon
    rather than a render of the shape -- and the indexing raises IndexError. A
    1-band image has no alpha to composite, so converting to RGB is what the
    original branch's "flatten onto white" intent reduces to.

    Guard 2 (empty foreground). A handful of thumbnails frame the object as a
    few-pixel speck on a flat background; BiRefNet then produces no alpha above
    0.8*255 anywhere, the bbox is empty and np.min raises. Fall back to a lower
    alpha threshold, then to the uncropped full frame. The object is genuinely
    tiny in these images, so the resulting embedding is weak conditioning -- but
    it is the real image, never zeros.

    Effect on outputs: for the placeholder shapes the conditioning carries no
    shape-specific information (a near-constant embedding); for the speck shapes
    it carries very little. Both are counted and named in the run report.
    """
    from PIL import Image
    import numpy as np
    if image.mode not in ("RGB", "RGBA"):
        # guard 1
        if "A" in image.getbands():
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
    if image.mode != "RGB":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    has_alpha = False
    if image.mode == "RGBA":
        alpha = np.array(image)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)),
                             Image.Resampling.LANCZOS)
    if has_alpha:
        output = image
    else:
        image = image.convert("RGB")
        output = rembg_model(image)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    used = "alpha>0.8"
    bbox_idx = np.argwhere(alpha > 0.8 * 255)
    if len(bbox_idx) == 0:
        # guard 2
        bbox_idx = np.argwhere(alpha > 0.1 * 255)
        used = "alpha>0.1"
    if len(bbox_idx) == 0:
        bbox = (0, 0, output.width - 1, output.height - 1)
        used = "full_frame"
    else:
        bbox = (int(bbox_idx[:, 1].min()), int(bbox_idx[:, 0].min()),
                int(bbox_idx[:, 1].max()), int(bbox_idx[:, 0].max()))
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = (center[0] - size // 2, center[1] - size // 2,
            center[0] + size // 2, center[1] + size // 2)
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    return Image.fromarray((output * 255).astype(np.uint8)), used

def img_to_cond_repair(rembg_model, dino_model, image_path, save_path):
    """--repair twin of img_to_cond: same get_cond (DINOv3 @512, neg_cond=zeros),
    only preprocess_image is swapped for the guarded version above."""
    from PIL import Image
    image = Image.open(image_path)
    image, used = preprocess_image_repair(rembg_model, image)
    torch.save(get_cond(dino_model, [image]), save_path)
    return used
