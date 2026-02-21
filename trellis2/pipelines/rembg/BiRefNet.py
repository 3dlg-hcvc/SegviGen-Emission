from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from contextlib import contextmanager
from torchvision import transforms
from PIL import Image
import os


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        @contextmanager
        def _force_cpu_default_device():
            try:
                get_default = torch.get_default_device
                set_default = torch.set_default_device
            except AttributeError:
                yield
                return
            prev = get_default()
            try:
                if str(prev) != "cpu":
                    set_default("cpu")
                yield
            finally:
                try:
                    set_default(prev)
                except Exception:
                    pass

        # Avoid meta-device initialization on newer clusters, but stay compatible
        # with older transformers that don't accept these kwargs.
        # Also set environment variable to disable meta device
        old_accelerate_use_cpu = os.environ.get('ACCELERATE_USE_CPU_ONLY', None)
        os.environ['ACCELERATE_USE_CPU_ONLY'] = '1'
        
        # Monkey-patch torch.linspace to ensure it always returns CPU tensors
        # This fixes an issue where .item() is called on meta tensors in the cached model code
        original_linspace = torch.linspace
        def patched_linspace(*args, **kwargs):
            kwargs.setdefault('device', 'cpu')
            return original_linspace(*args, **kwargs)
        torch.linspace = patched_linspace
        
        # Monkey-patch torch.nn.Module.__getattr__ to handle missing all_tied_weights_keys
        # This fixes compatibility with newer transformers versions
        from torch.nn import Module
        original_getattr = Module.__getattr__
        def patched_getattr(self, name):
            if name == 'all_tied_weights_keys':
                # If all_tied_weights_keys is missing, return _tied_weights_keys or empty dict
                if hasattr(self, '_tied_weights_keys') and self._tied_weights_keys is not None:
                    return self._tied_weights_keys
                return {}
            return original_getattr(self, name)
        Module.__getattr__ = patched_getattr
        
        try:
            with _force_cpu_default_device():
                try:
                    self.model = AutoModelForImageSegmentation.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        low_cpu_mem_usage=False,
                        device_map=None,
                        torch_dtype=torch.float32,
                    )
                except TypeError:
                    self.model = AutoModelForImageSegmentation.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                    )
        finally:
            # Restore original functions and environment
            torch.linspace = original_linspace
            Module.__getattr__ = original_getattr
            if old_accelerate_use_cpu is None:
                os.environ.pop('ACCELERATE_USE_CPU_ONLY', None)
            else:
                os.environ['ACCELERATE_USE_CPU_ONLY'] = old_accelerate_use_cpu
            
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    
