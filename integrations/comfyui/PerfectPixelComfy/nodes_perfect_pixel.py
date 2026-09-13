import importlib

import numpy as np
import torch
import torch.nn.functional as F

# ---- helpers ----

def _torch_image_to_uint8_rgb(img: torch.Tensor) -> np.ndarray:
    """
    ComfyUI IMAGE: float32, [B,H,W,C], range 0..1, C=3
    -> numpy uint8 RGB [B,H,W,3]
    """
    if (not isinstance(img, torch.Tensor) or img.ndim != 4
            or img.shape[-1] != 3 or min(img.shape[:3]) < 1):
        raise ValueError("Expected a non-empty IMAGE tensor shaped [B,H,W,3]")
    img = img.detach().cpu()
    if not torch.isfinite(img).all():
        raise ValueError("IMAGE must contain finite values")
    img = torch.clamp(img, 0.0, 1.0)
    img = (img * 255.0).round().to(torch.uint8)
    return img.numpy()

def _uint8_rgb_to_torch_image(rgb: np.ndarray) -> torch.Tensor:
    """
    numpy uint8 RGB [H,W,3] -> torch float32 [1,H,W,3] 0..1
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    t = torch.from_numpy(rgb).float() / 255.0
    return t.unsqueeze(0)  # [1,H,W,3]

def _nearest_scale_bhwc(img_bhwc: torch.Tensor, scale: int) -> torch.Tensor:
    """
    img_bhwc: [B,H,W,C] -> scaled [B,H*scale,W*scale,C] using nearest
    """
    if scale == 1:
        return img_bhwc
    x = img_bhwc.permute(0, 3, 1, 2)  # [B,C,H,W]
    x = F.interpolate(x, scale_factor=scale, mode="nearest")
    return x.permute(0, 2, 3, 1)  # [B,H,W,C]

def _import_core(module_name: str):
    """Support both a pip installation and backends copied beside this node."""
    if __package__:
        try:
            module = importlib.import_module("." + module_name, __package__)
        except ModuleNotFoundError as exc:
            # Only a missing backend is a reason to try the installed package.
            # Do not hide a missing dependency inside an existing backend.
            if exc.name != __package__ + "." + module_name:
                raise
        else:
            return module.get_perfect_pixel
    return importlib.import_module("perfect_pixel." + module_name).get_perfect_pixel


def _load_backend(backend: str):
    if backend not in ("Auto", "OpenCV Backend", "Lightweight Backend"):
        raise ValueError("Unknown Perfect Pixel backend: " + str(backend))
    if backend == "Lightweight Backend":
        return _import_core("perfect_pixel_noCV2")
    if backend == "OpenCV Backend":
        importlib.import_module("cv2")  # Explicit selection must not silently fall back.
        return _import_core("perfect_pixel")
    try:
        importlib.import_module("cv2")
    except ModuleNotFoundError as exc:
        if exc.name != "cv2":
            raise
        return _import_core("perfect_pixel_noCV2")
    return _import_core("perfect_pixel")


class PerfectPixelNode:
    """
    IMAGE -> IMAGE
    PerfectPixel grid detection + sampling (center/majority) + nearest zoom/export scale.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sampling": (["Majority Cluster", "Center Sample"], {"default": "Majority Cluster"}),
                "export_scale": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "backend": (["Auto", "OpenCV Backend", "Lightweight Backend"], {"default": "Auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "image/postprocessing"

    def run(self, image, sampling, export_scale, backend):
        if sampling not in ("Majority Cluster", "Center Sample"):
            raise ValueError("Unknown sampling method: " + str(sampling))
        if (not isinstance(export_scale, (int, np.integer))
                or isinstance(export_scale, (bool, np.bool_)) or not 1 <= export_scale <= 16):
            raise ValueError("export_scale must be an integer in [1, 16]")
        get_perfect_pixel = _load_backend(backend)

        # ComfyUI may pass batches: [B,H,W,C]
        imgs = _torch_image_to_uint8_rgb(image)  # -> numpy [B,H,W,C] uint8

        method = "majority" if sampling == "Majority Cluster" else "center"

        outs = []
        for i in range(imgs.shape[0]):
            rgb = imgs[i]  # [H,W,3] uint8

            # perfect_pixel expects RGB
            _, _, out_rgb = get_perfect_pixel(
                rgb,
                sample_method=method,
                debug=False
            )
            # On detection failure the library returns the original RGB image.

            out_t = _uint8_rgb_to_torch_image(out_rgb)  # [1,h,w,3]
            outs.append(out_t)

        # stack: require same H/W across batch (common case)
        Hs = {t.shape[1] for t in outs}
        Ws = {t.shape[2] for t in outs}
        if len(Hs) != 1 or len(Ws) != 1:
            raise ValueError(
                "PerfectPixel produced different sizes across the batch. "
                "Process images one-by-one; equal input dimensions do not guarantee equal grids."
            )

        out = torch.cat(outs, dim=0)  # [B,H,W,3]
        out = _nearest_scale_bhwc(out, int(export_scale))
        out = torch.clamp(out, 0.0, 1.0)
        return (out,)
