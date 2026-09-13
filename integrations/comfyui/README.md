# Perfect Pixel ComfyUI node

This repository provides **Perfect Pixel (Grid Restore)**, an IMAGE-to-IMAGE node in the `image/postprocessing` category.

![Example](../../images/comfyui.png)

## Install from this repository

1. Download or clone this repository.
2. Copy **`integrations/comfyui/PerfectPixelComfy`** into **`ComfyUI/custom_nodes/PerfectPixelComfy`**. Copy the folder, not only its Python files.
3. Using **the same Python interpreter that runs ComfyUI**, install the library:

~~~bash
# Recommended: OpenCV acceleration
python -m pip install "perfect-pixel[opencv]"

# NumPy-only alternative
python -m pip install perfect-pixel
~~~

To use the code from this checkout before a new release is published, install the repository root rather than the PyPI version:

~~~bash
python -m pip install "/path/to/perfectPixel[opencv]"
~~~

4. Restart ComfyUI and search for **Perfect Pixel (Grid Restore)**.

PyTorch is provided by ComfyUI; do not replace its PyTorch installation to install this node. The standalone library's debug/Matplotlib extra is not needed for the node.

## Alternative: copy the backends instead of installing the library

If you cannot install the library package, copy these two files from this checkout into the **same copied PerfectPixelComfy folder**:

- `src/perfect_pixel/perfect_pixel.py`
- `src/perfect_pixel/perfect_pixel_noCV2.py`

Install NumPy in ComfyUI's environment, and optionally opencv-python. The resulting directory is:

~~~text
ComfyUI/custom_nodes/PerfectPixelComfy/
    __init__.py
    nodes_perfect_pixel.py
    perfect_pixel.py
    perfect_pixel_noCV2.py
~~~

The loader first uses copied sibling backends, then falls back to the installed `perfect_pixel` package when no sibling backend exists. Copied backends must be updated together with the node; they take precedence over a pip-installed version.

## Parameters and behavior

- **sampling**: Majority Cluster or Center Sample.
- **export_scale**: integer nearest-neighbor enlargement from 1 to 16, default 4.
- **backend**:
  - **Auto**: use OpenCV when cv2 is installed, otherwise use the NumPy backend. Errors inside an installed backend are not silently hidden.
  - **OpenCV Backend**: explicitly require cv2.
  - **Lightweight Backend**: explicitly select the NumPy implementation, even if cv2 is installed.

Input is a finite RGB tensor shaped `[B,H,W,3]` in the usual ComfyUI 0..1 range. Output uses the same tensor convention. On detection failure the original image is retained, then enlarged by export_scale. Use scale 1 when checking uncertain inputs to avoid unnecessarily large fallback images.

Every output in a batch must have the same detected width and height. Equal **input** dimensions do not guarantee equal detected grids. If a batch contains different grid sizes (or a mix of successful and failed detections), process the images individually; the node reports a clear error rather than silently resizing or distorting them.

## Windows portable ComfyUI

Installing packages into system Python does not install them into ComfyUI. Find the **Python executable** in ComfyUI's startup log, then use that exact executable. For example, in PowerShell:

~~~powershell
& "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe" -m pip install "perfect-pixel[opencv]"

# For this local checkout instead of a published release:
& "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe" -m pip install "D:\Project\perfectPixel[opencv]"
~~~

Replace both paths with your actual locations. Restart ComfyUI after installing or updating the package.
