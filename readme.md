# Perfect Pixel

> **Auto detect and Get perfect Pixel art**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

<img src="https://github.com/theamusing/perfectPixel/raw/main/assets/process.png" width="100%" />

Standard scaling often fails to sample AI-generated pixel art due to inconsistent sizes and non-square grids. 

This tool automatically detects the optimal grid and delivers perfectly aligned, pixel-perfect results.

## Features
- Automatically detect grid size from pixel style images.
- Refines AI generated pixel style image to perfectly aligned grids.
- Easy to integrate into your own workflow.

[Try the Web Demo](https://theamusing.github.io/perfectPixel_webdemo/)

## Installation

**Perfect Pixel** provides two implementations of the same core algorithm. The Lightweight Backend is designed in case you can't or don't want to use cv2. You can choose the one that best fits your environment. Both backends share the API, but floating-point and clustering results are not guaranteed to be identical:

| Feature | OpenCV Backend ([`perfect_pixel.py`](./src/perfect_pixel/perfect_pixel.py)) | Lightweight Backend ([`perfect_pixel_no_cv2.py`](./src/perfect_pixel/perfect_pixel_noCV2.py)) |
| :--- | :--- | :--- |
| **Dependencies** | `opencv-python`, `numpy` | `numpy` |

You can install Perfect Pixel via `pip`. It is recommended to install the OpenCV version for better performance.

```bash
# Recommended: Fast version with OpenCV support
pip install "perfect-pixel[opencv]"

# Numpy version: Lightweight (NumPy only)
pip install perfect-pixel

# Optional plots used by debug=True and example.py --show/--debug
pip install "perfect-pixel[opencv,debug]"
```

## ComfyUI

A ComfyUI custom node is available for integrating Perfect Pixel into ComfyUI workflows.

- Uses the same Perfect Pixel API and RGB input convention
- Provides a ComfyUI-friendly interface for pixel art refinement

- [`Learn how to use Perfect Pixel as a ComfyUI node`](integrations/comfyui/README.md)

## Usage 

### Step 1: Get pixel style image
First you need extra tools to get a pixel styled image. **The recommanded size is between 512 to 1024.**

You can use Stable Diffusion with any Pixel Style Lora, or you can use ChatGPT or Gemini to generate one.


For example, I used ChatGPT to transfer an image into pixel style.

```
prompt: Convert the input image into a TRUE perler bead pixel pattern designed for physical bead crafting, not digital illustration. Canvas size must be exactly 32×32 pixels OR 16×16 pixels, where each pixel represents exactly one perler bead. Use extremely large, chunky pixels with very few active pixels overall. Simplicity is critical. Only keep the main subject. Remove the entire background. For human characters, make sure the face is flat and no shadow. The subject must be centered with clear empty bead rows around all edges to allow easy mounting on a bead board. Add a clean, continuous dark outline around the subject so the silhouette is clearly readable when made with beads. Use a very limited solid color palette (maximum 6–8 colors total). No gradients, no shading, no lighting, no dithering, no texture. No anti-aliasing or smoothing — every pixel must be a perfect square bead aligned to the grid. The output image should be pixel-perfect, each grid only contains one color. Background must be pure solid white.
```

<img src="https://github.com/theamusing/perfectPixel/raw/main/assets/generated.png" width="50%" />

The image is in pixel style but the grids are distorted. Also we don't know the number of grids.

### Step 2: Use Perfect Pixel to refine your image

```python
import cv2
from perfect_pixel import get_perfect_pixel

bgr = cv2.imread("images/avatar.png", cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

w, h, out = get_perfect_pixel(rgb)
```

<img src="https://github.com/theamusing/perfectPixel/raw/main/assets/refined2.png" width="50%" />

*Also see [example.py](./example.py).*
```bash
python example.py

# Optional interactive plots; requires the debug extra above
python example.py --show --debug
```

The grid size is automatically detected, and the image is refined.

<img src="https://github.com/theamusing/perfectPixel/raw/main/assets/process2.png" width="100%" />

Try integrate it into your own projects!

## API Reference
| Args | Description | 
| :--- | :--- |
| **image** | `Non-empty, finite, real numeric NumPy RGB array shaped (H, W, 3), normally uint8 in 0..255 or floats in 0..1 / 0..255.` |
| **sample_method** | `"center", "median" or "majority"` |
| **grid_size** | `Positive integer cell counts (grid_w, grid_h), no greater than input width/height. Preserves exactly this output size and overrides min_size and fix_square.` |
| **min_size** | `Minimum average source-pixel size of an automatic grid; enforced after detection and refinement. Must be finite and positive.` |
| **peak_width** | `Positive integer minimum peak width for FFT peak detection.` |
| **refine_intensity** | `Finite number in [0, 0.5]. Each grid line searches within +/- cell_size * refine_intensity. Zero disables refinement.` |
| **fix_square** | `Whether to adjust an almost-square automatic result to a square. Ignored for a manual grid_size.` |
| **debug** | `Whether to show debug plots. Requires pip install "perfect-pixel[debug]".` |

| Returns | Description |
| :--- | :--- |
| **refined_w** | `Width of the refined image` |
| **refined_h** | `Height of the refined image` |
| **scaled_image** | `Refined NumPy image shaped (refined_h, refined_w, 3)` |

Invalid image shapes or parameter values raise `ValueError`. If automatic detection cannot find a valid grid, the function returns `(None, None, original_image)`; callers should check both dimensions before treating the result as refined. Normal processing does not print status messages; enable Python debug logging to inspect detection decisions.

A manual grid is useful when automatic detection is uncertain:

~~~python
w, h, out = get_perfect_pixel(rgb, grid_size=(16, 16))
assert out.shape == (16, 16, 3)
~~~

## Development and tests

~~~bash
python -m pip install -e ".[test,opencv,debug]"
python -m pytest
python -m build
~~~

The tests cover NumPy-only and OpenCV processing, all samplers, regression cases, and packaging metadata. Source distributions include the test fixtures, sample images, example, and ComfyUI files. Packaging tests build an sdist offline; CI also installs and runs the unpacked source tree outside the Git checkout. ComfyUI tests run when PyTorch is available (it is supplied by a ComfyUI installation); CI also runs a separate CPU PyTorch integration job. The CI matrix includes Linux, Windows, macOS, Python 3.8 compatibility, and the minimum NumPy dependency.

## Algorithm

<img src="https://github.com/theamusing/perfectPixel/raw/main/assets/algorithm.png" width="100%" />

The whole algorithm mainly contains 3 steps:
1. Detect grid size from FFT magnitude of the original image and generate grids.
2. Detect edges using Sobel and refine the grids by aligning them to edges.
3. Use the grids to sample the original image and to get the scaled image.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=theamusing/perfectPixel&type=date&legend=top-left)](https://www.star-history.com/#theamusing/perfectPixel&type=date&legend=top-left)

Thanks so much!









