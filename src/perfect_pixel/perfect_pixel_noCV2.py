import logging
import numbers

import numpy as np

# ----------------------------
# Small utilities (no cv2)
# ----------------------------

_LOGGER = logging.getLogger(__name__)


def _is_positive_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value > 0


def _validate_positive_real(value, name):
    if (not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive number")


def _validate_intensity(value):
    if (not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or not 0 <= value <= 0.5):
        raise ValueError("refine_intensity must be a finite number in [0, 0.5]")


def _validate_image(image):
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[-1] != 3 or min(image.shape[:2]) < 1):
        raise ValueError("image must be a non-empty RGB array shaped (H, W, 3)")
    if image.dtype.kind not in "uif" or not np.isfinite(image).all():
        raise ValueError("image must contain finite real numeric values")
    # Gradients and their projections are computed in float32 in both backends.
    limit = np.finfo(np.float32).max / (16 * max(image.shape[:2]))
    if np.any(image > limit) or np.any(image < -limit):
        raise ValueError("image values are too large for float32 gradient processing")


def _validate_grid_size(grid_size, width, height):
    try:
        grid_x, grid_y = grid_size
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_size must be a pair of positive integer cell counts") from exc
    if not _is_positive_int(grid_x) or not _is_positive_int(grid_y):
        raise ValueError("grid_size must contain positive integer cell counts")
    if grid_x > width or grid_y > height:
        raise ValueError("grid_size cannot exceed the input image dimensions")
    return int(grid_x), int(grid_y)


def _gradient_peaks(values, rel_thr=0.0, min_dist=1):
    """Represent a flat Sobel maximum by its right-rounded midpoint."""
    values = np.asarray(values).reshape(-1)
    if len(values) < 3 or values.max() < 1e-6:
        return np.empty(0, dtype=np.int32)
    threshold = float(values.max()) * rel_thr
    equal = np.isclose(values[1:], values[:-1], rtol=1e-5, atol=1e-6)
    cuts = np.flatnonzero(~equal) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts - 1, len(values) - 1]
    peaks = []
    for start, end in zip(starts, ends):
        if start == 0 or end == len(values) - 1:
            continue
        peak = float(np.max(values[start:end + 1]))
        if peak >= threshold and peak > values[start - 1] and peak > values[end + 1]:
            index = int((start + end + 1) // 2)
            if not peaks or index - peaks[-1] >= min_dist:
                peaks.append(index)
    return np.asarray(peaks, dtype=np.int32)


def _refine_axis(length, count, gradient, intensity, preserve_count):
    cell = length / count
    peaks = _gradient_peaks(gradient)
    span = cell * intensity
    if preserve_count or intensity == 0 or not len(peaks) or count == 1:
        coords = [0]
        for index in range(1, count):
            origin = length * index / count
            candidate = find_best_grid(origin, span, span, gradient, peaks=peaks) if intensity else round(origin)
            # Reserve at least one source pixel for every remaining cell.
            candidate = np.clip(candidate, coords[-1] + 1, length - (count - index))
            coords.append(int(candidate))
        return coords + [length]

    origin = (count // 2) * cell
    anchor = find_best_grid(origin, cell, cell, gradient, peaks=peaks)
    anchor = int(np.clip(anchor, 1, length - 1))
    coords = {0, anchor, length}
    for direction in (-1, 1):
        previous = anchor
        # Every accepted coordinate advances at least one source pixel.
        for _ in range(length):
            origin = previous + direction * cell
            if origin <= cell / 2 or origin >= length - cell / 2:
                break
            candidate = find_best_grid(origin, span, span, gradient, peaks=peaks)
            lower = previous + 1 if direction > 0 else 1
            upper = length - 1 if direction > 0 else previous - 1
            if lower > upper:
                break
            candidate = int(np.clip(candidate, lower, upper))
            coords.add(candidate)
            previous = candidate
    return sorted(coords)


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8/float -> gray float32"""
    img = image_rgb.astype(np.float32)
    if img.ndim == 2:
        return img
    # assume RGB
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def normalize_minmax(x: np.ndarray, a=0.0, b=1.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32) + a
    y = (x - mn) / (mx - mn)
    return (a + (b - a) * y).astype(np.float32)


def conv2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution (same) for grayscale float32, naive but ok for demo sizes."""
    img = image.astype(np.float32, copy=False)
    k = kernel.astype(np.float32, copy=False)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)

    # sum of shifted windows (vectorized over shifts)
    for dy in range(kh):
        for dx in range(kw):
            w = k[dy, dx]
            if w == 0:
                continue
            out += w * pad[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return out


def sobel_xy(gray: np.ndarray, ksize: int = 3):
    """Return (gx, gy) similar to cv2.Sobel for ksize 3 or 5."""
    if ksize == 3:
        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)
    elif ksize == 5:
        # Common 5x5 Sobel kernel (approx). Good enough for grid refinement.
        kx = np.array([[-5, -4,  0,  4,  5],
                       [-8, -10, 0, 10,  8],
                       [-10,-20, 0, 20, 10],
                       [-8, -10, 0, 10,  8],
                       [-5, -4,  0,  4,  5]], dtype=np.float32)
        ky = kx.T
    else:
        raise ValueError("ksize must be 3 or 5")

    gx = conv2d_same(gray, kx)
    gy = conv2d_same(gray, ky)
    return gx, gy


def magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


# ----------------------------
# Your original logic (ported)
# ----------------------------

def compute_fft_magnitude(gray_image):
    f = np.fft.fft2(gray_image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag = 1 - np.log1p(mag)
    return normalize_minmax(mag, 0.0, 1.0)


def smooth_1d(v, k=17):
    """Smooth a projection without changing its length, including tiny images."""
    k = min(int(k), len(v))
    if k % 2 == 0:
        k -= 1
    if k < 3:
        return v
    sigma = k / 6.0
    x = np.arange(k) - k // 2
    ker = np.exp(-(x * x) / (2 * sigma * sigma))
    ker /= ker.sum()
    return np.convolve(v, ker, mode="same")


def detect_peak(proj, peak_width=6, rel_thr=0.35, min_dist=6):
    """Find paired FFT peaks with a meaningful drop on both local flanks."""
    center = len(proj) // 2
    mx = float(proj.max())
    if mx < 1e-6:
        return None

    thr = mx * float(rel_thr)

    # Flat shoulders around the DC notch have height but no periodic evidence.
    # Use a relative floor to ignore roundoff ripples without imposing a cell-size limit.
    prominence_thr = max(1e-6, float(np.ptp(proj)) * 1e-3)
    radius = max(2, int(peak_width))
    candidates = []
    for i in range(1, len(proj) - 1):
        is_peak = True
        for j in range(1, peak_width):
            if i - j < 0 or i + j >= len(proj):
                continue
            if proj[i - j + 1] < proj[i - j] or proj[i + j - 1] < proj[i + j]:
                is_peak = False
                break
        if is_peak and proj[i] >= thr:
            rise = proj[i] - np.min(proj[max(0, i - radius):i])
            fall = proj[i] - np.min(proj[i + 1:min(len(proj), i + radius + 1)])
            if min(rise, fall) <= prominence_thr:
                continue

            left_climb = 0
            for k in range(i, 0, -1):
                if proj[k] > proj[k - 1]:
                    left_climb = abs(proj[i] - proj[k - 1])
                else:
                    break

            right_fall = 0
            for k in range(i, len(proj) - 1):
                if proj[k] > proj[k + 1]:
                    right_fall = abs(proj[i] - proj[k + 1])
                else:
                    break

            candidates.append({
                "index": i,
                "climb": left_climb,
                "fall": right_fall,
                "score": max(left_climb, right_fall),
            })

    if not candidates:
        return None

    left = [c for c in candidates if c["index"] < center - min_dist and c["index"] > center * 0.25]
    right = [c for c in candidates if c["index"] > center + min_dist and c["index"] < center * 1.75]

    left.sort(key=lambda x: x["score"], reverse=True)
    right.sort(key=lambda x: x["score"], reverse=True)

    if not left or not right:
        return None

    peak_left = left[0]["index"]
    peak_right = right[0]["index"]

    return abs(peak_right - peak_left) / 2


def find_best_grid(origin, range_val_min, range_val_max, grad_mag, thr=0, peaks=None):
    """Find an edge near a grid line, treating a flat Sobel peak as one edge."""
    if peaks is None:
        peaks = _gradient_peaks(grad_mag, rel_thr=thr)
    lower = int(np.ceil(origin - range_val_min))
    upper = int(np.floor(origin + range_val_max))
    candidates = peaks[(peaks >= lower) & (peaks <= upper)]
    if not len(candidates):
        return int(round(origin))
    # Prefer the strongest edge, then the closest one rather than the leftmost.
    order = np.lexsort((np.abs(candidates - origin), -grad_mag[candidates]))
    return int(candidates[order[0]])


def sample_center(image, x_coords, y_coords):
    x = np.asarray(x_coords)
    y = np.asarray(y_coords)

    centers_x = np.clip((x[1:] + x[:-1]) * 0.5, 0, image.shape[1] - 1).astype(np.int32)
    centers_y = np.clip((y[1:] + y[:-1]) * 0.5, 0, image.shape[0] - 1).astype(np.int32)

    scaled_image = image[centers_y[:, None], centers_x[None, :]]
    return scaled_image


def sample_majority(image, x_coords, y_coords, max_samples=128, iters=6, seed=0):
    rng = np.random.default_rng(seed)

    img = image.astype(np.float32) if image.dtype != np.float32 else image
    H, W = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]
    C = img.shape[2]

    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)

    nx, ny = len(x) - 1, len(y) - 1
    out = np.empty((ny, nx, C), dtype=np.float32)

    for j in range(ny):
        y0, y1 = int(y[j]), int(y[j + 1])
        y0 = np.clip(y0, 0, H); y1 = np.clip(y1, 0, H)
        if y1 <= y0: y1 = min(y0 + 1, H)

        for i in range(nx):
            x0, x1 = int(x[i]), int(x[i + 1])
            x0 = np.clip(x0, 0, W); x1 = np.clip(x1, 0, W)
            if x1 <= x0: x1 = min(x0 + 1, W)

            cell = img[y0:y1, x0:x1].reshape(-1, C)
            n = cell.shape[0]
            if n == 0:
                out[j, i] = 0
                continue
            if n > max_samples:
                cell = cell[rng.integers(0, n, size=max_samples)]

            c0 = cell[0]
            c1 = cell[np.argmax(((cell - c0) ** 2).sum(1))]

            for _ in range(iters):
                d0 = ((cell - c0) ** 2).sum(1)
                d1 = ((cell - c1) ** 2).sum(1)
                m1 = d1 < d0
                if np.any(~m1): c0 = cell[~m1].mean(0)
                if np.any(m1):  c1 = cell[m1].mean(0)

            out[j, i] = c1 if m1.sum() >= (~m1).sum() else c0

    if image.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out

def sample_median(image, x_coords, y_coords):
    img = image.astype(np.float32) if image.dtype != np.float32 else image
    H, W = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]
    C = img.shape[2]

    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)

    nx, ny = len(x) - 1, len(y) - 1
    out = np.empty((ny, nx, C), dtype=np.float32)
    
    for j in range(ny):
        y0, y1 = int(y[j]), int(y[j + 1])
        y0 = np.clip(y0, 0, H); y1 = np.clip(y1, 0, H)
        if y1 <= y0: y1 = min(y0 + 1, H)

        for i in range(nx):
            x0, x1 = int(x[i]), int(x[i + 1])
            x0 = np.clip(x0, 0, W); x1 = np.clip(x1, 0, W)
            if x1 <= x0: x1 = min(x0 + 1, W)

            cell = img[y0:y1, x0:x1].reshape(-1, C)
            if cell.shape[0] == 0:
                out[j, i] = 0
            else:
                out[j, i] = np.median(cell, axis=0)

    if image.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out

def refine_grids(image, grid_x, grid_y, refine_intensity=0.25, preserve_count=False):
    """Refine bounded grid lines; manual grids retain their requested cell count."""
    H, W = image.shape[:2]
    grid_x, grid_y = _validate_grid_size((grid_x, grid_y), W, H)
    _validate_intensity(refine_intensity)
    gray = rgb_to_gray(image)
    gx, gy = sobel_xy(gray, ksize=3)
    grad_x = np.sum(np.abs(gx), axis=0).reshape(-1)
    grad_y = np.sum(np.abs(gy), axis=1).reshape(-1)
    return (
        _refine_axis(W, grid_x, grad_x, refine_intensity, preserve_count),
        _refine_axis(H, grid_y, grad_y, refine_intensity, preserve_count),
    )

def estimate_grid_fft(gray, peak_width=6):
    """Return (grid_w, grid_h), or (None, None) when FFT detection fails."""
    H, W = gray.shape
    if min(H, W) < 3 or float(gray.max()) - float(gray.min()) < 1e-6:
        return None, None

    mag = compute_fft_magnitude(gray)

    band_row = W // 2
    band_col = H // 2
    row_sum = np.sum(mag[:, W//2 - band_row: W//2 + band_row], axis=1)
    col_sum = np.sum(mag[H//2 - band_col: H//2 + band_col, :], axis=0)

    row_sum = normalize_minmax(row_sum, 0.0, 1.0).flatten()
    col_sum = normalize_minmax(col_sum, 0.0, 1.0).flatten()

    row_sum = smooth_1d(row_sum, k=17)
    col_sum = smooth_1d(col_sum, k=17)

    scale_row = detect_peak(row_sum, peak_width=peak_width)
    scale_col = detect_peak(col_sum, peak_width=peak_width)

    if scale_row is None or scale_col is None or scale_col <= 0 or scale_row <= 0:
        return None, None

    return int(round(scale_col)), int(round(scale_row))

def estimate_grid_gradient(gray, rel_thr=0.2):
    """Estimate grid counts from the spacing of distinct (possibly flat) edges."""
    H, W = gray.shape
    gx, gy = sobel_xy(gray, ksize=3)
    peak_x = _gradient_peaks(np.sum(np.abs(gx), axis=0), rel_thr, min_dist=4)
    peak_y = _gradient_peaks(np.sum(np.abs(gy), axis=1), rel_thr, min_dist=4)
    if len(peak_x) < 4 or len(peak_y) < 4:
        return None, None
    scale_x = W / np.median(np.diff(peak_x))
    scale_y = H / np.median(np.diff(peak_y))
    _LOGGER.debug("Detected grid size from gradient: (%.2f, %.2f)", scale_x, scale_y)
    return int(round(scale_x)), int(round(scale_y))

def detect_grid_scale(image, peak_width=6, max_ratio=1.5, min_size=4.0):
    gray = rgb_to_gray(image)
    H, W = gray.shape

    def valid_size(grid_w, grid_h):
        return (
            grid_w is not None and grid_h is not None
            and np.isfinite(grid_w) and np.isfinite(grid_h)
            and grid_w > 0 and grid_h > 0
            and W / grid_w >= min_size and H / grid_h >= min_size
        )

    grid_w, grid_h = estimate_grid_fft(gray, peak_width=peak_width)
    valid_fft = valid_size(grid_w, grid_h)
    if valid_fft:
        px, py = W / grid_w, H / grid_h
        valid_fft = max(px, py) / min(px, py) <= max_ratio
    if not valid_fft:
        _LOGGER.debug("FFT grid estimation failed validation; trying gradient detection.")
        grid_w, grid_h = estimate_grid_gradient(gray)
    if not valid_size(grid_w, grid_h):
        _LOGGER.debug("No grid satisfies min_size=%s.", min_size)
        return None, None

    px, py = W / grid_w, H / grid_h
    pixel_size = min(px, py) if max(px, py) / min(px, py) > max_ratio else (px + py) / 2
    grid_w, grid_h = int(round(W / pixel_size)), int(round(H / pixel_size))
    if not valid_size(grid_w, grid_h):
        return None, None
    _LOGGER.debug("Detected pixel size: %.2f", pixel_size)
    return grid_w, grid_h

def grid_layout(image, x_coords, y_coords, scale_x, scale_y):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        raise ImportError(
            'debug=True requires matplotlib; install "perfect-pixel[debug]".'
        ) from exc
    plt.figure()
    plt.imshow(image)
    plt.title(f"Scaled Image by Grid Sampling({scale_x}x{scale_y})")
    for x in x_coords:
        plt.axvline(x=x, linewidth=0.6)
    for y in y_coords:
        plt.axhline(y=y, linewidth=0.6)
    plt.show()

def get_perfect_pixel(image, sample_method="center", grid_size=None, min_size=4.0,
                      peak_width=6, refine_intensity=0.25, fix_square=True, debug=False):
    """Detect and sample an RGB image into a pixel grid.

    image must be a non-empty, finite real-valued NumPy array shaped (H, W, 3).
    sample_method is "center", "median", or "majority". A manual grid_size is
    a pair of positive integer cell counts no larger than the input dimensions;
    it preserves the requested output size and overrides min_size/fix_square.
    min_size is the minimum average source-pixel size of an automatic grid.
    peak_width is a positive integer. refine_intensity must be in [0, 0.5];
    each line searches within +/- cell_size * refine_intensity. Zero disables
    refinement. fix_square adjusts almost-square automatic outputs only.
    debug=True shows a grid plot and requires the optional [debug] extra.

    Return (width, height, image), or (None, None, original_image) when no valid
    automatic grid is found. Invalid input or parameters raise ValueError.
    """
    _validate_image(image)
    H, W = image.shape[:2]
    if sample_method not in ("center", "median", "majority"):
        raise ValueError("sample_method must be 'center', 'median', or 'majority'")
    _validate_positive_real(min_size, "min_size")
    if not _is_positive_int(peak_width):
        raise ValueError("peak_width must be a positive integer")
    _validate_intensity(refine_intensity)

    manual = grid_size is not None
    if manual:
        size_x, size_y = _validate_grid_size(grid_size, W, H)
    else:
        size_x, size_y = detect_grid_scale(
            image, peak_width=peak_width, max_ratio=1.5, min_size=min_size)
        if size_x is None or size_y is None:
            _LOGGER.debug("Failed to estimate a valid grid size.")
            return None, None, image

    x_coords, y_coords = refine_grids(
        image, size_x, size_y, refine_intensity, preserve_count=manual)
    sampler = {"center": sample_center, "median": sample_median, "majority": sample_majority}
    scaled_image = sampler[sample_method](image, x_coords, y_coords)
    refined_h, refined_w = scaled_image.shape[:2]

    if fix_square and not manual and abs(refined_w - refined_h) == 1:
        if refined_w > refined_h:
            if refined_w % 2:
                scaled_image = scaled_image[:, :-1]
            else:
                scaled_image = np.concatenate([scaled_image[:1, :], scaled_image], axis=0)
        elif refined_h % 2:
            scaled_image = scaled_image[:-1, :]
        else:
            scaled_image = np.concatenate([scaled_image[:, :1], scaled_image], axis=1)

    refined_h, refined_w = scaled_image.shape[:2]
    if not manual and min(W / refined_w, H / refined_h) < min_size:
        _LOGGER.debug("Refined grid is smaller than min_size=%s.", min_size)
        return None, None, image
    _LOGGER.debug("Refined grid size: (%d, %d)", refined_w, refined_h)
    if debug:
        grid_layout(image, x_coords, y_coords, refined_w, refined_h)
    return refined_w, refined_h, scaled_image
