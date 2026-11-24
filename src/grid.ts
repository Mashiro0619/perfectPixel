// Core grid detection & refinement ported from Mashiro0619/perfectPixel @ 6e2de13
// (originally by theAmusing and contributors, MIT). Optimized for browser/Node TypedArrays.
import { fftMagnitude } from "./fft.ts";
import type { GradientProjections } from "./types.ts";

const f32 = Math.fround;

export function clamp(n: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, n));
}

/** Ties-to-even rounding matching Python round() and NumPy np.rint(). */
export function roundEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  return fraction === 0.5
    ? floor + (floor % 2 !== 0 ? 1 : 0)
    : Math.round(value);
}

export function median(values: number[]): number {
  if (!values.length) return 0;
  values.sort((a, b) => a - b);
  const n = values.length;
  const mid = Math.floor(n / 2);
  return n % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

export function normalize(values: ArrayLike<number>): Float32Array {
  const len = values.length;
  const result = new Float32Array(len);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < len; i++) {
    const v = f32(values[i]);
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (max - min < 1e-8) return result;
  const range = f32(max - min);
  for (let i = 0; i < len; i++) {
    result[i] = f32(f32(values[i]) - min) / range;
  }
  return result;
}

export function smooth1d(values: ArrayLike<number>, kernelSize = 17): Float64Array {
  let k = Math.min(Math.trunc(kernelSize), values.length);
  if (k % 2 === 0) k--;
  if (k < 3) return Float64Array.from(values);

  const kernel = new Float64Array(k);
  const half = Math.floor(k / 2);
  const sigma = k / 6;
  let total = 0;
  for (let i = 0; i < k; i++) {
    const d = i - half;
    const w = Math.exp(-(d * d) / (2 * sigma * sigma));
    kernel[i] = w;
    total += w;
  }
  for (let i = 0; i < k; i++) kernel[i] /= total;

  const len = values.length;
  const result = new Float64Array(len);
  for (let i = 0; i < len; i++) {
    let sum = 0;
    const jStart = Math.max(0, half - i);
    const jEnd = Math.min(k - 1, len - 1 - i + half);
    for (let j = jStart; j <= jEnd; j++) {
      sum += values[i + j - half] * kernel[j];
    }
    result[i] = sum;
  }
  return result;
}

/**
 * Locate 1D edge peaks, representing flat Sobel plateaus as a single peak at
 * their right-rounded midpoint.
 */
export function gradientPeaks(
  values: ArrayLike<number>,
  relativeThreshold = 0,
  minDistance = 1,
): number[] {
  const len = values.length;
  let max = 0;
  for (let i = 0; i < len; i++) {
    if (values[i] > max) max = values[i];
  }
  if (len < 3 || max < 1e-6) return [];

  const threshold = max * relativeThreshold;
  const peaks: number[] = [];
  let start = 0;
  for (let end = 0; end < len; end++) {
    if (
      end < len - 1 &&
      Math.abs(values[end + 1] - values[end]) <=
        1e-6 + 1e-5 * Math.abs(values[end])
    ) {
      continue;
    }
    if (start > 0 && end < len - 1) {
      let peak = -Infinity;
      for (let i = start; i <= end; i++) {
        if (values[i] > peak) peak = values[i];
      }
      const index = Math.floor((start + end + 1) / 2);
      if (
        peak >= threshold &&
        peak > values[start - 1] &&
        peak > values[end + 1] &&
        (!peaks.length || index - peaks[peaks.length - 1] >= minDistance)
      ) {
        peaks.push(index);
      }
    }
    start = end + 1;
  }
  return peaks;
}

/**
 * Find paired symmetric peaks in a smoothed 1D FFT projection with local flank
 * prominence filtering so flat DC shoulders never masquerade as grid peaks.
 */
export function detectPeak(
  projection: ArrayLike<number>,
  peakWidth = 6,
  relativeThreshold = 0.35,
  minDist = 6,
): number | null {
  const len = projection.length;
  const center = Math.floor(len / 2);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < len; i++) {
    const v = projection[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (max < 1e-6) return null;

  const thr = max * relativeThreshold;
  const radius = Math.max(2, Math.trunc(peakWidth));
  const prominence = Math.max(1e-6, (max - min) * 1e-3);
  const maxStep = Math.min(peakWidth, len);
  const candidates: Array<{ index: number; score: number }> = [];

  for (let i = 1; i < len - 1; i++) {
    if (projection[i] < thr) continue;
    let isPeak = true;
    for (let j = 1; j < maxStep; j++) {
      if (i - j < 0 || i + j >= len) continue;
      if (
        projection[i - j + 1] < projection[i - j] ||
        projection[i + j - 1] < projection[i + j]
      ) {
        isPeak = false;
        break;
      }
    }
    if (!isPeak) continue;

    let leftMin = Infinity;
    for (let j = Math.max(0, i - radius); j < i; j++) {
      if (projection[j] < leftMin) leftMin = projection[j];
    }
    let rightMin = Infinity;
    for (let j = i + 1; j < Math.min(len, i + radius + 1); j++) {
      if (projection[j] < rightMin) rightMin = projection[j];
    }
    if (Math.min(projection[i] - leftMin, projection[i] - rightMin) <= prominence) {
      continue;
    }

    let leftClimb = 0;
    for (let k = i; k > 0; k--) {
      if (projection[k] > projection[k - 1]) {
        leftClimb = Math.abs(projection[i] - projection[k - 1]);
      } else {
        break;
      }
    }
    let rightFall = 0;
    for (let k = i; k < len - 1; k++) {
      if (projection[k] > projection[k + 1]) {
        rightFall = Math.abs(projection[i] - projection[k + 1]);
      } else {
        break;
      }
    }
    candidates.push({ index: i, score: Math.max(leftClimb, rightFall) });
  }

  let peakLeft: number | null = null;
  let peakLeftScore = -Infinity;
  let peakRight: number | null = null;
  let peakRightScore = -Infinity;

  for (const item of candidates) {
    if (
      item.index < center - minDist &&
      item.index > center * 0.25 &&
      item.score > peakLeftScore
    ) {
      peakLeft = item.index;
      peakLeftScore = item.score;
    } else if (
      item.index > center + minDist &&
      item.index < center * 1.75 &&
      item.score > peakRightScore
    ) {
      peakRight = item.index;
      peakRightScore = item.score;
    }
  }

  if (peakLeft === null || peakRight === null) return null;
  return Math.abs(peakRight - peakLeft) / 2;
}

/**
 * Single-pass 3x3 Sobel gradient projections (reflect101 boundary matching
 * OpenCV / NumPy port) without allocating full HxW gx/gy scratch arrays.
 */
export function gradientProjections(
  gray: Float32Array,
  width: number,
  height: number,
): GradientProjections {
  const projX = new Float32Array(width);
  const projY = new Float32Array(height);
  if (width < 2 && height < 2) return { x: projX, y: projY };

  const reflect = (index: number, limit: number): number => {
    if (limit <= 1) return 0;
    let i = index;
    while (i < 0 || i >= limit) {
      i = i < 0 ? -i : 2 * limit - 2 - i;
    }
    return i;
  };

  for (let y = 0; y < height; y++) {
    const r0 = reflect(y - 1, height) * width;
    const r1 = y * width;
    const r2 = reflect(y + 1, height) * width;
    let rowGradY = 0;

    for (let x = 0; x < width; x++) {
      const c0 = reflect(x - 1, width);
      const c1 = x;
      const c2 = reflect(x + 1, width);

      const p00 = gray[r0 + c0];
      const p01 = gray[r0 + c1];
      const p02 = gray[r0 + c2];
      const p10 = gray[r1 + c0];
      const p12 = gray[r1 + c2];
      const p20 = gray[r2 + c0];
      const p21 = gray[r2 + c1];
      const p22 = gray[r2 + c2];

      const gx = f32(
        f32(p02 - p00) + f32(2 * f32(p12 - p10)) + f32(p22 - p20),
      );
      const gy = f32(
        f32(p20 - p00) + f32(2 * f32(p21 - p01)) + f32(p22 - p02),
      );

      projX[x] = f32(projX[x] + Math.abs(gx));
      rowGradY = f32(rowGradY + Math.abs(gy));
    }
    projY[y] = rowGradY;
  }
  return { x: projX, y: projY };
}

function estimateGridFft(
  gray: Float32Array,
  width: number,
  height: number,
  peakWidth: number,
): [number, number] | null {
  if (Math.min(width, height) < 3) return null;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (max - min < 1e-6) return null;

  const mag = normalize(fftMagnitude(gray, width, height));
  const rowRaw = new Float32Array(height);
  const colRaw = new Float32Array(width);
  const bandWidth = width - (width % 2);
  const bandHeight = height - (height % 2);
  for (let y = 0; y < height; y++) {
    const rowStart = y * width;
    let rowSum = 0;
    for (let x = 0; x < bandWidth; x++) {
      rowSum = f32(rowSum + mag[rowStart + x]);
    }
    rowRaw[y] = rowSum;
    if (y < bandHeight) {
      for (let x = 0; x < width; x++) {
        colRaw[x] = f32(colRaw[x] + mag[rowStart + x]);
      }
    }
  }

  const rowNorm = normalize(rowRaw);
  const colNorm = normalize(colRaw);
  const rowProj = smooth1d(rowNorm, 17);
  const colProj = smooth1d(colNorm, 17);

  const scaleRow = detectPeak(rowProj, peakWidth);
  const scaleCol = detectPeak(colProj, peakWidth);
  if (scaleRow === null || scaleCol === null || scaleCol <= 0 || scaleRow <= 0) {
    return null;
  }
  return [roundEven(scaleCol), roundEven(scaleRow)];
}

function estimateGridGradient(
  gradients: GradientProjections,
  width: number,
  height: number,
  relThreshold = 0.2,
): [number, number] | null {
  const peakX = gradientPeaks(gradients.x, relThreshold, 4);
  const peakY = gradientPeaks(gradients.y, relThreshold, 4);
  if (peakX.length < 4 || peakY.length < 4) return null;

  const diffsX: number[] = [];
  for (let i = 1; i < peakX.length; i++) diffsX.push(peakX[i] - peakX[i - 1]);
  const diffsY: number[] = [];
  for (let i = 1; i < peakY.length; i++) diffsY.push(peakY[i] - peakY[i - 1]);

  const medX = median(diffsX);
  const medY = median(diffsY);
  if (medX <= 0 || medY <= 0) return null;
  return [roundEven(width / medX), roundEven(height / medY)];
}

export function detectGrid(
  gray: Float32Array,
  width: number,
  height: number,
  minSize: number,
  peakWidth: number,
  getGradients: () => GradientProjections,
  maxRatio = 1.5,
): [number, number] | null {
  const validSize = (grid: [number, number] | null): grid is [number, number] =>
    grid !== null &&
    Number.isFinite(grid[0]) &&
    Number.isFinite(grid[1]) &&
    grid[0] > 0 &&
    grid[1] > 0 &&
    width / grid[0] >= minSize &&
    height / grid[1] >= minSize;

  let grid = estimateGridFft(gray, width, height, peakWidth);
  let validFft = validSize(grid);
  if (validFft && grid) {
    const px = width / grid[0];
    const py = height / grid[1];
    validFft = Math.max(px, py) / Math.min(px, py) <= maxRatio;
  }
  if (!validFft) {
    grid = estimateGridGradient(getGradients(), width, height);
  }
  if (!validSize(grid)) return null;

  const px = width / grid[0];
  const py = height / grid[1];
  const pixelSize =
    Math.max(px, py) / Math.min(px, py) > maxRatio
      ? Math.min(px, py)
      : (px + py) / 2;
  const resolved: [number, number] = [
    roundEven(width / pixelSize),
    roundEven(height / pixelSize),
  ];
  return validSize(resolved) ? resolved : null;
}

export function bestGrid(
  origin: number,
  span: number,
  gradient: ArrayLike<number>,
  peaks: number[],
): number {
  const lower = Math.ceil(origin - span);
  const upper = Math.floor(origin + span);
  let best = -1;
  let bestGrad = -Infinity;
  let bestDist = Infinity;

  for (let i = 0; i < peaks.length; i++) {
    const candidate = peaks[i];
    if (candidate < lower) continue;
    if (candidate > upper) break;
    const g = gradient[candidate];
    const dist = Math.abs(candidate - origin);
    if (g > bestGrad || (g === bestGrad && dist < bestDist)) {
      best = candidate;
      bestGrad = g;
      bestDist = dist;
    }
  }
  return best >= 0 ? best : roundEven(origin);
}

/**
 * Refine axis grid coordinates within [0, length], strictly monotonically
 * increasing. Manual grids strictly preserve their requested cell count.
 */
export function refineAxis(
  length: number,
  count: number,
  gradient: ArrayLike<number>,
  intensity: number,
  preserveCount: boolean,
): number[] {
  const cell = length / count;
  const peaks = gradientPeaks(gradient);
  const span = cell * intensity;

  if (preserveCount || intensity === 0 || !peaks.length || count === 1) {
    const coords = [0];
    for (let i = 1; i < count; i++) {
      const origin = (length * i) / count;
      const candidate = intensity
        ? bestGrid(origin, span, gradient, peaks)
        : roundEven(origin);
      coords.push(clamp(candidate, coords[coords.length - 1] + 1, length - (count - i)));
    }
    coords.push(length);
    return coords;
  }

  const anchor = clamp(
    bestGrid(Math.floor(count / 2) * cell, cell, gradient, peaks),
    1,
    length - 1,
  );
  const coords = new Set<number>([0, anchor, length]);

  for (const direction of [-1, 1] as const) {
    let previous = anchor;
    for (let i = 0; i < length; i++) {
      const origin = previous + direction * cell;
      if (origin <= cell / 2 || origin >= length - cell / 2) break;
      const lower = direction > 0 ? previous + 1 : 1;
      const upper = direction > 0 ? length - 1 : previous - 1;
      if (lower > upper) break;
      previous = clamp(bestGrid(origin, span, gradient, peaks), lower, upper);
      coords.add(previous);
    }
  }

  return Array.from(coords).sort((a, b) => a - b);
}
