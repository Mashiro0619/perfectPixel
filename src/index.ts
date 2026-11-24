// Standalone Web/Node TypeScript engine refactored from Mashiro0619/perfectPixel @ 6e2de13.
// Original algorithm: theAmusing and contributors, MIT. See ../NOTICE.md.
import {
  clamp,
  detectGrid,
  gradientProjections,
  refineAxis,
  roundEven,
} from "./grid.ts";
import { DEFAULT_OPTIONS, INPUT_LIMITS } from "./types.ts";
import type {
  GradientProjections,
  PixelImage,
  PixelOptions,
  PixelResult,
  PreparedPixelImage,
  ProcessingStage,
} from "./types.ts";

export { DEFAULT_OPTIONS, INPUT_LIMITS, EXPORT_PIXEL_LIMIT } from "./types.ts";
export type {
  GradientProjections,
  GridLines,
  PixelImage,
  PixelOptions,
  PixelResult,
  PreparedPixelImage,
  ProcessingStage,
  SampleMethod,
  WorkerRequest,
  WorkerResponse,
} from "./types.ts";

const f32 = Math.fround;
const W_R = f32(0.299);
const W_G = f32(0.587);
const W_B = f32(0.114);

const positiveInt = (n: unknown): n is number =>
  typeof n === "number" && Number.isSafeInteger(n) && n > 0;

function nowMs(): number {
  return typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();
}

export function validateImage(image: PixelImage): void {
  if (
    !image ||
    !positiveInt(image.width) ||
    !positiveInt(image.height) ||
    image.width > INPUT_LIMITS.side ||
    image.height > INPUT_LIMITS.side ||
    image.width * image.height > INPUT_LIMITS.pixels ||
    !(image.data instanceof Uint8ClampedArray) ||
    image.data.length !== image.width * image.height * 4
  ) {
    throw new Error("invalid-image");
  }
}

export function validateInput(image: PixelImage, options: PixelOptions): void {
  validateImage(image);
  if (
    options.sampleMethod !== "center" &&
    options.sampleMethod !== "median" &&
    options.sampleMethod !== "majority"
  ) {
    throw new Error("invalid-sample-method");
  }
  if (
    typeof options.minSize !== "number" ||
    !Number.isFinite(options.minSize) ||
    options.minSize <= 0
  ) {
    throw new Error("invalid-min-size");
  }
  if (!positiveInt(options.peakWidth)) throw new Error("invalid-peak-width");
  if (
    typeof options.refineIntensity !== "number" ||
    !Number.isFinite(options.refineIntensity) ||
    options.refineIntensity < 0 ||
    options.refineIntensity > 0.5
  ) {
    throw new Error("invalid-refinement");
  }
  if (typeof options.fixSquare !== "boolean") {
    throw new Error("invalid-square-option");
  }
  if (
    options.gridSize !== null &&
    (!Array.isArray(options.gridSize) ||
      options.gridSize.length !== 2 ||
      !positiveInt(options.gridSize[0]) ||
      !positiveInt(options.gridSize[1]) ||
      options.gridSize[0] > image.width ||
      options.gridSize[1] > image.height)
  ) {
    throw new Error("invalid-grid");
  }
}

/**
 * Single-pass RGB-over-white grayscale conversion + transparency detection
 * with zero per-pixel object allocations.
 */
function analyzeSource(image: PixelImage): {
  transparent: boolean;
  rgbGray: Float32Array;
} {
  const pixelCount = image.width * image.height;
  const rgbGray = new Float32Array(pixelCount);
  const data = image.data;
  let transparent = false;

  for (let p = 0, i = 0; p < pixelCount; p++, i += 4) {
    const a = data[i + 3];
    if (a === 255) {
      rgbGray[p] = f32(
        f32(f32(W_R * data[i]) + f32(W_G * data[i + 1])) + f32(W_B * data[i + 2]),
      );
    } else {
      transparent = true;
      const alpha = a / 255;
      const bg = 255 - a;
      const r = f32(data[i] * alpha + bg);
      const g = f32(data[i + 1] * alpha + bg);
      const b = f32(data[i + 2] * alpha + bg);
      rgbGray[p] = f32(f32(f32(W_R * r) + f32(W_G * g)) + f32(W_B * b));
    }
  }

  return { transparent, rgbGray };
}

function extractAlphaGray(image: PixelImage): Float32Array {
  const pixelCount = image.width * image.height;
  const gray = new Float32Array(pixelCount);
  const data = image.data;
  for (let p = 0, i = 3; p < pixelCount; p++, i += 4) {
    gray[p] = data[i];
  }
  return gray;
}

/**
 * Prepare a reusable per-image analysis handle. Subsequent parameter changes
 * via `processPreparedImage` skip repeated grayscale, FFT, and Sobel work.
 */
export function preparePixelImage(image: PixelImage): PreparedPixelImage {
  validateImage(image);
  const { transparent, rgbGray } = analyzeSource(image);
  return {
    image,
    transparent,
    rgbGray,
    alphaGray: null,
    rgbGradients: null,
    alphaGradients: null,
    rgbGridCache: new Map(),
    alphaGridCache: new Map(),
  };
}

function getRgbGradients(prepared: PreparedPixelImage): GradientProjections {
  if (!prepared.rgbGradients) {
    prepared.rgbGradients = gradientProjections(
      prepared.rgbGray,
      prepared.image.width,
      prepared.image.height,
    );
  }
  return prepared.rgbGradients;
}

function getAlphaGray(prepared: PreparedPixelImage): Float32Array {
  if (!prepared.alphaGray) {
    prepared.alphaGray = extractAlphaGray(prepared.image);
  }
  return prepared.alphaGray;
}

function getAlphaGradients(prepared: PreparedPixelImage): GradientProjections {
  if (!prepared.alphaGradients) {
    prepared.alphaGradients = gradientProjections(
      getAlphaGray(prepared),
      prepared.image.width,
      prepared.image.height,
    );
  }
  return prepared.alphaGradients;
}

function createRng(seed = 0): () => number {
  let state = (seed + 0x9e3779b9) >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}

function sampleGrid(
  image: PixelImage,
  xs: number[],
  ys: number[],
  method: PixelOptions["sampleMethod"],
  transparent: boolean,
  progress: (fraction: number) => void,
): PixelImage {
  const outWidth = xs.length - 1;
  const outHeight = ys.length - 1;
  const srcWidth = image.width;
  const srcHeight = image.height;
  const src = image.data;
  const out = new Uint8ClampedArray(outWidth * outHeight * 4);

  if (method === "center") {
    for (let row = 0; row < outHeight; row++) {
      const cy = clamp(Math.floor((ys[row] + ys[row + 1]) / 2), 0, srcHeight - 1);
      const srcRowBase = cy * srcWidth;
      const dstRowBase = row * outWidth;
      for (let col = 0; col < outWidth; col++) {
        const cx = clamp(Math.floor((xs[col] + xs[col + 1]) / 2), 0, srcWidth - 1);
        const s = (srcRowBase + cx) * 4;
        const d = (dstRowBase + col) * 4;
        const a = src[s + 3];
        if (a === 0) {
          out[d] = 0;
          out[d + 1] = 0;
          out[d + 2] = 0;
          out[d + 3] = 0;
        } else {
          out[d] = src[s];
          out[d + 1] = src[s + 1];
          out[d + 2] = src[s + 2];
          out[d + 3] = a;
        }
      }
      if (row % 8 === 0 || row === outHeight - 1) progress((row + 1) / outHeight);
    }
    return { width: outWidth, height: outHeight, data: out };
  }

  // Hoisted scratch buffers eliminate per-cell typed array allocations.
  const histogram = new Uint32Array(256);
  let medianScratch = new Float32Array(64);
  const samples = new Float32Array(128 * 4);
  const center0 = new Float32Array(4);
  const center1 = new Float32Array(4);
  const sum0 = new Float32Array(4);
  const sum1 = new Float32Array(4);
  const result = new Float32Array(4);
  const random = createRng();

  for (let row = 0; row < outHeight; row++) {
    const y0 = ys[row];
    const y1 = ys[row + 1];
    const cellH = y1 - y0;
    const dstRowBase = row * outWidth;

    for (let col = 0; col < outWidth; col++) {
      const x0 = xs[col];
      const x1 = xs[col + 1];
      const cellW = x1 - x0;
      const count = cellW * cellH;
      const target = (dstRowBase + col) * 4;

      if (method === "median") {
        if (!transparent) {
          const lo = Math.floor((count - 1) / 2);
          const hi = Math.floor(count / 2);
          for (let c = 0; c < 3; c++) {
            histogram.fill(0);
            for (let py = y0; py < y1; py++) {
              let idx = (py * srcWidth + x0) * 4 + c;
              for (let px = 0; px < cellW; px++, idx += 4) {
                histogram[src[idx]]++;
              }
            }
            let running = 0;
            let left = -1;
            let right = 0;
            for (let v = 0; v < 256; v++) {
              running += histogram[v];
              if (left < 0 && running > lo) left = v;
              if (running > hi) {
                right = v;
                break;
              }
            }
            result[c] = (left + right) / 2;
          }
          result[3] = 255;
        } else {
          if (medianScratch.length < count) {
            medianScratch = new Float32Array(count);
          }
          const slice = medianScratch.subarray(0, count);
          const half = Math.floor(count / 2);
          for (let c = 0; c < 4; c++) {
            let n = 0;
            for (let py = y0; py < y1; py++) {
              let idx = (py * srcWidth + x0) * 4;
              for (let px = 0; px < cellW; px++, idx += 4) {
                slice[n++] =
                  c === 3
                    ? src[idx + 3]
                    : f32(src[idx + c] * (src[idx + 3] / 255));
              }
            }
            slice.sort();
            result[c] =
              count % 2 ? slice[half] : f32(slice[half - 1] + slice[half]) / 2;
          }
        }
      } else {
        // Majority 2-means clustering on up to 128 sampled cell pixels.
        const size = Math.min(count, 128);
        if (count <= 128) {
          let n = 0;
          for (let py = y0; py < y1; py++) {
            let idx = (py * srcWidth + x0) * 4;
            for (let px = 0; px < cellW; px++, idx += 4) {
              const dst = n * 4;
              if (transparent) {
                const a = src[idx + 3];
                const alpha = a / 255;
                samples[dst] = f32(src[idx] * alpha);
                samples[dst + 1] = f32(src[idx + 1] * alpha);
                samples[dst + 2] = f32(src[idx + 2] * alpha);
                samples[dst + 3] = a;
              } else {
                samples[dst] = src[idx];
                samples[dst + 1] = src[idx + 1];
                samples[dst + 2] = src[idx + 2];
                samples[dst + 3] = 255;
              }
              n++;
            }
          }
        } else {
          for (let n = 0; n < size; n++) {
            const pick = Math.floor(random() * count);
            const py = y0 + Math.floor(pick / cellW);
            const px = x0 + (pick % cellW);
            const idx = (py * srcWidth + px) * 4;
            const dst = n * 4;
            if (transparent) {
              const a = src[idx + 3];
              const alpha = a / 255;
              samples[dst] = f32(src[idx] * alpha);
              samples[dst + 1] = f32(src[idx + 1] * alpha);
              samples[dst + 2] = f32(src[idx + 2] * alpha);
              samples[dst + 3] = a;
            } else {
              samples[dst] = src[idx];
              samples[dst + 1] = src[idx + 1];
              samples[dst + 2] = src[idx + 2];
              samples[dst + 3] = 255;
            }
          }
        }

        center0[0] = samples[0];
        center0[1] = samples[1];
        center0[2] = samples[2];
        center0[3] = samples[3];

        let farthest = 0;
        let maxDistance = -1;
        for (let n = 0; n < size; n++) {
          const base = n * 4;
          const d0 = f32(samples[base] - center0[0]);
          const d1 = f32(samples[base + 1] - center0[1]);
          const d2 = f32(samples[base + 2] - center0[2]);
          let total = f32(f32(f32(d0 * d0) + f32(d1 * d1)) + f32(d2 * d2));
          if (transparent) {
            const d3 = f32(samples[base + 3] - center0[3]);
            total = f32(total + f32(d3 * d3));
          }
          if (total > maxDistance) {
            maxDistance = total;
            farthest = n;
          }
        }

        const fBase = farthest * 4;
        center1[0] = samples[fBase];
        center1[1] = samples[fBase + 1];
        center1[2] = samples[fBase + 2];
        center1[3] = samples[fBase + 3];

        let count1 = 0;
        for (let iter = 0; iter < 6; iter++) {
          sum0.fill(0);
          sum1.fill(0);
          count1 = 0;
          for (let n = 0; n < size; n++) {
            const base = n * 4;
            const s0 = samples[base];
            const s1 = samples[base + 1];
            const s2 = samples[base + 2];
            const s3 = samples[base + 3];

            const a0 = f32(s0 - center0[0]);
            const a1 = f32(s1 - center0[1]);
            const a2 = f32(s2 - center0[2]);
            let dist0 = f32(f32(f32(a0 * a0) + f32(a1 * a1)) + f32(a2 * a2));
            const b0 = f32(s0 - center1[0]);
            const b1 = f32(s1 - center1[1]);
            const b2 = f32(s2 - center1[2]);
            let dist1 = f32(f32(f32(b0 * b0) + f32(b1 * b1)) + f32(b2 * b2));
            if (transparent) {
              const a3 = f32(s3 - center0[3]);
              const b3 = f32(s3 - center1[3]);
              dist0 = f32(dist0 + f32(a3 * a3));
              dist1 = f32(dist1 + f32(b3 * b3));
            }

            if (dist1 < dist0) {
              count1++;
              sum1[0] += s0;
              sum1[1] += s1;
              sum1[2] += s2;
              sum1[3] += s3;
            } else {
              sum0[0] += s0;
              sum0[1] += s1;
              sum0[2] += s2;
              sum0[3] += s3;
            }
          }
          const count0 = size - count1;
          if (count0) {
            center0[0] = sum0[0] / count0;
            center0[1] = sum0[1] / count0;
            center0[2] = sum0[2] / count0;
            center0[3] = sum0[3] / count0;
          }
          if (count1) {
            center1[0] = sum1[0] / count1;
            center1[1] = sum1[1] / count1;
            center1[2] = sum1[2] / count1;
            center1[3] = sum1[3] / count1;
          }
        }

        const winner = count1 >= size - count1 ? center1 : center0;
        result[0] = winner[0];
        result[1] = winner[1];
        result[2] = winner[2];
        result[3] = winner[3];
      }

      const alpha = clamp(roundEven(result[3]), 0, 255);
      out[target + 3] = alpha;
      if (alpha === 0) {
        out[target] = 0;
        out[target + 1] = 0;
        out[target + 2] = 0;
      } else if (transparent) {
        const invAlpha = 255 / result[3];
        out[target] = clamp(roundEven(result[0] * invAlpha), 0, 255);
        out[target + 1] = clamp(roundEven(result[1] * invAlpha), 0, 255);
        out[target + 2] = clamp(roundEven(result[2] * invAlpha), 0, 255);
      } else {
        out[target] = clamp(roundEven(result[0]), 0, 255);
        out[target + 1] = clamp(roundEven(result[1]), 0, 255);
        out[target + 2] = clamp(roundEven(result[2]), 0, 255);
      }
    }

    if (row % 8 === 0 || row === outHeight - 1) progress((row + 1) / outHeight);
  }

  return { width: outWidth, height: outHeight, data: out };
}

/**
 * Run detection, refinement, and sampling against a prepared image handle,
 * reusing cached grayscale, FFT grid estimates, and Sobel projections.
 */
export function processPreparedImage(
  prepared: PreparedPixelImage,
  partial: Partial<PixelOptions> = {},
  onProgress: (stage: ProcessingStage, fraction?: number) => void = () => {},
): PixelResult {
  const started = nowMs();
  const options: PixelOptions = { ...DEFAULT_OPTIONS, ...partial };
  const { image, transparent } = prepared;
  validateInput(image, options);

  let detection: "manual" | "rgb" | "alpha" = options.gridSize ? "manual" : "rgb";
  let grid: [number, number] | null = options.gridSize;
  let useAlphaGradients = false;

  onProgress("detect");
  if (!grid) {
    const cacheKey = `${options.minSize}:${options.peakWidth}`;
    if (prepared.rgbGridCache.has(cacheKey)) {
      grid = prepared.rgbGridCache.get(cacheKey) ?? null;
    } else {
      grid = detectGrid(
        prepared.rgbGray,
        image.width,
        image.height,
        options.minSize,
        options.peakWidth,
        () => getRgbGradients(prepared),
      );
      prepared.rgbGridCache.set(cacheKey, grid);
    }

    if (!grid && transparent) {
      if (prepared.alphaGridCache.has(cacheKey)) {
        grid = prepared.alphaGridCache.get(cacheKey) ?? null;
      } else {
        grid = detectGrid(
          getAlphaGray(prepared),
          image.width,
          image.height,
          options.minSize,
          options.peakWidth,
          () => getAlphaGradients(prepared),
        );
        prepared.alphaGridCache.set(cacheKey, grid);
      }
      if (grid) {
        detection = "alpha";
        useAlphaGradients = true;
      }
    }
  }

  if (!grid) {
    return { status: "no-grid", elapsedMs: Math.round((nowMs() - started) * 10) / 10 };
  }

  onProgress("refine");
  const gradients = useAlphaGradients
    ? getAlphaGradients(prepared)
    : getRgbGradients(prepared);
  const manual = options.gridSize !== null;
  const xs = refineAxis(image.width, grid[0], gradients.x, options.refineIntensity, manual);
  const ys = refineAxis(image.height, grid[1], gradients.y, options.refineIntensity, manual);

  onProgress("sample", 0);
  let result = sampleGrid(
    image,
    xs,
    ys,
    options.sampleMethod,
    transparent,
    (fraction) => onProgress("sample", fraction),
  );

  if (options.fixSquare && !manual && Math.abs(result.width - result.height) === 1) {
    const { width: w, height: h, data } = result;
    const width = w > h ? (w % 2 ? w - 1 : w) : h % 2 ? w : w + 1;
    const height = w > h ? (w % 2 ? h : h + 1) : h % 2 ? h - 1 : h;
    const adjusted = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
      const sy = height > h ? Math.max(0, y - 1) : y;
      for (let x = 0; x < width; x++) {
        const sx = width > w ? Math.max(0, x - 1) : x;
        const s = (sy * w + sx) * 4;
        const d = (y * width + x) * 4;
        adjusted[d] = data[s];
        adjusted[d + 1] = data[s + 1];
        adjusted[d + 2] = data[s + 2];
        adjusted[d + 3] = data[s + 3];
      }
    }
    result = { width, height, data: adjusted };
  }

  if (
    !manual &&
    Math.min(image.width / result.width, image.height / result.height) <
      options.minSize
  ) {
    return { status: "no-grid", elapsedMs: Math.round((nowMs() - started) * 10) / 10 };
  }

  return {
    status: "success",
    image: result,
    detection,
    detectedGrid: [grid[0], grid[1]],
    gridLines: { x: xs, y: ys },
    elapsedMs: Math.round((nowMs() - started) * 10) / 10,
  };
}

/** Convenience one-shot API wrapping `preparePixelImage` + `processPreparedImage`. */
export function getPerfectPixel(
  image: PixelImage,
  partial: Partial<PixelOptions> = {},
  onProgress: (stage: ProcessingStage, fraction?: number) => void = () => {},
): PixelResult {
  const options: PixelOptions = { ...DEFAULT_OPTIONS, ...partial };
  validateInput(image, options);
  onProgress("prepare");
  const prepared = preparePixelImage(image);
  return processPreparedImage(prepared, options, onProgress);
}
