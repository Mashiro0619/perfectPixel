# Perfect Pixel (Web & TypeScript Engine)

> Fast, local-first TypeScript engine for snapping blurry AI-generated pixel art back to crisp, true pixel grids — built specifically for web applications ([www.mashiro.tech/tools/pixel/](https://www.mashiro.tech/tools/pixel/)) and Node.js.

Refactored and optimized from [theamusing/perfectPixel](https://github.com/theamusing/perfectPixel) (MIT), replacing the Python / OpenCV / ComfyUI stack with a standalone, zero-WASM, zero-backend TypeScript library powered by [`fft.js`](https://github.com/indutny/fft.js).

## Why This Version

- **100% browser-local & CSP-safe**: No server uploads, no Python/OpenCV runtime, no WASM blob, and zero `<img src="blob:">` dependencies (`img-src 'self' data:` compatible).
- **Exact grid detection**:
  - 2D FFT magnitude projection with `realTransform` acceleration on power-of-two widths (512 / 1024 / 2048px) and exact Bluestein DFT on arbitrary dimensions.
  - Prominence-filtered FFT peak detection without the upstream 20px hard cell-size cutoff (handles high-res 32px/64px sprites cleanly).
  - Single-pass 3×3 Sobel gradient fallback with flat-plateau peak detection (`_gradient_peaks`) and RGBA alpha-only fallback for transparent silhouettes.
  - Strictly monotonic, bounded `[0, W]` edge snapping that preserves exact manual `gridSize` counts.
- **Instant interactive re-sampling (`preparePixelImage`)**:
  - Caches grayscale conversion, FFT grid estimates, and Sobel projections per image.
  - On 1024×1024 images: ~100 ms initial FFT detection, **5–17 ms** cached re-sampling when switching `center` / `median` / `majority` or tuning `refineIntensity`.
- **Web workbench batteries included**:
  - Exposes detected `gridLines: { x, y }` for live canvas/SVG grid overlays.
  - Ready-to-use cached Web Worker handler (`@mashirotech/perfect-pixel/worker`).
  - Strict PNG / WebP / JPEG header inspector (`@mashirotech/perfect-pixel/image`) rejecting animated, malformed, or decompression-bomb inputs.
  - Lossless streaming RGBA PNG encoder (`@mashirotech/perfect-pixel/png`) with `AbortSignal` and 1×/2×/4×/8× scaling.

---

## Installation

Copy `packages/perfect-pixel` into a workspace or install directly as an ESM TypeScript/JavaScript package:

```bash
npm install
npm run check   # typecheck + build dist/ + unit & 9-image regression tests
```

## Quick Start

### 1. Prepared Session API (Recommended for Interactive UIs)

```ts
import {
  preparePixelImage,
  processPreparedImage,
} from "@mashirotech/perfect-pixel";

// Run once per uploaded RGBA image
const prepared = preparePixelImage({ width, height, data });

// Initial auto-detection
const initial = processPreparedImage(prepared, {
  sampleMethod: "center",
});

// Subsequent tweaks skip grayscale, FFT & Sobel work
const refined = processPreparedImage(prepared, {
  sampleMethod: "median",
  refineIntensity: 0.3,
});

if (refined.status === "success") {
  console.log(refined.image.width, refined.image.height, refined.elapsedMs);
  console.log(refined.gridLines); // { x: number[], y: number[] }
}
```

### 2. Web Worker Integration (2 Lines)

```ts
/// <reference lib="webworker" />
import { attachPerfectPixelWorker } from "@mashirotech/perfect-pixel/worker";

attachPerfectPixelWorker(self as unknown as DedicatedWorkerGlobalScope);
```

Send `{ type: "process", id, imageKey, image, options }` on the first upload (transferring `[image.data.buffer]`), and omit `image` on subsequent option tweaks for the same `imageKey`. Send `{ type: "clear" }` on reset/unmount to release cached buffers.

### 3. Browser File Decoding & Lossless PNG Export

```ts
import { encodePng, exportName, loadImageFile } from "@mashirotech/perfect-pixel/image";

const loaded = await loadImageFile(file); // Validates PNG/WebP/JPEG headers; no blob: <img> URLs
// ...process via Worker...
const pngBlob = await encodePng(result.image, 4, abortController.signal);
const filename = exportName(loaded.name, 4);
```

---

## Subpath Exports

| Import Path | Key Exports |
| --- | --- |
| `@mashirotech/perfect-pixel` | `preparePixelImage`, `processPreparedImage`, `getPerfectPixel`, `validateImage`, `validateInput`, `DEFAULT_OPTIONS`, `INPUT_LIMITS`, `EXPORT_PIXEL_LIMIT` |
| `@mashirotech/perfect-pixel/types` | `PixelImage`, `PixelOptions`, `PixelResult`, `PreparedPixelImage`, `GridLines`, `SampleMethod`, `WorkerRequest`, `WorkerResponse` |
| `@mashirotech/perfect-pixel/worker` | `attachPerfectPixelWorker` |
| `@mashirotech/perfect-pixel/image` | `inspectImage`, `loadImageFile`, `canExport`, `encodePng`, `exportName`, `PixelImageError` |
| `@mashirotech/perfect-pixel/png` | `encodeRgbaPng` |
| `@mashirotech/perfect-pixel/sample` | `createSampleBlurryPixelArt` |

---

## Verified Benchmark (9 Sample Images in `images/`)

All 9 repository sample images match the verified `6e2de13` Python reference dimensions 9/9 with zero out-of-bounds grid coordinates:

| Image | Input Size | Output Grid | Cold (FFT + Sample) | Warm Cached Re-sample |
| --- | --- | --- | ---: | ---: |
| `avatar.png` | 1024×1024 | `48×48` | ~105 ms | ~15 ms |
| `car.png` | 1024×1024 | `82×84` | ~103 ms | ~17 ms |
| `comfyui.png` | 2016×839 | `115×51` | ~298 ms | ~36 ms |
| `girl.jpg` | 1024×1024 | `24×24` | ~98 ms | ~7 ms |
| `pig.png` | 488×502 | `no-grid` | ~40 ms | 0 ms |
| `rika.png` | 1024×1024 | `56×56` | ~113 ms | ~6 ms |
| `robot.jpeg` | 832×1216 | `52×76` | ~233 ms | ~5 ms |
| `shanxi.jpg` | 911×877 | `73×69` | ~165 ms | ~5 ms |
| `skull.png` | 1024×1024 | `125×123` | ~96 ms | ~8 ms |

---

## License & Attribution

MIT License — see [LICENSE](./LICENSE) and [NOTICE.md](./NOTICE.md).

- Original algorithm: **theAmusing and contributors** ([theamusing/perfectPixel](https://github.com/theamusing/perfectPixel), MIT)
- FFT implementation: **Fedor Indutny** ([fft.js](https://github.com/indutny/fft.js) 4.0.4, MIT)
- Web TypeScript refactor & optimizations: **Mashiro0619**
