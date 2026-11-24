import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";
import sharp from "sharp";

import {
  DEFAULT_OPTIONS,
  getPerfectPixel,
  preparePixelImage,
  processPreparedImage,
} from "../dist/index.js";
import { canExport, exportName, inspectImage, PixelImageError } from "../dist/image.js";
import { encodeRgbaPng } from "../dist/png.js";
import { createSampleBlurryPixelArt } from "../dist/sample.js";
import { attachPerfectPixelWorker } from "../dist/worker.js";

function makeRandomBlocks(gridW, gridH, cellW, cellH = cellW) {
  let state = 0x51a7c3d9;
  const nextByte = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) & 255;
  };
  const expected = new Uint8ClampedArray(gridW * gridH * 4);
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      const idx = (y * gridW + x) * 4;
      expected[idx] = nextByte();
      expected[idx + 1] = nextByte();
      expected[idx + 2] = nextByte();
      expected[idx + 3] = 255;
    }
  }
  const width = gridW * cellW;
  const height = gridH * cellH;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const s = (gy * gridW + gx) * 4;
      for (let dy = 0; dy < cellH; dy++) {
        for (let dx = 0; dx < cellW; dx++) {
          const d = ((gy * cellH + dy) * width + (gx * cellW + dx)) * 4;
          data[d] = expected[s];
          data[d + 1] = expected[s + 1];
          data[d + 2] = expected[s + 2];
          data[d + 3] = expected[s + 3];
        }
      }
    }
  }
  return {
    source: { width, height, data },
    expected: { width: gridW, height: gridH, data: expected },
  };
}

test("exact round-trip across center, median, and majority with cached session parity", () => {
  const { source, expected } = makeRandomBlocks(32, 32, 16);
  const prepared = preparePixelImage(source);

  for (const sampleMethod of ["center", "median", "majority"]) {
    const cached = processPreparedImage(prepared, { sampleMethod });
    const oneShot = getPerfectPixel(source, { sampleMethod });
    assert.equal(cached.status, "success");
    assert.equal(oneShot.status, "success");
    assert.equal(cached.image.width, 32);
    assert.equal(cached.image.height, 32);
    assert.deepEqual(cached.image.data, expected.data);
    assert.deepEqual(cached.image.data, oneShot.image.data);
    assert.equal(cached.gridLines.x[0], 0);
    assert.equal(cached.gridLines.x.at(-1), source.width);
    assert.equal(cached.gridLines.y[0], 0);
    assert.equal(cached.gridLines.y.at(-1), source.height);
  }
});

test("coarse high-resolution grids (>20px/cell) succeed cleanly", () => {
  for (const [grid, cell] of [[16, 32], [32, 32], [8, 64]]) {
    const { source, expected } = makeRandomBlocks(grid, grid, cell);
    const result = getPerfectPixel(source);
    assert.equal(result.status, "success");
    assert.equal(result.image.width, grid);
    assert.equal(result.image.height, grid);
    assert.deepEqual(result.image.data, expected.data);
  }
});

test("built-in blurry sample sprite snaps to exact 24x24", () => {
  const sample = createSampleBlurryPixelArt();
  const result = getPerfectPixel(sample.image);
  assert.equal(result.status, "success");
  assert.deepEqual([result.image.width, result.image.height], sample.trueGrid);
});

test("alpha-only fallback detects transparent white silhouettes and zeroes hidden RGB", () => {
  const grid = 16;
  const cell = 12;
  const width = grid * cell;
  const height = grid * cell;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let gy = 0; gy < grid; gy++) {
    for (let gx = 0; gx < grid; gx++) {
      const opaque = ((gx + gy) & 1) === 0;
      for (let dy = 0; dy < cell; dy++) {
        for (let dx = 0; dx < cell; dx++) {
          const i = ((gy * cell + dy) * width + (gx * cell + dx)) * 4;
          data[i] = 255;
          data[i + 1] = 255;
          data[i + 2] = 255;
          data[i + 3] = opaque ? 255 : 0;
        }
      }
    }
  }
  const result = getPerfectPixel({ width, height, data });
  assert.equal(result.status, "success");
  assert.equal(result.detection, "alpha");
  assert.equal(result.image.width, grid);
  assert.equal(result.image.height, grid);
  assert.deepEqual(Array.from(result.image.data.subarray(4, 8)), [0, 0, 0, 0]);
});

test("manual grids preserve requested dimensions and invalid inputs fail closed", () => {
  const uniform = {
    width: 64,
    height: 64,
    data: new Uint8ClampedArray(64 * 64 * 4).fill(200),
  };
  assert.equal(getPerfectPixel(uniform).status, "no-grid");

  const manual = getPerfectPixel(uniform, {
    gridSize: [7, 8],
    fixSquare: true,
    minSize: 999,
  });
  assert.equal(manual.status, "success");
  assert.equal(manual.detection, "manual");
  assert.equal(manual.image.width, 7);
  assert.equal(manual.image.height, 8);

  assert.throws(() => getPerfectPixel(uniform, { refineIntensity: 0.7 }), /invalid-refinement/);
  assert.throws(() => getPerfectPixel(uniform, { gridSize: [65, 8] }), /invalid-grid/);
});

test("worker helper caches prepared image across subsequent option requests", () => {
  const messages = [];
  const scope = {
    onmessage: null,
    postMessage(msg) {
      messages.push(msg);
    },
  };
  const detach = attachPerfectPixelWorker(scope);
  const sample = createSampleBlurryPixelArt();

  scope.onmessage({
    data: {
      type: "process",
      id: 1,
      imageKey: "k1",
      image: sample.image,
      options: { ...DEFAULT_OPTIONS, sampleMethod: "center" },
    },
  });
  scope.onmessage({
    data: {
      type: "process",
      id: 2,
      imageKey: "k1",
      options: { ...DEFAULT_OPTIONS, sampleMethod: "median" },
    },
  });
  scope.onmessage({ data: { type: "clear" } });
  scope.onmessage({
    data: {
      type: "process",
      id: 3,
      imageKey: "k1",
      options: { ...DEFAULT_OPTIONS },
    },
  });
  detach();

  const r1 = messages.find((m) => m.id === 1 && m.type === "result");
  const r2 = messages.find((m) => m.id === 2 && m.type === "result");
  const e3 = messages.find((m) => m.id === 3 && m.type === "error");
  assert.equal(r1?.result.status, "success");
  assert.equal(r2?.result.status, "success");
  assert.equal(r2?.result.image.width, 24);
  assert.equal(e3?.code, "missing-image");
});

test("PNG encoder and binary header inspector validate formats cleanly", async () => {
  const { expected } = makeRandomBlocks(12, 10, 1);
  const blob = await encodeRgbaPng(expected, 4);
  const bytes = new Uint8Array(await blob.arrayBuffer());
  assert.deepEqual(inspectImage(bytes), { width: 48, height: 40, mime: "image/png" });

  const gif = new TextEncoder().encode("GIF89a\x01\x00\x01\x00\x00\x00\x00;");
  assert.throws(() => inspectImage(gif), (err) => err instanceof PixelImageError && err.code === "unsupported");

  const minimalJpeg = new Uint8Array([
    0xff, 0xd8,
    0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x20, 0x00, 0x30, 0x01, 0x01, 0x11, 0x00,
    0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00,
    0x12, 0x34,
    0xff, 0xd9,
    0x00, 0x00, 0x00, 0x00,
  ]);
  assert.deepEqual(inspectImage(minimalJpeg), { width: 48, height: 32, mime: "image/jpeg" });
  assert.throws(
    () => inspectImage(minimalJpeg.subarray(0, minimalJpeg.length - 6)),
    (err) => err instanceof PixelImageError && err.code === "corrupt",
  );

  assert.equal(canExport({ width: 128, height: 128 }, 8), true);
  assert.equal(exportName("hero.webp", 2), "hero-perfect-pixel-2x.png");
});

test("all 9 real repository benchmark images match verified reference dimensions", async () => {
  const expectedDims = {
    "avatar.png": [48, 48],
    "car.png": [82, 84],
    "comfyui.png": [115, 51],
    "girl.jpg": [24, 24],
    "pig.png": null,
    "rika.png": [56, 56],
    "robot.jpeg": [52, 76],
    "shanxi.jpg": [73, 69],
    "skull.png": [125, 123],
  };

  for (const [name, expected] of Object.entries(expectedDims)) {
    const raw = await readFile(join("images", name));
    const { data, info } = await sharp(raw).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    const img = {
      width: info.width,
      height: info.height,
      data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength),
    };
    const result = getPerfectPixel(img, { sampleMethod: "center" });
    if (expected === null) {
      assert.equal(result.status, "no-grid", name);
    } else {
      assert.equal(result.status, "success", name);
      assert.deepEqual([result.image.width, result.image.height], expected, name);
      assert.equal(result.gridLines.x[0], 0, name);
      assert.equal(result.gridLines.x.at(-1), img.width, name);
      assert.equal(result.gridLines.y[0], 0, name);
      assert.equal(result.gridLines.y.at(-1), img.height, name);
    }
  }
});
