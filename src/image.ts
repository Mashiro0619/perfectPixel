import { encodeRgbaPng } from "./png.ts";
import { EXPORT_PIXEL_LIMIT, INPUT_LIMITS } from "./types.ts";
import type { PixelImage } from "./types.ts";

export type ImageErrorCode =
  | "file-size"
  | "unsupported"
  | "animated"
  | "dimensions"
  | "corrupt"
  | "decode"
  | "export-size";

export class PixelImageError extends Error {
  readonly code: ImageErrorCode;
  constructor(code: ImageErrorCode) {
    super(code);
    this.name = "PixelImageError";
    this.code = code;
  }
}

function fail(code: ImageErrorCode): never {
  throw new PixelImageError(code);
}

function validateDimensions(
  width: number,
  height: number,
  limits = INPUT_LIMITS,
): void {
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width < 1 ||
    height < 1 ||
    width > limits.side ||
    height > limits.side ||
    width * height > limits.pixels
  ) {
    fail("dimensions");
  }
}

function ascii(bytes: Uint8Array, start: number, length: number): string {
  let out = "";
  for (let i = 0; i < length; i++) out += String.fromCharCode(bytes[start + i]);
  return out;
}

/**
 * Inspect PNG, WebP, or JPEG binary headers before browser decoding so
 * animated, malformed, or decompression-bomb images fail closed immediately.
 */
export function inspectImage(
  data: Uint8Array,
  limits = INPUT_LIMITS,
): { width: number; height: number; mime: "image/png" | "image/webp" | "image/jpeg" } {
  if (data.length < 12) fail("unsupported");
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);

  // GIF header -> animated/unsupported
  if (ascii(data, 0, 6) === "GIF87a" || ascii(data, 0, 6) === "GIF89a") {
    fail("unsupported");
  }

  // PNG
  if (
    data[0] === 0x89 &&
    data[1] === 0x50 &&
    data[2] === 0x4e &&
    data[3] === 0x47 &&
    data[4] === 0x0d &&
    data[5] === 0x0a &&
    data[6] === 0x1a &&
    data[7] === 0x0a
  ) {
    if (data.length < 33 || ascii(data, 12, 4) !== "IHDR") fail("corrupt");
    const width = view.getUint32(16);
    const height = view.getUint32(20);
    validateDimensions(width, height, limits);

    let offset = 8;
    let sawEnd = false;
    while (offset + 12 <= data.length) {
      const length = view.getUint32(offset);
      const type = ascii(data, offset + 4, 4);
      if (offset + 12 + length > data.length) fail("corrupt");
      if (type === "acTL") fail("animated");
      offset += 12 + length;
      if (type === "IEND") {
        sawEnd = true;
        break;
      }
    }
    if (!sawEnd) fail("corrupt");
    return { width, height, mime: "image/png" };
  }

  // WebP
  if (ascii(data, 0, 4) === "RIFF" && ascii(data, 8, 4) === "WEBP") {
    const riffSize = view.getUint32(4, true) + 8;
    if (riffSize > data.length) fail("corrupt");
    let offset = 12;
    let width = 0;
    let height = 0;
    while (offset + 8 <= riffSize) {
      const tag = ascii(data, offset, 4);
      const size = view.getUint32(offset + 4, true);
      const payload = offset + 8;
      if (payload + size > riffSize) fail("corrupt");
      if (tag === "VP8X") {
        if (size < 10) fail("corrupt");
        if ((data[payload] & 0x02) !== 0) fail("animated");
        width = 1 + (data[payload + 4] | (data[payload + 5] << 8) | (data[payload + 6] << 16));
        height = 1 + (data[payload + 7] | (data[payload + 8] << 8) | (data[payload + 9] << 16));
      } else if (tag === "ANIM") {
        fail("animated");
      } else if (tag === "VP8 " && !width) {
        if (size < 10 || data[payload + 3] !== 0x9d || data[payload + 4] !== 0x01 || data[payload + 5] !== 0x2a) {
          fail("corrupt");
        }
        width = view.getUint16(payload + 6, true) & 0x3fff;
        height = view.getUint16(payload + 8, true) & 0x3fff;
      } else if (tag === "VP8L" && !width) {
        if (size < 5 || data[payload] !== 0x2f) fail("corrupt");
        const bits = view.getUint32(payload + 1, true);
        width = (bits & 0x3fff) + 1;
        height = ((bits >>> 14) & 0x3fff) + 1;
      }
      offset = payload + size + (size & 1);
    }
    validateDimensions(width, height, limits);
    return { width, height, mime: "image/webp" };
  }

  // JPEG
  if (data[0] === 0xff && data[1] === 0xd8) {
    let offset = 2;
    let width = 0;
    let height = 0;
    let sawScan = false;
    while (offset < data.length) {
      while (offset < data.length && data[offset] !== 0xff) offset++;
      while (offset < data.length && data[offset] === 0xff) offset++;
      if (offset >= data.length) fail("corrupt");
      const marker = data[offset++];
      if (marker === 0xd9) break;
      if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
      if (offset + 2 > data.length) fail("corrupt");
      const length = view.getUint16(offset);
      if (length < 2 || length > data.length - offset) fail("corrupt");
      if (
        (marker >= 0xc0 && marker <= 0xc3) ||
        (marker >= 0xc5 && marker <= 0xc7) ||
        (marker >= 0xc9 && marker <= 0xcb) ||
        (marker >= 0xcd && marker <= 0xcf)
      ) {
        if (length < 8) fail("corrupt");
        height = view.getUint16(offset + 3);
        width = view.getUint16(offset + 5);
        validateDimensions(width, height, limits);
      }
      if (marker === 0xda) {
        sawScan = true;
        offset += length;
        break;
      }
      offset += length;
    }
    let sawEoi = false;
    if (sawScan && offset < data.length) {
      for (let i = data.length - 2; i >= offset; i--) {
        if (data[i] === 0xff && data[i + 1] === 0xd9) {
          sawEoi = true;
          break;
        }
      }
    }
    if (!sawScan || !sawEoi || !width) {
      fail("corrupt");
    }
    validateDimensions(width, height, limits);
    return { width, height, mime: "image/jpeg" };
  }

  fail("unsupported");
}

function readBlobAsDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new PixelImageError("decode"));
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.readAsDataURL(blob);
  });
}

export type LoadedPixelFile = {
  image: PixelImage;
  name: string;
  byteSize: number;
};

/**
 * Decode a local File into RGBA pixels strictly without `blob:` image URLs
 * (compatible with strict `img-src 'self' data:` Content-Security-Policy).
 */
export async function loadImageFile(file: File): Promise<LoadedPixelFile> {
  if (file.size > INPUT_LIMITS.bytes) fail("file-size");
  const buffer = await file.arrayBuffer();
  const header = inspectImage(new Uint8Array(buffer));
  const blob = new Blob([buffer], { type: header.mime });

  let bitmap: ImageBitmap | undefined;
  let canvas: HTMLCanvasElement | undefined;
  try {
    let drawable: CanvasImageSource;
    let width: number;
    let height: number;

    if (typeof createImageBitmap === "function") {
      bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
      drawable = bitmap;
      width = bitmap.width;
      height = bitmap.height;
    } else {
      const dataUrl = await readBlobAsDataUrl(blob);
      const image = new Image();
      image.src = dataUrl;
      await image.decode();
      drawable = image;
      width = image.naturalWidth;
      height = image.naturalHeight;
    }

    validateDimensions(width, height);
    if (width * height !== header.width * header.height) fail("decode");

    canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) fail("decode");
    ctx.drawImage(drawable, 0, 0);
    const data = ctx.getImageData(0, 0, width, height).data;
    return {
      image: { width, height, data },
      name: file.name,
      byteSize: file.size,
    };
  } catch (error) {
    if (error instanceof PixelImageError) throw error;
    throw new PixelImageError("decode");
  } finally {
    if (canvas) {
      canvas.width = 0;
      canvas.height = 0;
    }
    bitmap?.close();
  }
}

export function canExport(
  image: Pick<PixelImage, "width" | "height">,
  scale: number,
): boolean {
  return (
    Number.isSafeInteger(image.width) &&
    image.width > 0 &&
    Number.isSafeInteger(image.height) &&
    image.height > 0 &&
    [1, 2, 4, 8].includes(scale) &&
    image.width * image.height * scale * scale <= EXPORT_PIXEL_LIMIT
  );
}

export async function encodePng(
  image: PixelImage,
  scale = 1,
  signal?: AbortSignal,
): Promise<Blob> {
  if (!canExport(image, scale)) fail("export-size");
  if (
    !(image.data instanceof Uint8ClampedArray) ||
    image.data.length !== image.width * image.height * 4
  ) {
    fail("decode");
  }
  if (signal?.aborted) {
    throw new DOMException("PNG export cancelled", "AbortError");
  }
  if (typeof CompressionStream === "function") {
    return encodeRgbaPng(image, scale, signal);
  }

  const canvas = document.createElement("canvas");
  const source = document.createElement("canvas");
  try {
    source.width = image.width;
    source.height = image.height;
    const sctx = source.getContext("2d");
    if (!sctx) fail("decode");
    const pixels = sctx.createImageData(image.width, image.height);
    pixels.data.set(image.data);
    sctx.putImageData(pixels, 0, 0);

    canvas.width = image.width * scale;
    canvas.height = image.height * scale;
    const octx = canvas.getContext("2d");
    if (!octx) fail("decode");
    octx.imageSmoothingEnabled = false;
    octx.drawImage(source, 0, 0, canvas.width, canvas.height);
    return await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new PixelImageError("decode"))),
        "image/png",
      ),
    );
  } finally {
    canvas.width = 0;
    canvas.height = 0;
    source.width = 0;
    source.height = 0;
  }
}

export function exportName(name: string, scale: number): string {
  const base =
    name
      .replace(/\.[^.]+$/u, "")
      .replace(/[<>:"/\\|?*\x00-\x1f]/gu, "_")
      .slice(0, 100) || "image";
  return `${base}-perfect-pixel${scale === 1 ? "" : `-${scale}x`}.png`;
}
