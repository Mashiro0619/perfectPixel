import type { PixelImage } from "./types.ts";

const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, n) => {
  let value = n;
  for (let bit = 0; bit < 8; bit++) {
    value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  }
  return value >>> 0;
});

function pngChunk(type: string, payload = new Uint8Array(0)): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(payload.length + 12);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, payload.length);
  for (let i = 0; i < 4; i++) bytes[4 + i] = type.charCodeAt(i);
  bytes.set(payload, 8);
  let crc = 0xffffffff;
  for (let i = 4; i < bytes.length - 4; i++) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 255] ^ (crc >>> 8);
  }
  view.setUint32(bytes.length - 4, (crc ^ 0xffffffff) >>> 0);
  return bytes;
}

const abortError = () => new DOMException("PNG export cancelled", "AbortError");

/**
 * Lossless streaming RGBA PNG encoder using native `CompressionStream("deflate")`.
 * Encodes exact RGBA bytes without premultiplied-alpha canvas round-tripping
 * and supports wide 8x exports beyond browser canvas limits.
 */
export async function encodeRgbaPng(
  image: PixelImage,
  scale = 1,
  signal?: AbortSignal,
): Promise<Blob> {
  if (signal?.aborted) throw abortError();

  const compressor = new CompressionStream("deflate");
  const writer = compressor.writable.getWriter();
  const compressed = new Response(compressor.readable).arrayBuffer().then(
    (buffer) => ({ buffer, error: undefined }),
    (error) => ({ buffer: undefined, error }),
  );
  const cancel = () => {
    void writer.abort(abortError()).catch(() => {});
  };
  signal?.addEventListener("abort", cancel, { once: true });

  try {
    const width = image.width * scale;
    const height = image.height * scale;
    const src = image.data;
    for (let y = 0; y < image.height; y++) {
      if (signal?.aborted) throw abortError();
      const row = new Uint8Array(width * 4 + 1); // Filter 0 (None)
      for (let x = 0; x < image.width; x++) {
        const s = (y * image.width + x) * 4;
        for (let repeat = 0; repeat < scale; repeat++) {
          const d = 1 + (x * scale + repeat) * 4;
          row[d] = src[s];
          row[d + 1] = src[s + 1];
          row[d + 2] = src[s + 2];
          row[d + 3] = src[s + 3];
        }
      }
      for (let repeat = 0; repeat < scale; repeat++) {
        await writer.write(row);
      }
    }
    await writer.close();
    const result = await compressed;
    if (result.error) throw result.error;
    if (signal?.aborted) throw abortError();
    if (!result.buffer) throw new Error("PNG compression failed");

    const ihdr = new Uint8Array(13);
    const view = new DataView(ihdr.buffer);
    view.setUint32(0, width);
    view.setUint32(4, height);
    ihdr[8] = 8; // 8-bit depth
    ihdr[9] = 6; // RGBA color type

    return new Blob(
      [
        new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
        pngChunk("IHDR", ihdr),
        pngChunk("IDAT", new Uint8Array(result.buffer)),
        pngChunk("IEND"),
      ],
      { type: "image/png" },
    );
  } catch (error) {
    await writer.abort(error).catch(() => {});
    await compressed;
    throw error;
  } finally {
    signal?.removeEventListener("abort", cancel);
    writer.releaseLock();
  }
}
