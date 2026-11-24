import type { PixelImage } from "./types.ts";

/**
 * 24x24 pixel-art sakura potion sprite rendered onto a blurry, subtly
 * AI-artifacted 384x384 image (16px/cell) so visitors can immediately try
 * automatic grid snapping & comparison without uploading a file.
 */
const SPRITE_ROWS = [
  "........................",
  ".........OOOOOO.........",
  "........OCCCCCCOO.......",
  "........OCCCCCCOO.......",
  ".........OOOOOO.........",
  "........OggggggO........",
  "........OgWWWWgO........",
  ".........OggggO.........",
  "........OOggggOO........",
  "......OOggWWWWggOO......",
  ".....OgWPPPPPPPPgWO.....",
  "....OgWPPppPPppPPgWO....",
  "....OgWPPpWWWWpPPgWO....",
  "...OgWWPPPWWWWPPPWWgO...",
  "...OgWWPPPPPPPPPPWWgO...",
  "...OgWWPPddPPddPPWWgO...",
  "...OgWPPPddddddPPPWgO...",
  "....OgPPddddddddPPgO....",
  "....OggPddddddddPggO....",
  ".....OOggddddddggOO.....",
  ".......OOggggggOO.......",
  ".........OOOOOO.........",
  "........................",
  "........................",
];

const PALETTE: Record<string, [number, number, number, number]> = {
  ".": [244, 244, 248, 255],
  O: [28, 25, 36, 255],
  C: [186, 134, 92, 255],
  g: [198, 212, 226, 255],
  W: [255, 255, 255, 255],
  P: [238, 126, 166, 255],
  p: [252, 182, 208, 255],
  d: [194, 72, 122, 255],
};

function clampByte(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
}

/**
 * Synthesize a 384x384 "AI-blurry pseudo-pixel" image from the 24x24 sprite
 * using bilinear resampling + slight ringing/noise (zero DOM dependency).
 */
export function createSampleBlurryPixelArt(): {
  image: PixelImage;
  trueGrid: [number, number];
} {
  const gw = SPRITE_ROWS[0].length;
  const gh = SPRITE_ROWS.length;
  const cell = 16;
  const width = gw * cell;
  const height = gh * cell;

  // 1. Crisp upscaled image
  const crisp = new Uint8ClampedArray(width * height * 4);
  for (let gy = 0; gy < gh; gy++) {
    const row = SPRITE_ROWS[gy];
    for (let gx = 0; gx < gw; gx++) {
      const rgba = PALETTE[row[gx]] ?? PALETTE["."];
      for (let dy = 0; dy < cell; dy++) {
        const y = gy * cell + dy;
        for (let dx = 0; dx < cell; dx++) {
          const x = gx * cell + dx;
          const i = (y * width + x) * 4;
          crisp[i] = rgba[0];
          crisp[i + 1] = rgba[1];
          crisp[i + 2] = rgba[2];
          crisp[i + 3] = rgba[3];
        }
      }
    }
  }

  // 2. Separable 5-tap blur + subtle deterministic AI color bleed near edges
  const temp = new Float32Array(width * height * 4);
  const kernel = [0.1, 0.2, 0.4, 0.2, 0.1];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let k = -2; k <= 2; k++) {
        const sx = Math.max(0, Math.min(width - 1, x + k));
        const idx = (y * width + sx) * 4;
        const w = kernel[k + 2];
        r += crisp[idx] * w;
        g += crisp[idx + 1] * w;
        b += crisp[idx + 2] * w;
      }
      const d = (y * width + x) * 4;
      temp[d] = r;
      temp[d + 1] = g;
      temp[d + 2] = b;
      temp[d + 3] = 255;
    }
  }

  const out = new Uint8ClampedArray(width * height * 4);
  let noiseState = 0x12345678;
  const nextNoise = () => {
    noiseState ^= noiseState << 13;
    noiseState ^= noiseState >>> 17;
    noiseState ^= noiseState << 5;
    return ((noiseState >>> 0) / 4294967296 - 0.5) * 10;
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let k = -2; k <= 2; k++) {
        const sy = Math.max(0, Math.min(height - 1, y + k));
        const idx = (sy * width + x) * 4;
        const w = kernel[k + 2];
        r += temp[idx] * w;
        g += temp[idx + 1] * w;
        b += temp[idx + 2] * w;
      }
      const n = nextNoise();
      const d = (y * width + x) * 4;
      out[d] = clampByte(r + n);
      out[d + 1] = clampByte(g + n);
      out[d + 2] = clampByte(b + n);
      out[d + 3] = 255;
    }
  }

  return {
    image: { width, height, data: out },
    trueGrid: [gw, gh],
  };
}
