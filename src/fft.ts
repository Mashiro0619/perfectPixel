import FFT from "fft.js";

const f32 = Math.fround;

type ComplexTransform = (input: Float64Array, output: Float64Array) => void;

function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

/**
 * Exact-length complex 1D DFT.
 * - Uses native radix-4/2 FFT from fft.js for powers of two.
 * - Uses Bluestein chirp-z convolution for arbitrary lengths so the source
 *   image itself is never zero-padded (preserving exact frequency bins).
 */
export function createTransform(length: number): ComplexTransform {
  if (length === 1) {
    return (input, output) => {
      output[0] = input[0];
      output[1] = input[1];
    };
  }
  if (isPowerOfTwo(length)) {
    const fft = new FFT(length);
    return (input, output) => fft.transform(output, input);
  }

  let padded = 1;
  while (padded < 2 * length - 1) padded *= 2;
  const fft = new FFT(padded);
  const chirp = new Float64Array(length * 2);
  const kernel = new Float64Array(padded * 2);
  const spectrum = new Float64Array(padded * 2);
  const work = new Float64Array(padded * 2);
  const product = new Float64Array(padded * 2);
  const convolution = new Float64Array(padded * 2);

  for (let n = 0; n < length; n++) {
    const angle = (Math.PI * ((n * n) % (2 * length))) / length;
    const re = Math.cos(angle);
    const im = Math.sin(angle);
    chirp[2 * n] = re;
    chirp[2 * n + 1] = -im;
    kernel[2 * n] = re;
    kernel[2 * n + 1] = im;
    if (n) {
      kernel[2 * (padded - n)] = re;
      kernel[2 * (padded - n) + 1] = im;
    }
  }
  fft.transform(spectrum, kernel);

  return (input, output) => {
    work.fill(0);
    for (let n = 0; n < length; n++) {
      const i = n * 2;
      const a = input[i];
      const b = input[i + 1];
      const c = chirp[i];
      const d = chirp[i + 1];
      work[i] = a * c - b * d;
      work[i + 1] = a * d + b * c;
    }
    fft.transform(product, work);
    for (let n = 0; n < padded; n++) {
      const i = n * 2;
      const a = product[i];
      const b = product[i + 1];
      const c = spectrum[i];
      const d = spectrum[i + 1];
      product[i] = a * c - b * d;
      product[i + 1] = a * d + b * c;
    }
    fft.inverseTransform(convolution, product);
    for (let n = 0; n < length; n++) {
      const i = n * 2;
      const a = convolution[i];
      const b = convolution[i + 1];
      const c = chirp[i];
      const d = chirp[i + 1];
      output[i] = a * c - b * d;
      output[i + 1] = a * d + b * c;
    }
  };
}

/**
 * Compute centered, inverted log-magnitude 2D FFT matching NumPy float32
 * semantics (`1 - log1p(|fftshift(fft2(gray))|)`).
 * Power-of-two rows leverage `realTransform` + `completeSpectrum` (~2x faster
 * on common 512 / 1024 / 2048 AI pixel art inputs).
 */
export function fftMagnitude(
  gray: Float32Array,
  width: number,
  height: number,
): Float32Array {
  const spectrum = new Float32Array(width * height * 2);
  const rowOut = new Float64Array(width * 2);

  if (width >= 2 && isPowerOfTwo(width)) {
    const rowFft = new FFT(width);
    const realRow = new Float64Array(width);
    for (let y = 0; y < height; y++) {
      const rowStart = y * width;
      for (let x = 0; x < width; x++) realRow[x] = gray[rowStart + x];
      rowFft.realTransform(rowOut, realRow);
      rowFft.completeSpectrum(rowOut);
      spectrum.set(rowOut, rowStart * 2);
    }
  } else {
    const rowTransform = createTransform(width);
    const rowIn = new Float64Array(width * 2);
    for (let y = 0; y < height; y++) {
      const rowStart = y * width;
      for (let x = 0; x < width; x++) {
        rowIn[x * 2] = gray[rowStart + x];
        rowIn[x * 2 + 1] = 0;
      }
      rowTransform(rowIn, rowOut);
      spectrum.set(rowOut, rowStart * 2);
    }
  }

  const colTransform = createTransform(height);
  const colIn = new Float64Array(height * 2);
  const colOut = new Float64Array(height * 2);
  const rowStride = width * 2;

  for (let x = 0; x < width; x++) {
    const colOffset = x * 2;
    for (let y = 0; y < height; y++) {
      const idx = y * rowStride + colOffset;
      colIn[y * 2] = spectrum[idx];
      colIn[y * 2 + 1] = spectrum[idx + 1];
    }
    colTransform(colIn, colOut);
    for (let y = 0; y < height; y++) {
      const idx = y * rowStride + colOffset;
      spectrum[idx] = colOut[y * 2];
      spectrum[idx + 1] = colOut[y * 2 + 1];
    }
  }

  const magnitude = new Float32Array(width * height);
  const shiftY = Math.ceil(height / 2);
  const shiftX = Math.ceil(width / 2);

  for (let y = 0; y < height; y++) {
    const srcRowOffset = ((y + shiftY) % height) * width;
    const dstRowOffset = y * width;
    for (let x = 0; x < width; x++) {
      const src = (srcRowOffset + ((x + shiftX) % width)) * 2;
      magnitude[dstRowOffset + x] = f32(
        1 - f32(Math.log1p(f32(Math.hypot(spectrum[src], spectrum[src + 1])))),
      );
    }
  }
  return magnitude;
}
