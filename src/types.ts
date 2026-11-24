export type SampleMethod = "center" | "median" | "majority";

export type PixelImage = {
  width: number;
  height: number;
  data: Uint8ClampedArray;
};

export type PixelOptions = {
  sampleMethod: SampleMethod;
  gridSize: [number, number] | null;
  minSize: number;
  peakWidth: number;
  refineIntensity: number;
  fixSquare: boolean;
};

export const DEFAULT_OPTIONS: Readonly<PixelOptions> = Object.freeze({
  sampleMethod: "center",
  gridSize: null,
  minSize: 4,
  peakWidth: 6,
  refineIntensity: 0.25,
  fixSquare: true,
});

export const INPUT_LIMITS = Object.freeze({
  bytes: 10 * 1024 * 1024,
  side: 4096,
  pixels: 4_194_304,
});

export const EXPORT_PIXEL_LIMIT = 16_777_216;

export type ProcessingStage = "prepare" | "detect" | "refine" | "sample";

export type GridLines = {
  x: number[];
  y: number[];
};

export type PixelResult =
  | {
      status: "success";
      image: PixelImage;
      detection: "manual" | "rgb" | "alpha";
      detectedGrid: [number, number];
      gridLines: GridLines;
      elapsedMs: number;
    }
  | {
      status: "no-grid";
      elapsedMs: number;
    };

export type GradientProjections = {
  x: Float32Array;
  y: Float32Array;
};

/**
 * Reusable per-image analysis cache so changing sampling method, refinement
 * intensity, fixSquare, or manual grid dimensions avoids repeating grayscale
 * conversion, 2D FFT, and Sobel projections.
 */
export type PreparedPixelImage = {
  readonly image: PixelImage;
  readonly transparent: boolean;
  readonly rgbGray: Float32Array;
  alphaGray: Float32Array | null;
  rgbGradients: GradientProjections | null;
  alphaGradients: GradientProjections | null;
  readonly rgbGridCache: Map<string, [number, number] | null>;
  readonly alphaGridCache: Map<string, [number, number] | null>;
};

export type WorkerRequest =
  | {
      type: "process";
      id: number;
      imageKey: string;
      image?: PixelImage;
      options: PixelOptions;
    }
  | {
      type: "clear";
    };

export type WorkerResponse =
  | { id: number; type: "progress"; stage: ProcessingStage; fraction?: number }
  | { id: number; type: "result"; result: PixelResult }
  | { id: number; type: "error"; code: "invalid-input" | "processing-failed" | "missing-image" };
