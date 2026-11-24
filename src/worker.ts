import { preparePixelImage, processPreparedImage } from "./index.ts";
import type {
  PreparedPixelImage,
  WorkerRequest,
  WorkerResponse,
} from "./types.ts";

export type WorkerMessageScope = {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void;
};

/**
 * Attach a cached Perfect Pixel handler to a Dedicated Web Worker scope.
 * Transfers the pixel buffer once per uploaded `imageKey` and reuses cached
 * grayscale, FFT grid detection, and Sobel projections on subsequent tweaks.
 */
export function attachPerfectPixelWorker(scope: WorkerMessageScope): () => void {
  let cached: { imageKey: string; prepared: PreparedPixelImage } | null = null;

  const onMessage = (event: MessageEvent<WorkerRequest>) => {
    const message = event.data;
    if (message.type === "clear") {
      cached = null;
      return;
    }

    const { id, imageKey, image, options } = message;
    const send = (response: WorkerResponse, transfer: Transferable[] = []) =>
      scope.postMessage(response, transfer);

    try {
      let prepared: PreparedPixelImage;
      if (cached && cached.imageKey === imageKey) {
        prepared = cached.prepared;
      } else if (image) {
        send({ id, type: "progress", stage: "prepare" });
        prepared = preparePixelImage(image);
        cached = { imageKey, prepared };
      } else {
        send({ id, type: "error", code: "missing-image" });
        return;
      }

      const result = processPreparedImage(prepared, options, (stage, fraction) =>
        send({ id, type: "progress", stage, fraction }),
      );
      send(
        { id, type: "result", result },
        result.status === "success" ? [result.image.data.buffer] : [],
      );
    } catch (error) {
      send({
        id,
        type: "error",
        code:
          error instanceof Error && error.message.startsWith("invalid-")
            ? "invalid-input"
            : "processing-failed",
      });
    }
  };

  scope.onmessage = onMessage;
  return () => {
    cached = null;
    if (scope.onmessage === onMessage) scope.onmessage = null;
  };
}
