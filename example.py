"""Run the example without a GUI, or opt into plots with --show/--debug."""

import argparse
from pathlib import Path

import cv2
from perfect_pixel import get_perfect_pixel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "images" / "girl.jpg")
    parser.add_argument("--output", type=Path, default=Path("output.png"))
    parser.add_argument("--show", action="store_true", help="Show input/output (requires the debug extra)")
    parser.add_argument("--debug", action="store_true", help="Show the detected grid (requires the debug extra)")
    args = parser.parse_args()

    bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    w, h, out = get_perfect_pixel(rgb, sample_method="center", refine_intensity=0.3, debug=args.debug)
    if w is None or h is None:
        raise SystemExit("Failed to estimate a pixel grid; try a manual grid_size.")

    if args.show:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            raise ImportError('Install "perfect-pixel[debug]" to use --show.') from exc
        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(rgb)
        axes[0].set_title("Input")
        axes[1].imshow(out)
        axes[1].set_title(f"Pixel-perfect ({w} x {h})")
        for axis in axes:
            axis.axis("off")
        plt.show()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    scaled_path = args.output.with_name(args.output.stem + "_8x" + args.output.suffix)
    out_8x = cv2.resize(out_bgr, (w * 8, h * 8), interpolation=cv2.INTER_NEAREST)
    for path, data in ((args.output, out_bgr), (scaled_path, out_8x)):
        if not cv2.imwrite(str(path), data):
            raise OSError(f"Could not write image: {path}")
    print(f"Saved {w} x {h} result to {args.output} and {scaled_path}")


if __name__ == "__main__":
    main()
