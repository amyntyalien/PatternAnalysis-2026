"""Run MambaIRv2 x4 super-resolution on local image files."""

import argparse
import sys
import zipfile
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
MAMBAIR_ROOT = MODULE_DIR / "reference" / "MambaIR"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upscale local images 4x with the MambaIRv2 Base model."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=MODULE_DIR / "samples" / "inputs",
        help="Folder containing input images (default: samples/inputs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODULE_DIR / "samples" / "results",
        help="Folder for super-resolved images (default: samples/results).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODULE_DIR / "experiments" / "mambairv2_classicSR_Base_x4.pth",
        help="Path to the MambaIRv2 x4 checkpoint.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Optional path to write a ZIP archive of this run's results.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open input/output comparisons after inference.",
    )
    return parser.parse_args()


def run_inference(args):
    if not args.input_dir.is_dir():
        raise FileNotFoundError(
            f"Input folder not found: {args.input_dir}. "
            "Create it and add images, or pass --input-dir."
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Pass its location with --checkpoint."
        )

    image_paths = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files found in {args.input_dir}."
        )

    if not MAMBAIR_ROOT.is_dir():
        raise FileNotFoundError(f"MambaIR source folder not found: {MAMBAIR_ROOT}")
    sys.path.insert(0, str(MAMBAIR_ROOT))

    import cv2
    import numpy as np
    import torch
    from analysis.model_zoo.mambairv2 import buildMambaIRv2Base
    from basicsr.utils import img2tensor, tensor2img
    from tqdm import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError(
            "MambaIRv2 uses CUDA selective-scan operations and requires a "
            "compatible NVIDIA GPU and CUDA-enabled PyTorch environment."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = buildMambaIRv2Base(upscale=4).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or "params" not in checkpoint:
        raise ValueError(
            f"Expected a checkpoint containing a 'params' state dict: "
            f"{args.checkpoint}"
        )
    model.load_state_dict(checkpoint["params"])

    result_paths = []
    for image_path in tqdm(image_paths, desc="Upscaling"):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = image.astype(np.float32) / 255.0
        image_tensor = img2tensor(image, bgr2rgb=True, float32=True)
        image_tensor = image_tensor.unsqueeze(0).cuda()
        with torch.no_grad():
            output = model(image_tensor)

        result = tensor2img([output.detach().cpu()])
        result_path = args.output_dir / image_path.name
        if not cv2.imwrite(str(result_path), result):
            raise OSError(f"Could not write output image: {result_path}")
        result_paths.append(result_path)

    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for result_path in result_paths:
                archive.write(result_path, arcname=result_path.name)

    print(f"Saved {len(result_paths)} result(s) to {args.output_dir}")
    if args.archive:
        print(f"Created archive: {args.archive}")

    if args.show:
        import matplotlib.pyplot as plt

        for image_path, result_path in zip(image_paths, result_paths):
            input_image = cv2.cvtColor(
                cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            output_image = cv2.cvtColor(
                cv2.imread(str(result_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            figure, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(input_image)
            axes[0].set_title("Input")
            axes[1].imshow(output_image)
            axes[1].set_title("4x super-resolution")
            for axis in axes:
                axis.axis("off")
            figure.tight_layout()
        plt.show()


def main():
    args = parse_args()
    try:
        run_inference(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
