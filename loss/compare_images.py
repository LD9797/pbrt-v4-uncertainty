#!/usr/bin/env python3
"""Compare two rendered images (e.g. baseline pbrt vs. pbrt+NRC) and report
error metrics (MSE, RMSE, MAE, relMSE, PSNR, SSIM).

GUI usage (file pickers via zenity):
    ./compare_images.py

CLI usage (skip the GUI):
    ./compare_images.py reference.exr test.exr

Supports .exr, .png, .pfm, .hdr, .qoi, .jpg and anything imageio can read.
The first image is treated as the reference (ground truth / baseline), the
second as the test image (e.g. NRC prediction).
"""
import subprocess
import sys

import imageio.v3 as iio
import numpy as np
import OpenEXR
from skimage.metrics import structural_similarity

RELMSE_EPS = 1e-2
FILE_FILTER = "Images | *.exr *.png *.pfm *.hdr *.qoi *.jpg *.jpeg *.tga *.bmp"


def pick_file(title: str) -> str:
    try:
        result = subprocess.run(
            ["zenity", "--file-selection", f"--title={title}", f"--file-filter={FILE_FILTER}"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        sys.exit("zenity not found. Install it (e.g. `sudo apt install zenity`) "
                 "or pass two image paths on the command line instead.")
    if result.returncode != 0 or not result.stdout.strip():
        sys.exit("No file selected, aborting.")
    return result.stdout.strip()


def show_result(title: str, text: str, is_error: bool = False) -> None:
    print(text)
    flag = "--error" if is_error else "--info"
    try:
        subprocess.run(["zenity", flag, f"--title={title}", f"--text={text}", "--width=400"])
    except FileNotFoundError:
        pass  # already printed to stdout


def load_image(path: str) -> np.ndarray:
    """Load an image as a linear float32 (H, W, 3) array."""
    if path.lower().endswith(".exr"):
        part = OpenEXR.File(path).parts[0]
        channels = part.channels
        if "RGB" in channels:
            arr = channels["RGB"].pixels
        elif "RGBA" in channels:
            arr = channels["RGBA"].pixels[..., :3]
        elif "Y" in channels:
            arr = np.stack([channels["Y"].pixels] * 3, axis=-1)
        else:
            raise ValueError(f"No RGB/RGBA/Y channel found in {path}")
        return arr.astype(np.float32)

    raw = iio.imread(path)
    if raw.ndim == 2:
        raw = np.stack([raw] * 3, axis=-1)
    raw = raw[..., :3]
    if raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    if raw.dtype == np.uint16:
        return raw.astype(np.float32) / 65535.0
    return raw.astype(np.float32)


def compute_metrics(reference: np.ndarray, test: np.ndarray) -> dict:
    diff = reference - test
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    rel_mse = float(np.mean(diff ** 2 / (reference ** 2 + RELMSE_EPS)))

    data_range = float(max(reference.max(), test.max()) - min(reference.min(), test.min()))
    data_range = data_range if data_range > 0 else 1.0
    psnr = float("inf") if mse == 0 else 20 * np.log10(data_range) - 10 * np.log10(mse)
    ssim = float(structural_similarity(reference, test, channel_axis=2, data_range=data_range))

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "relMSE": rel_mse,
            "PSNR (dB)": psnr, "SSIM": ssim}


def main() -> None:
    if len(sys.argv) == 3:
        ref_path, test_path = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        ref_path = pick_file("Select the reference image (e.g. baseline)")
        test_path = pick_file("Select the test image (e.g. NRC prediction)")
    else:
        sys.exit("Usage: compare_images.py [reference_image test_image]")

    reference = load_image(ref_path)
    test = load_image(test_path)

    if reference.shape != test.shape:
        show_result("Image Comparison Error",
                     f"Resolution/channel mismatch:\n{ref_path}: {reference.shape}\n"
                     f"{test_path}: {test.shape}", is_error=True)
        sys.exit(1)

    metrics = compute_metrics(reference, test)
    lines = [f"Reference: {ref_path}", f"Test:      {test_path}",
             f"Resolution: {reference.shape[1]}x{reference.shape[0]}", ""]
    lines += [f"{name:<10} {value:.6f}" for name, value in metrics.items()]
    show_result("Image Comparison Results", "\n".join(lines))


if __name__ == "__main__":
    main()
