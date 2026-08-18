import OpenEXR
import numpy as np
import sys
from skimage.metrics import structural_similarity


def load_exr(path):
    channels = OpenEXR.File(path).channels()
    return channels["RGB"].pixels.astype(np.float32)


def mse(reference, test):
    if reference.shape != test.shape:
        raise ValueError(
            f"Image dimensions do not match: "
            f"{reference.shape} vs {test.shape}"
        )

    return np.mean((reference - test) ** 2)


def mae(reference, test):
    return np.mean(np.abs(reference - test))


def log_mse(reference, test):
    reference = np.log1p(np.clip(reference, 0, None))
    test = np.log1p(np.clip(test, 0, None))
    return np.mean((reference - test) ** 2)


def psnr(reference, test, mse_value):
    data_range = reference.max() - reference.min()
    return 10 * np.log10((data_range ** 2) / mse_value)


def ssim(reference, test):
    data_range = max(reference.max(), test.max()) - min(reference.min(), test.min())
    return structural_similarity(
        reference, test, data_range=data_range, channel_axis=-1
    )

def ssim_log(reference, test):
    reference_log = np.log1p(np.clip(reference, 0, None))
    test_log = np.log1p(np.clip(test, 0, None))

    max_value = max(reference_log.max(), test_log.max())

    if max_value > 0:
        reference_log /= max_value
        test_log /= max_value

    return structural_similarity(
        reference_log,
        test_log,
        data_range=1.0,
        channel_axis=-1
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("python loss.py reference.exr test.exr")
        sys.exit(1)

    reference_path = sys.argv[1]
    test_path = sys.argv[2]

    reference = load_exr(reference_path)
    test = load_exr(test_path)

    print("-- Loss Metrics --")
    print(f"Reference image: {reference_path}")
    print(f"Test image: {test_path}")
    print(f"Max reference: {reference.max():.6f}")
    print(f"Max prediction: {test.max():.6f}")
    print(f"99th percentile reference: {np.percentile(reference, 99):.6f}")
    print(f"99.9th percentile reference: {np.percentile(reference, 99.9):.6f}")
    print(f"99th percentile prediction: {np.percentile(test, 99):.6f}")
    print(f"99.9th percentile prediction: {np.percentile(test, 99.9):.6f}")
    print(f"MAE: {mae(reference, test):.10f}")
    mse_value = mse(reference, test)
    print(f"MSE: {mse_value:.10f}")
    print(f"Log MSE: {log_mse(reference, test):.10f}")
    print(f"PSNR: {psnr(reference, test, mse_value):.6f} dB")
    print(f"SSIM: {ssim(reference, test):.6f}")
    print("------------------")