import numpy as np
import os
import csv
from PIL import Image


def load_image(filepath):
    """
    Load a PNG image and return as a numpy array of shape (H, W, 4).
    Pixel values are in range [0, 255].
    """
    img = Image.open(filepath).convert('RGBA')
    return np.array(img, dtype=np.float64)


def extract_histogram_features(image, num_bins=64):
    """
    Extract color histogram features from an RGBA image.
    For each of the 4 channels (R, G, B, A), compute a histogram
    with num_bins bins over pixel values [0, 255].
    Returns a feature vector of length 4 * num_bins.
    """
    features = []
    for channel in range(4):  # R, G, B, A channels
        channel_pixels = image[:, :, channel].ravel()
        hist, _ = np.histogram(channel_pixels, bins=num_bins, range=(0, 256))
        # normalize by total number of pixels so values are in [0, 1]
        hist = hist / hist.sum()
        features.append(hist)
    return np.concatenate(features)


def load_dataset(image_dir, csv_path, num_bins=64):
    """
    Load dataset: read CSV for filenames/labels, then extract histogram
    features from each image.

    Returns:
        features  : numpy array of shape (num_samples, 4 * num_bins)
        labels    : numpy array of shape (num_samples,) with values 1-5,
                    or None if csv has no 'label' column (test set)
        filenames : list of image filenames (strings)
    """
    filenames = []
    labels = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filenames.append(row['id'])
            if 'label' in row:
                labels.append(int(row['label']))

    num_samples = len(filenames)
    num_features = 4 * num_bins
    features = np.zeros((num_samples, num_features), dtype=np.float64)

    print(f'  Extracting features from {num_samples} images...')
    for i, fname in enumerate(filenames):
        if (i + 1) % 1000 == 0:
            print(f'    {i + 1}/{num_samples}')
        img_path = os.path.join(image_dir, fname)
        image = load_image(img_path)
        features[i] = extract_histogram_features(image, num_bins)

    labels_arr = np.array(labels, dtype=np.int32) if labels else None
    return features, labels_arr, filenames


# ============================================================
# GRAYSCALE PIXEL FEATURES  (v3 / v4 experiments)
# ============================================================

def extract_grayscale_features(image):
    """
    Extract raw grayscale pixel features from an RGBA image.
    Converts RGBA to grayscale using standard luminance weights:
        gray = 0.299 * R + 0.587 * G + 0.114 * B
    Alpha channel is ignored.
    Pixel values are normalized to [0, 1].
    Returns a feature vector of length H * W (55 * 128 = 7040 for this dataset).
    """
    # image shape: (H, W, 4) — use only R, G, B channels
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    gray = gray / 255.0       # normalize to [0, 1]
    return gray.ravel()       # flatten to 1D vector of length H*W


def load_dataset_pixels(image_dir, csv_path):
    """
    Load dataset: read CSV for filenames/labels, then extract raw grayscale
    pixel features from each image.

    Returns:
        features  : numpy array of shape (num_samples, H * W)
        labels    : numpy array of shape (num_samples,) with values 1-5,
                    or None if csv has no 'label' column (test set)
        filenames : list of image filenames (strings)
    """
    filenames = []
    labels = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filenames.append(row['id'])
            if 'label' in row:
                labels.append(int(row['label']))

    num_samples = len(filenames)

    # determine feature size from the first image
    first_img = load_image(os.path.join(image_dir, filenames[0]))
    num_features = first_img.shape[0] * first_img.shape[1]   # H * W
    features = np.zeros((num_samples, num_features), dtype=np.float64)

    print(f'  Extracting grayscale pixel features from {num_samples} images...')
    for i, fname in enumerate(filenames):
        if (i + 1) % 1000 == 0:
            print(f'    {i + 1}/{num_samples}')
        img_path = os.path.join(image_dir, fname)
        image = load_image(img_path)
        features[i] = extract_grayscale_features(image)

    labels_arr = np.array(labels, dtype=np.int32) if labels else None
    return features, labels_arr, filenames


# ============================================================
# COMPREHENSIVE FEATURES  (v9+ experiments)
# ============================================================
#
# Combines 6 feature groups into a single 1366-dim vector:
#   1. Color histograms (256)  — color distribution baseline
#   2. HOG features (864)      — gradient orientation per 8×8 cell,
#                                captures vertical signal line edges
#   3. Channel percentiles (27) — per-channel intensity statistics
#   4. Local variance (8)      — texture roughness features
#   5. Connected components (28) — blob count at 7 thresholds
#   6. Row/Col projections (183) — spatial energy distribution
#
# Gaussian denoising (sigma=1.5) is applied before gradient
# and spatial features to reduce pixel-level noise.
# ============================================================

from scipy.ndimage import label, uniform_filter, gaussian_filter


def extract_comprehensive_features(image, num_bins=64):
    """
    Extract a comprehensive feature vector from an RGBA image.

    Combines color histograms, HOG (gradient orientation histograms),
    per-channel statistics, texture features, connected component
    counts at multiple thresholds, and row/column energy projections.

    Args:
        image    : numpy array of shape (H, W, 4), pixel values [0, 255]
        num_bins : number of bins for color histograms (per channel)

    Returns:
        1D numpy array of length 1366 (for 128×55 images with 64 bins)
    """
    features = []
    H, W, _ = image.shape

    # --- Grayscale with Gaussian denoise ---
    gray_raw = (0.299 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2]) / 255.0
    gray = gaussian_filter(gray_raw, sigma=1.5)

    # ---- 1. Color histograms (4 × num_bins = 256 features) ----
    # same as extract_histogram_features — proven baseline
    for ch in range(4):
        hist, _ = np.histogram(image[:, :, ch], bins=num_bins, range=(0, 256))
        hist = hist.astype(np.float64)
        hist /= hist.sum()
        features.extend(hist.tolist())

    # ---- 2. HOG — Histogram of Oriented Gradients (864 features) ----
    # Compute image gradients using central differences
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]   # horizontal gradient
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]    # vertical gradient
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx) * 180.0 / np.pi % 180  # [0, 180)

    # Divide image into 8×8 cells, 9 orientation bins per cell
    cell_size = 8
    hog_bins = 9
    n_cells_y = H // cell_size   # 128 // 8 = 16
    n_cells_x = W // cell_size   # 55 // 8 = 6

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0 = cy * cell_size
            y1 = (cy + 1) * cell_size
            x0 = cx * cell_size
            x1 = (cx + 1) * cell_size
            cell_mag = magnitude[y0:y1, x0:x1].ravel()
            cell_ori = orientation[y0:y1, x0:x1].ravel()
            # histogram of orientations, weighted by gradient magnitude
            hist, _ = np.histogram(cell_ori, bins=hog_bins,
                                   range=(0, 180), weights=cell_mag)
            # L2-normalize each cell independently
            norm = np.sqrt(np.sum(hist ** 2) + 1e-6)
            features.extend((hist / norm).tolist())

    # ---- 3. Per-channel percentile statistics (3 × 9 = 27 features) ----
    for ch in range(3):   # R, G, B only
        channel = image[:, :, ch] / 255.0
        features.append(channel.mean())
        features.append(channel.std())
        features.extend(
            np.percentile(channel, [5, 10, 25, 50, 75, 90, 95]).tolist()
        )

    # ---- 4. Local variance texture (8 features) ----
    local_mean = uniform_filter(gray, size=5)
    local_var = uniform_filter((gray - local_mean) ** 2, size=5)
    features.extend([local_var.mean(), local_var.std(), local_var.max()])
    features.extend(
        np.percentile(local_var, [50, 75, 90, 95, 99]).tolist()
    )

    # ---- 5. Connected components at multiple thresholds (7 × 4 = 28) ----
    for threshold in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        binary = (gray > threshold).astype(np.int32)
        labeled_arr, num_cc = label(binary)
        features.append(num_cc)
        if num_cc > 0:
            areas = np.array([np.sum(labeled_arr == j)
                              for j in range(1, num_cc + 1)])
            features.extend([areas.max(), areas.mean(),
                             areas.std() if num_cc > 1 else 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])

    # ---- 6. Row/Column energy projections (128 + 55 = 183 features) ----
    # column projection — mean intensity per column (signal positions)
    features.extend(gray.mean(axis=0).tolist())   # (W,) = 55
    # row projection — mean intensity per row (frequency distribution)
    features.extend(gray.mean(axis=1).tolist())    # (H,) = 128

    return np.array(features, dtype=np.float64)


def load_dataset_comprehensive(image_dir, csv_path, num_bins=64):
    """
    Load dataset with comprehensive features (1366-dim).

    Returns:
        features  : numpy array of shape (num_samples, 1366)
        labels    : numpy array of shape (num_samples,) with values 1-5,
                    or None if csv has no 'label' column (test set)
        filenames : list of image filenames (strings)
    """
    filenames = []
    labels = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filenames.append(row['id'])
            if 'label' in row:
                labels.append(int(row['label']))

    num_samples = len(filenames)

    # determine feature size from first image
    first_img = load_image(os.path.join(image_dir, filenames[0]))
    first_feat = extract_comprehensive_features(first_img, num_bins)
    num_features = len(first_feat)

    features = np.zeros((num_samples, num_features), dtype=np.float64)
    features[0] = first_feat

    print(f'  Extracting comprehensive features ({num_features} dims) '
          f'from {num_samples} images...')
    for i in range(1, num_samples):
        if (i + 1) % 1000 == 0:
            print(f'    {i + 1}/{num_samples}')
        img_path = os.path.join(image_dir, filenames[i])
        image = load_image(img_path)
        features[i] = extract_comprehensive_features(image, num_bins)

    labels_arr = np.array(labels, dtype=np.int32) if labels else None
    return features, labels_arr, filenames


# ============================================================
# RANDOM CONVOLUTIONAL FEATURES  (v10 experiments)
# ============================================================
#
# Applies random 2D filters (3×3, 5×5, 7×7) to the grayscale
# image and pools the responses. This captures spatial patterns
# that hand-crafted features miss, without any learned weights.
#
# Uses scipy.signal.convolve2d — pure signal processing,
# NOT deep learning (no neural network, no backpropagation).
# ============================================================

from scipy.signal import fftconvolve


def extract_rcf_features(image, n_filters_per_size=50, seed=42):
    """
    Extract Random Convolutional Features from an RGBA image.

    Generates random 2D filters of sizes 3×3, 5×5, 7×7 and applies
    them to the grayscale image. For each filter response, computes
    global pooling statistics (mean, max, std).

    Args:
        image             : numpy array of shape (H, W, 4), values [0, 255]
        n_filters_per_size: number of random filters per kernel size
        seed              : random seed for reproducible filters

    Returns:
        1D numpy array of length 3 * 3 * n_filters_per_size
        (default: 3 sizes × 50 filters × 3 stats = 450 features)
    """
    # grayscale, normalized to [0, 1]
    gray = (0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]) / 255.0

    rng = np.random.RandomState(seed)
    features = []

    for kernel_size in [3, 5, 7]:
        for _ in range(n_filters_per_size):
            # random filter with unit variance
            filt = rng.randn(kernel_size, kernel_size)

            # apply convolution (fftconvolve is faster for larger kernels)
            response = fftconvolve(gray, filt, mode='valid')

            # global pooling statistics
            features.append(response.mean())
            features.append(response.max())
            features.append(response.std())

    return np.array(features, dtype=np.float64)


def load_dataset_v10(image_dir, csv_path, num_bins=64,
                     n_filters_per_size=50, rcf_seed=42):
    """
    Load dataset with v10 features: comprehensive (1366-dim) +
    RCF (450-dim) = 1816-dim total.

    Returns:
        features  : numpy array of shape (num_samples, 1816)
        labels    : numpy array of shape (num_samples,) with values 1-5,
                    or None if csv has no 'label' column (test set)
        filenames : list of image filenames (strings)
    """
    filenames = []
    labels = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filenames.append(row['id'])
            if 'label' in row:
                labels.append(int(row['label']))

    num_samples = len(filenames)

    # determine feature size from first image
    first_img = load_image(os.path.join(image_dir, filenames[0]))
    comp_feat = extract_comprehensive_features(first_img, num_bins)
    rcf_feat  = extract_rcf_features(first_img, n_filters_per_size, rcf_seed)
    first_combined = np.concatenate([comp_feat, rcf_feat])
    num_features = len(first_combined)

    features = np.zeros((num_samples, num_features), dtype=np.float64)
    features[0] = first_combined

    print(f'  Extracting v10 features ({num_features} dims: '
          f'{len(comp_feat)} comprehensive + {len(rcf_feat)} RCF) '
          f'from {num_samples} images...')
    for i in range(1, num_samples):
        if (i + 1) % 1000 == 0:
            print(f'    {i + 1}/{num_samples}')
        img_path = os.path.join(image_dir, filenames[i])
        image = load_image(img_path)
        comp = extract_comprehensive_features(image, num_bins)
        rcf  = extract_rcf_features(image, n_filters_per_size, rcf_seed)
        features[i] = np.concatenate([comp, rcf])

    labels_arr = np.array(labels, dtype=np.int32) if labels else None
    return features, labels_arr, filenames
