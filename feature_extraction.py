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
# SPATIAL HISTOGRAM FEATURES  (v9+ experiments)
# ============================================================

def extract_spatial_histogram_features(image, num_bins=32, grid_rows=4, grid_cols=2):
    """
    Extract spatial color histogram features from an RGBA image.

    The image is divided into a grid_rows x grid_cols grid of cells.
    For each cell, a color histogram with num_bins bins is computed
    for R, G, B channels only (alpha is always 255 — constant, excluded).
    All per-cell histograms are concatenated into a single feature vector.

    Default parameters (grid_rows=4, grid_cols=2, num_bins=32):
        Images are 128 rows (H, frequency axis) x 55 cols (W, time axis).
        4 frequency bands x 2 time windows x 3 channels x 32 bins = 768 features.

    Physical motivation (spectrograms):
        - Row (frequency) axis: 4 bands capture frequency structure of signals
        - Column (time) axis: 2 windows capture left/right temporal distribution
        - A spectrogram with N objects will show N cells with elevated
          signal color intensity, directly discriminating the 5 classes

    Args:
        image     : numpy array of shape (H, W, 4), pixel values in [0, 255]
        num_bins  : histogram bins per channel per cell
        grid_rows : number of vertical divisions (frequency axis)
        grid_cols : number of horizontal divisions (time axis)

    Returns:
        feature vector of shape (grid_rows * grid_cols * 3 * num_bins,)
    """
    H, W, _ = image.shape
    cell_h = H // grid_rows
    cell_w = W // grid_cols

    features = []
    for r in range(grid_rows):
        # last row absorbs any remainder pixels
        r0 = r * cell_h
        r1 = (r + 1) * cell_h if r < grid_rows - 1 else H
        for c in range(grid_cols):
            c0 = c * cell_w
            c1 = (c + 1) * cell_w if c < grid_cols - 1 else W

            cell = image[r0:r1, c0:c1]   # (cell_h, cell_w, 4)

            for ch in range(3):  # R, G, B only — skip alpha
                hist, _ = np.histogram(cell[:, :, ch], bins=num_bins, range=(0, 256))
                hist = hist.astype(np.float64)
                total = hist.sum()
                if total > 0:
                    hist /= total          # normalize to sum to 1
                features.append(hist)

    return np.concatenate(features)


def load_dataset_spatial(image_dir, csv_path, num_bins=32, grid_rows=4, grid_cols=2):
    """
    Load dataset: read CSV for filenames/labels, then extract spatial
    histogram features (RGB only, alpha excluded) from each image.

    Returns:
        features  : numpy array of shape (num_samples, grid_rows*grid_cols*3*num_bins)
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

    num_samples  = len(filenames)
    num_features = grid_rows * grid_cols * 3 * num_bins  # RGB only
    features     = np.zeros((num_samples, num_features), dtype=np.float64)

    print(f'  Extracting spatial histograms ({grid_rows}x{grid_cols} grid, '
          f'{num_bins} bins/ch, RGB only) from {num_samples} images...')
    for i, fname in enumerate(filenames):
        if (i + 1) % 1000 == 0:
            print(f'    {i + 1}/{num_samples}')
        img_path = os.path.join(image_dir, fname)
        image    = load_image(img_path)
        features[i] = extract_spatial_histogram_features(image, num_bins, grid_rows, grid_cols)

    labels_arr = np.array(labels, dtype=np.int32) if labels else None
    return features, labels_arr, filenames
def normalize_data(train_data, test_data, norm_type='standard'):
    """
    Normalize train and test data. Always fits only on train_data.

    norm_type options:
        'standard' - zero mean, unit variance (StandardScaler)
        'l1'       - scale each sample to unit L1 norm
        'l2'       - scale each sample to unit L2 norm
        None       - no normalization

    Returns: (scaled_train, scaled_test)
    """
    from sklearn import preprocessing

    if norm_type == 'standard':
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        return scaler.transform(train_data), scaler.transform(test_data)
    elif norm_type == 'l1':
        return (preprocessing.normalize(train_data, norm='l1'),
                preprocessing.normalize(test_data, norm='l1'))
    elif norm_type == 'l2':
        return (preprocessing.normalize(train_data, norm='l2'),
                preprocessing.normalize(test_data, norm='l2'))
    else:
        return train_data, test_data


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
