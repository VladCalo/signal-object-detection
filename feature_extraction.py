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
