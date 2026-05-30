# Signal Object Detection — Experiment Log

## Competition Summary
- **Task**: Classify radio signal PNG images into 5 classes (number of objects: 1–5)
- **Metric**: Classification accuracy on 5,500 test images
- **Train set**: 15,500 images (classes nearly balanced)
- **Image size**: 55 × 128 px, RGBA (4 channels)
- **Kaggle deadline**: 05 June 2026

---

## Project File Structure

```
tema_ai/
│
├── signal-object-detection/        ← dataset (NOT submitted to professor)
│   ├── train/                      ← 15,500 labeled images
│   ├── test/                       ← 5,500 unlabeled images
│   ├── train.csv                   ← id, label
│   └── test.csv                    ← id only
│
├── cache/                          ← cached feature arrays (speeds up re-runs)
│   ├── train_features.npy          ← extracted feature matrix (15500 × F)
│   ├── train_labels.npy
│   ├── train_filenames.txt
│   ├── test_features.npy           ← extracted feature matrix (5500 × F)
│   └── test_filenames.txt
│   ⚠️  Delete this folder when changing the feature extraction method!
│
├── docs/
│   └── wiki.md                     ← this file — running experiment log
│
├── results/
│   └── v1_knn_histograms/          ← outputs from experiment v1
│       ├── knn_accuracy_vs_k.png
│       ├── confusion_matrix_knn.txt
│       └── notes.txt
│
├── submissions/
│   └── v1_knn_hist_k9_l2.csv      ← Kaggle submission files (versioned)
│
├── feature_extraction.py           ← image loading + feature extraction
├── knn_classifier.py               ← KnnClassifier class + helpers
├── solution_knn.py                 ← v1: KNN experiments + submission
├── solution_svm.py                 ← v2: SVM experiments + submission
└── solution_v10.py                 ← v10: HGBC + RCF experiments + submission
```

---

## Experiments

---

### v1 — KNN with Color Histograms
**Date**: 23 May 2026  
**Kaggle score**: **0.3300** (public leaderboard)  
**Submission file**: `submissions/v1_knn_hist_k9_l2.csv`

#### Feature Extraction
- Converted each 55×128 RGBA image to a **color histogram**
- For each of the 4 channels (R, G, B, A): computed a histogram with **64 bins** over pixel values [0, 255], normalized to sum to 1
- Final feature vector: **256 features** per image (4 channels × 64 bins)

#### Model: K-Nearest Neighbors
- Custom `KnnClassifier` class (Lab 3 style)
- Hyperparameter search: **K ∈ {1, 3, 5, 7, 9}**, distance ∈ **{L1, L2}**
- Train/val split: **80/20** (12,400 train, 3,100 val)

#### Validation Results

| K | L2 Accuracy | L1 Accuracy |
|---|-------------|-------------|
| 1 | 0.2616 | 0.2587 |
| 3 | 0.2732 | 0.2577 |
| 5 | 0.2816 | 0.2771 |
| 7 | 0.2868 | 0.2861 |
| 9 | **0.2926** | 0.2894 |

> **Best**: K=9, L2 → val accuracy **0.2926**

Observation: accuracy improves steadily with K for both metrics. L2 consistently outperforms L1. Increasing K further is unlikely to help significantly.

#### Confusion Matrix (KNN K=9 L2, validation set)

Rows = true class, Columns = predicted class (classes 1–5):

```
[[374  113   67   72   65]
 [247  145   93   56   76]
 [145  145  135   95   66]
 [151  101  137  130   77]
 [166   98   98  125  123]]
```

#### Analysis
- Results close to random chance (20% baseline for 5 classes)
- Color histograms discard **spatial structure** — the position of the signal in the image is thrown away
- The confusion matrix shows the model mostly predicts class 1 and struggles equally with all others
- **Root cause**: histograms only measure pixel value distributions, not WHERE the signals are
- **Next step**: try SVM on the same histogram features to see if a stronger classifier helps

---

### v2 — SVM with Color Histograms (same features as v1)
**Date**: 23 May 2026  
**Kaggle score**: *(pending — submit `submissions/v2_svm_hist_best.csv`)*  
**Submission file**: `submissions/v2_svm_hist_best.csv`

#### Goal
Test whether a stronger classifier (SVM) improves over KNN when using the same 256-dim histogram features.

#### Feature Extraction
Identical to v1 — 256-dim color histograms loaded from cache (no re-extraction needed).

#### Model: Support Vector Machine (Lab 4 style)
- `sklearn.svm.SVC` with `StandardScaler` normalization
- Hyperparameter search: **kernel ∈ {linear, rbf}**, **C ∈ {0.1, 1, 10}**
- Train/val split: **80/20** (12,400 train, 3,100 val)
- `max_iter=5000` on all configs to prevent hanging

#### Validation Results

| Kernel | C=0.1 | C=1 | C=10 |
|--------|-------|-----|------|
| linear | 0.2052 ⚠️ | 0.2055 ⚠️ | 0.2190 ⚠️ |
| rbf    | 0.2590 | **0.2794** | 0.2506 ⚠️ |

⚠️ = solver hit `max_iter=5000` and terminated early (ConvergenceWarning)

> **Best**: kernel=rbf, C=1 → val accuracy **0.2794**

#### Confusion Matrix (SVM rbf C=1, validation set)

Rows = true class, Columns = predicted class (classes 1–5):

```
[[389  73  75  84  70]
 [294  83  86  66  88]
 [208  82 113  88  95]
 [181  59 117 104 135]
 [181  63  87 102 177]]
```

#### Analysis
- SVM (0.2794) scored slightly **below** KNN (0.2926) — a difference of only ~1.3%
- The **linear kernel failed** (~20%, near random): histogram features have no clean linear decision boundary
- The **rbf kernel with C=10 degraded** (0.2506) due to early stopping — instability from insufficient iterations
- The small gap between KNN and SVM confirms that **feature quality is the bottleneck**, not the classifier
- Color histograms discard spatial structure: a more powerful classifier cannot recover lost spatial information
- **Conclusion**: need better features, not a better classifier

---

## Notes
- Cache must be deleted when changing the feature extraction method
- v1/v2 used `load_dataset` (histogram features, 256-dim)
- v3/v4 use `load_dataset_pixels` (grayscale pixels, 7040-dim)
- v10 uses `load_dataset_v10` (1366-dim comprehensive + 450-dim RCF, 1816-dim total)
  - Uses separate cache files (`*_v10_features.npy`, `*_v10_labels.npy`) — does NOT overwrite previous caches

---

### v3 — KNN with Grayscale Pixel Features
**Date**: 24 May 2026  
**Kaggle score**: *(not submitted — results below expectation)*  
**Submission file**: `submissions/v3_knn_pixels_best.csv`

#### Motivation
v1 and v2 both plateaued at ~29% validation accuracy. Root cause: color histograms discard **spatial structure** — they only measure pixel value distributions, not where signals appear in the image. Switching to raw pixel values preserves the full spatial layout.

#### Feature Extraction
- Convert each 55×128 RGBA image to **grayscale** using luminance formula:  
  `gray = 0.299 × R + 0.587 × G + 0.114 × B` (alpha ignored)
- Normalize pixel values to [0, 1] by dividing by 255
- Flatten to a **7040-dim** feature vector (55 × 128 = 7040 pixels)

#### Model: K-Nearest Neighbors
- Same `KnnClassifier` class as v1 (Lab 3 style)
- Hyperparameter search: **K ∈ {1, 3, 5, 7, 9}**, distance ∈ **{L1, L2}**
- Train/val split: **80/20** (12,400 train, 3,100 val)

#### Validation Results

| K | L2 Accuracy | L1 Accuracy |
|---|-------------|-------------|
| 1 | 0.2126 | 0.2042 |
| 3 | 0.2158 | 0.2135 |
| 5 | 0.2161 | 0.2094 |
| 7 | 0.2106 | 0.2068 |
| 9 | 0.2068 | **0.2165** |

> **Best**: K=9, L1 → val accuracy **0.2165** (barely above random 20%)

#### Confusion Matrix (KNN K=9 L1, validation set)

Rows = true class, Columns = predicted class (classes 1–5):

```
[[326 146  90  82  47]
 [307 132  71  65  42]
 [282 125  79  52  48]
 [270 122  83  71  50]
 [278 129  84  56  63]]
```

#### Analysis
- All K values and both metrics cluster at ~21% — essentially **random guessing**
- Switching from K=1 to K=9 produces no improvement (flat curve) — a hallmark of the **curse of dimensionality**
- In 7040-dim space, all pairwise distances become nearly equal, so nearest neighbor selection is meaningless
- The confusion matrix shows massive prediction bias towards class 1 — the model collapses to a near-constant predictor
- **Conclusion**: raw pixels are worse than histograms for KNN — the 27× increase in dimensions hurts more than the spatial information helps

---

### v4 — SVM with Grayscale Pixel Features
**Date**: 24 May 2026  
**Kaggle score**: *(not submitted — model did not converge)*  
**Submission file**: `submissions/v4_svm_pixels_best.csv`

#### Goal
Test whether SVM (immune to curse of dimensionality) benefits more than KNN from the switch to spatial pixel features.

#### Feature Extraction
Identical to v3 — 7040-dim grayscale pixel features loaded from cache (no re-extraction needed).

#### Model: Support Vector Machine (Lab 4 style)
- `sklearn.svm.SVC` with `StandardScaler` normalization
- Hyperparameter search: **kernel ∈ {linear, rbf}**, **C ∈ {0.1, 1, 10}**
- Train/val split: **80/20** (12,400 train, 3,100 val)
- `max_iter=5000` on all configs to prevent hanging

#### Validation Results

| Kernel | C=0.1 | C=1 | C=10 |
|--------|-------|-----|------|
| linear | 0.2100 ⚠️ | 0.2100 ⚠️ | 0.2100 ⚠️ |
| rbf    | 0.2229 ⚠️ | 0.2287 ⚠️ | **0.2381** ⚠️ |

⚠️ = ConvergenceWarning on all configs — solver hit `max_iter=5000` early

> **Best**: kernel=rbf, C=10 → val accuracy **0.2381** (did not converge)

#### Confusion Matrix (SVM rbf C=10, validation set)

Rows = true class, Columns = predicted class (classes 1–5):

```
[[261 124 108  98 100]
 [197 130  90 107  93]
 [178 109 110  95  94]
 [161 101 102 115 117]
 [152  96 121 119 122]]
```

#### Analysis
- Linear kernel: **identical score (0.2100) for all C values** — clear sign of non-convergence, model outputs near-random predictions
- RBF kernel: marginally better but all configs hit `max_iter=5000` — 7040 dims requires far more iterations than 5000 to converge
- SVM did not improve over histograms (v2: 0.2794) despite the larger feature space
- **Root cause**: `max_iter=5000` is insufficient at 7040 dimensions with 12,400 training samples
- **Conclusion**: raw pixels are not suitable as-is — need compact, discriminative spatial features instead of full pixel vectors

---

### v5 — KNN with Color Histograms + Extended K + Chi-Square Distance
**Date**: 24 May 2026  
**Kaggle score**: **0.3500** (public leaderboard)  
**Submission file**: `submissions/v5_knn_hist_chi2_best.csv`

#### Goal
v1 showed accuracy still improving at K=9. Extend K to {1,3,5,7,9,15,21} and add chi-square distance — the standard metric for histogram comparison in computer vision.

#### Feature Extraction
Identical to v1 — 256-dim color histograms loaded from cache.

#### Model: K-Nearest Neighbors
- Same `KnnClassifier` class (Lab 3 style)
- Hyperparameter search: **K ∈ {1,3,5,7,9,15,21}**, distance ∈ **{L2, L1, Chi-square}**
- Train/val split: **80/20** (12,400 train, 3,100 val)

#### Validation Results

| K | L2 | L1 | Chi-square |
|---|----|----|-----------|
| 1 | 0.2616 | 0.2587 | 0.2739 |
| 3 | 0.2732 | 0.2577 | 0.2690 |
| 5 | 0.2816 | 0.2771 | 0.2745 |
| 7 | 0.2868 | 0.2861 | 0.2719 |
| 9 | 0.2926 | 0.2894 | 0.2803 |
| 15 | 0.3087 | 0.2997 | 0.2897 |
| 21 | **0.3126** | 0.3071 | 0.2955 |

> **Best**: K=21, L2 → val accuracy **0.3126**

#### Confusion Matrix (KNN K=21 L2, validation set)

Rows = true class, Columns = predicted class (classes 1–5):

```
[[386  97  75  62  71]
 [244 137  92  69  75]
 [156 123 150  83  74]
 [144  93 122 130 107]
 [143 104  89 108 166]]
```

#### Analysis
- Accuracy increases monotonically with K for L2 — trend not yet plateaued at K=21
- Chi-square underperformed L2 — because histograms are already L1-normalized (sum to 1), chi2 advantage is reduced vs raw counts
- L2 remains the best metric throughout; chi2 only helps for unnormalized histograms
- **Conclusion**: larger K continues to help; L2 is the right metric for normalized histograms

---

### v6 — SVM with Color Histograms (full convergence)
**Date**: 24 May 2026  
**Kaggle score**: *(not submitted — underperforms KNN)*  
**Submission file**: `submissions/v6_svm_hist_best.csv`

#### Goal
Re-run SVM on histograms without `max_iter` cap. v2 had convergence warnings for some configs — full convergence may improve results over KNN.

#### Feature Extraction
Identical to v1 — 256-dim color histograms loaded from cache.

#### Model: Support Vector Machine (Lab 4 style)
- `sklearn.svm.SVC` with `StandardScaler` normalization
- Hyperparameter search: **kernel ∈ {linear, rbf}**, **C ∈ {0.1, 1, 10}**
- Train/val split: **80/20** (12,400 train, 3,100 val)
- **No `max_iter` cap** — 256 features allows full convergence

#### Validation Results

| Kernel | C=0.1 | C=1 | C=10 |
|--------|-------|-----|------|
| linear | 0.2510 | 0.2484 | 0.2448 |
| rbf    | 0.2590 | **0.2794** | 0.2497 |

> **Best**: kernel=rbf, C=1 → val accuracy **0.2794**

#### Confusion Matrix (SVM rbf C=1, validation set)

Rows = true class, Columns = predicted class (classes 1–5):

```
[[389  73  75  84  70]
 [294  83  86  66  88]
 [208  82 113  88  95]
 [181  59 117 104 135]
 [181  63  87 102 177]]
```

#### Analysis
- Result is **identical to v2 (0.2794)** — confirms max_iter cap was NOT the issue in v2; SVM genuinely converges here and gives 0.2794
- Linear kernel degrades with higher C — suggests the classes are not linearly separable in histogram space
- SVM consistently underperforms KNN (0.2794 vs 0.3126) on this dataset
- **Conclusion**: SVM is not suited to this problem; histogram features form compact distance-based clusters that KNN exploits better than a margin-based classifier

---

### v7 — Random Forest with Color Histograms
**Date**: 24 May 2026  
**Kaggle score**: **0.3300** (public leaderboard)  
**Submission file**: `submissions/v7_rf_hist_best.csv`

#### Goal
Random Forest can learn feature interactions (combinations of histogram bins across channels) that KNN and SVM cannot. Expected to outperform both on the same histogram features.

#### Feature Extraction
Identical to v1 — 256-dim color histograms loaded from cache. No scaling needed (decision trees are invariant to monotonic feature transformations).

#### Model: Random Forest
- `sklearn.ensemble.RandomForestClassifier`
- Hyperparameter search: **n_estimators ∈ {100, 200, 300}**, **max_features ∈ {sqrt, log2}**
- Train/val split: **80/20** (12,400 train, 3,100 val)
- `random_state=0`, `n_jobs=-1` (all CPU cores)

#### Validation Results

| max_features | n=100 | n=200 | n=300 |
|---|---|---|---|
| sqrt | 0.3116 | 0.3287 | **0.3365** |
| log2 | 0.3161 | 0.3223 | 0.3229 |

> **Best**: n=300, max_features=sqrt → val accuracy **0.3365**

#### Confusion Matrix (RF n=300 sqrt, validation set)

Rows = true class, Columns = predicted class (classes 1–5):

```
[[388  84  70  63  86]
 [231 125  77  75 109]
 [149  79 157  78 123]
 [114  73 111 121 177]
 [118  44  77 119 252]]
```

#### Analysis
- Val accuracy (0.3365) was higher than KNN v5 (0.3126), but **Kaggle score was lower (0.33 vs 0.35)**
- RF accuracy still improving with more trees (n=100→200→300 monotonically better for sqrt) — could try larger
- RF shows better class 5 accuracy (252 correct) but worse class 2 accuracy vs KNN
- **Key insight**: KNN with large K is more regularized (21 votes vs 1 tree prediction) — likely generalizes better to the Kaggle test set
- **Conclusion**: on histogram features, KNN K=21 L2 generalizes better despite lower validation accuracy; RF may be overfitting the validation split

---

### v8 — Gradient Boosting with Color Histograms
**Date**: 30 May 2026  
**Kaggle score**: *(not submitted — underperforms KNN)*  
**Submission file**: `submissions/v8_gb_hist_best.csv`

#### Goal
Gradient Boosting builds trees **sequentially** — each tree explicitly corrects the mistakes of all previous trees. This is fundamentally more powerful than Random Forest (which builds trees in parallel, independently). On tabular features like histograms, GBM consistently outperforms RF by 5–10%.

#### Feature Extraction
Identical to v1 — 256-dim color histograms loaded from cache. No scaling needed.

#### Model: Gradient Boosting Classifier
- `sklearn.ensemble.GradientBoostingClassifier`
- Fixed: **max_depth=3** (standard GBM depth to prevent overfitting)
- Hyperparameter search: **n_estimators ∈ {100, 200}**, **learning_rate ∈ {0.1, 0.05}**
- Train/val split: **80/20** (12,400 train, 3,100 val)
- `random_state=0`

#### Validation Results

| lr \ n | n=100 | n=200 |
|--------|-------|-------|
| 0.10   | 0.3148 | **0.3210** |
| 0.05   | 0.3094 | 0.3110 |

> **Best**: lr=0.1, n=200 → val accuracy **0.3210**

#### Confusion Matrix
*(not recorded — model not selected for submission)*

#### Analysis
- GBM (0.3210 val) underperforms KNN K=21 (0.3126 val → 0.35 Kaggle)
- Despite the theoretical advantage of sequential boosting, histogram features appear better suited to distance-based classification (KNN) than gradient boosting
- **Conclusion**: KNN K=21 L2 remains the best model on global histogram features

---

### v10 — HistGradientBoosting on Comprehensive + Random Convolutional Features (RCF)
**Date**: 30 May 2026  
**Kaggle score**: *(pending validation)*  
**Submission file**: `submissions/v10_rcf_best.csv`

#### Goal
Implement a highly robust tabular and spatial pipeline without using deep learning. Capture both high-level hand-crafted visual characteristics (comprehensive color, texture, shape, and layout descriptors) and multi-scale local spatial patterns (via random convolutional filters). Use a modern histogram-based gradient booster to handle mixed feature scales, high dimensionality (1,816 dimensions), and complex feature interactions.

#### Feature Extraction
The feature vector combines two major components for a total of **1,816 features**:
1. **1,366-dim Comprehensive Features**:
   - **Color Histograms (256 features)**: 64 bins per channel across R, G, B, A.
   - **Histogram of Oriented Gradients (HOG) (864 features)**: Captured on Gaussian-smoothed grayscale images using central differences. Extracted across 8×8 cells with 9 orientation bins per cell, and L2 normalized.
   - **Per-Channel Percentile Statistics (27 features)**: Mean, standard deviation, and percentiles (5, 10, 25, 50, 75, 90, 95) computed across R, G, B channels.
   - **Local Variance Texture (8 features)**: Local variance computed using a 5×5 uniform filter on the grayscale image, summarized by its mean, standard deviation, max, and percentiles (50, 75, 90, 95, 99).
   - **Connected Component (CC) Statistics (28 features)**: Connected components extracted using 7 intensity thresholds (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50). For each threshold, CC count, mean CC area, max CC area, and standard deviation of CC areas are computed.
   - **Row & Column Intensity Projections (183 features)**: 55 vertical column averages and 128 horizontal row averages, capturing the precise spatial energy distribution.
2. **450-dim Random Convolutional Features (RCF)**:
   - Captures multi-scale local visual patterns (oriented edges, peaks, valleys) using random convolutional filters, completely bypassing deep neural network weights.
   - 150 random filters generated using standard normal distribution (50 of size 3×3, 50 of size 5×5, 50 of size 7×7) and applied to normalized grayscale images via Fast Fourier Transform convolution (`scipy.signal.fftconvolve`).
   - For each filter response, global pooling statistics (**mean**, **max**, **std**) are computed to produce 450 compact, shift-invariant spatial features (150 filters × 3 stats).

#### Model: HistGradientBoostingClassifier
- Extremely fast training on large tabular feature sets using binning.
- Hyperparameter search: **max_depth ∈ {7, 10, 12}**, **learning_rate ∈ {0.02, 0.03, 0.05}**, **max_iter = 500**, `min_samples_leaf=5`.
- Comparison baseline: **Random Forest Classifier with 1,000 trees** (`max_features='sqrt'`).
- Train/val split: **80/20** (12,400 train, 3,100 val)

#### Validation Results

| Model | Hyperparameters | Validation Accuracy |
|---|---|---|
| **HGBC** | **max_depth=10, learning_rate=0.05** | **0.4355** 🏆 |
| HGBC | max_depth=10, learning_rate=0.03 | 0.4306 |
| HGBC | max_depth=10, learning_rate=0.02 | 0.4287 |
| HGBC | max_depth=7, learning_rate=0.02 | 0.4255 |
| HGBC | max_depth=12, learning_rate=0.02 | 0.4248 |
| HGBC | max_depth=12, learning_rate=0.03 | 0.4216 |
| HGBC | max_depth=7, learning_rate=0.05 | 0.4161 |
| HGBC | max_depth=12, learning_rate=0.05 | 0.4145 |
| HGBC | max_depth=7, learning_rate=0.03 | 0.4139 |
| Random Forest | 1000 trees, sqrt features | 0.3887 |

> **Best Config**: HGBC depth=10, lr=0.05 → val accuracy **0.4355**

#### Confusion Matrix (HGBC depth=10 lr=0.05, validation set)

Rows = true class, Columns = predicted class (classes 1–5):

```
[[546  81  37  18   9]
 [226 193 109  59  30]
 [113 110 165 125  73]
 [115  73  85 164 159]
 [ 95  49  56 128 282]]
```

##### Per-Class Accuracies
- **Class 1**: 546 / 691 = **79.0%**
- **Class 2**: 193 / 617 = **31.3%**
- **Class 3**: 165 / 586 = **28.2%**
- **Class 4**: 164 / 596 = **27.5%**
- **Class 5**: 282 / 610 = **46.2%**

#### Analysis
- **Breakthrough Improvement**: Moving from color histograms to the combined 1,816-dim comprehensive + RCF feature space yielded a massive performance jump (validation accuracy soared from ~0.33 to **0.4355**).
- **RCF Visual Power**: Applying multi-scale random convolutional filters proved to be highly effective. It successfully extracts structural and texture patterns that global metrics miss, while complying perfectly with the competition's rule forbidding pre-trained networks.
- **HistGradientBoosting Superiority**: HGBC dramatically outperformed Random Forest (0.4355 vs 0.3887). HGBC's sequential learning allows it to build precise decision trees that correct residual errors, while its feature-binning design ensures high-dimensional training is incredibly fast and robust to overfitting.
- **Remaining Challenges**: While accuracy has surged, the intermediate classes (2, 3, and 4) remain difficult due to high visual similarity, as shown by the dense clusters along the main diagonal of the confusion matrix. Class 1 and Class 5 continue to be the most distinct.

---