import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from feature_extraction import load_dataset
from knn_classifier import KnnClassifier, confusion_matrix, compute_accuracy

# ============================================================
# PATHS
# ============================================================

DATA_DIR    = 'signal-object-detection'
TRAIN_DIR   = os.path.join(DATA_DIR, 'train')
TEST_DIR    = os.path.join(DATA_DIR, 'test')
TRAIN_CSV   = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV    = os.path.join(DATA_DIR, 'test.csv')
CACHE_DIR   = 'cache'
RESULTS_DIR = 'results/v5_knn_hist'
SUBMIT_DIR  = 'submissions'

os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUBMIT_DIR,  exist_ok=True)

# ============================================================
# PARAMETERS
# ============================================================

NUM_BINS     = 64   # histogram bins per channel -> 4 * 64 = 256 total features
RANDOM_STATE = 0

K_VALUES = [1, 3, 5, 7, 9, 15, 21, 31, 41]
METRICS  = ['l2', 'l1', 'chi2']

# ============================================================
# 1. LOAD DATA  (with caching)
# ============================================================

train_feat_cache  = os.path.join(CACHE_DIR, 'train_features.npy')
train_label_cache = os.path.join(CACHE_DIR, 'train_labels.npy')
train_fname_cache = os.path.join(CACHE_DIR, 'train_filenames.txt')
test_feat_cache   = os.path.join(CACHE_DIR, 'test_features.npy')
test_fname_cache  = os.path.join(CACHE_DIR, 'test_filenames.txt')

print('=' * 55)
print('Loading training data...')
if os.path.exists(train_feat_cache):
    print('  (loading from cache)')
    train_features = np.load(train_feat_cache)
    train_labels   = np.load(train_label_cache)
    with open(train_fname_cache) as f:
        train_filenames = f.read().splitlines()
else:
    train_features, train_labels, train_filenames = load_dataset(
        TRAIN_DIR, TRAIN_CSV, NUM_BINS
    )
    np.save(train_feat_cache,  train_features)
    np.save(train_label_cache, train_labels)
    with open(train_fname_cache, 'w') as f:
        f.write('\n'.join(train_filenames))

print(f'  {train_features.shape[0]} samples, {train_features.shape[1]} features')
print(f'  Class counts: {np.bincount(train_labels)}')

print('\nLoading test data...')
if os.path.exists(test_feat_cache):
    print('  (loading from cache)')
    test_features = np.load(test_feat_cache)
    with open(test_fname_cache) as f:
        test_filenames = f.read().splitlines()
else:
    test_features, _, test_filenames = load_dataset(
        TEST_DIR, TEST_CSV, NUM_BINS
    )
    np.save(test_feat_cache, test_features)
    with open(test_fname_cache, 'w') as f:
        f.write('\n'.join(test_filenames))

print(f'  {test_features.shape[0]} samples')

# ============================================================
# 2. TRAIN / VALIDATION SPLIT  (80 / 20)
# ============================================================

train_features, train_labels = shuffle(
    train_features, train_labels, random_state=RANDOM_STATE
)

split_idx  = int(0.8 * len(train_labels))
tr_feat    = train_features[:split_idx]
tr_labels  = train_labels[:split_idx]
val_feat   = train_features[split_idx:]
val_labels = train_labels[split_idx:]

print(f'\nTrain split : {len(tr_labels)} samples')
print(f'Val   split : {len(val_labels)} samples')

# ============================================================
# 3. KNN HYPERPARAMETER SEARCH  (Lab 3 style)
# ============================================================

print('\n' + '=' * 55)
print('MODEL: K-Nearest Neighbors  (v5 — color histograms, L2/L1/chi2)')
print('=' * 55)

knn = KnnClassifier(tr_feat, tr_labels)
knn_results = {}

for metric in METRICS:
    accuracies = []
    for k in K_VALUES:
        print(f'\n  K={k}, metric={metric}')
        preds = knn.classify_batch(val_feat, num_neighbors=k, metric=metric)
        acc   = compute_accuracy(val_labels, preds)
        accuracies.append(acc)
        print(f'  Validation accuracy: {acc:.4f}')
    knn_results[metric] = accuracies

# print results table
print('\n--- Results Table ---')
print(f'{"K":<5}', end='')
for metric in METRICS:
    print(f'  {metric.upper():>10}', end='')
print()
for i, k in enumerate(K_VALUES):
    print(f'{k:<5}', end='')
    for metric in METRICS:
        print(f'  {knn_results[metric][i]:>10.4f}', end='')
    print()

# save results
results_path = os.path.join(RESULTS_DIR, 'knn_results.txt')
with open(results_path, 'w') as f:
    f.write(f'KNN Validation Results (features: {train_features.shape[1]}-dim color histograms, {NUM_BINS} bins/channel)\n')
    f.write(f'Metrics: L2, L1, Chi-square | K values: {K_VALUES}\n')
    f.write(f'Train: {len(tr_labels)} samples | Val: {len(val_labels)} samples\n\n')
    f.write(f'{"K":<5}')
    for metric in METRICS:
        f.write(f'  {metric.upper():>10}')
    f.write('\n')
    for i, k in enumerate(K_VALUES):
        f.write(f'{k:<5}')
        for metric in METRICS:
            f.write(f'  {knn_results[metric][i]:>10.4f}')
        f.write('\n')
print(f'\nResults saved -> {results_path}')

# plot
plt.figure(figsize=(9, 5))
markers = {'l2': 'o', 'l1': 's', 'chi2': '^'}
for metric in METRICS:
    plt.plot(K_VALUES, knn_results[metric], marker=markers[metric], label=f'{metric.upper()} distance')
plt.xlabel('Number of neighbors (K)')
plt.ylabel('Validation Accuracy')
plt.title('KNN: Accuracy vs K (color histogram features, L2 / L1 / Chi-square)')
plt.legend()
plt.grid(True)
plot_path = os.path.join(RESULTS_DIR, 'knn_accuracy_vs_k.png')
plt.savefig(plot_path)
plt.close()
print(f'Plot saved    -> {plot_path}')

# find best
best_acc    = 0
best_k      = K_VALUES[0]
best_metric = METRICS[0]
for metric in METRICS:
    for i, k in enumerate(K_VALUES):
        if knn_results[metric][i] > best_acc:
            best_acc    = knn_results[metric][i]
            best_k      = k
            best_metric = metric

print(f'\nBest KNN: K={best_k}, metric={best_metric}, val accuracy={best_acc:.4f}')

best_preds = knn.classify_batch(val_feat, num_neighbors=best_k, metric=best_metric)
cm = confusion_matrix(val_labels, best_preds)
print('\nConfusion Matrix (validation set):')
print(cm)

cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_knn.txt')
np.savetxt(cm_path, cm, fmt='%d')
print(f'Confusion matrix saved -> {cm_path}')

# ============================================================
# 4. GENERATE SUBMISSION
# ============================================================

print('\n' + '=' * 55)
print('GENERATING SUBMISSION')
print('=' * 55)

print(f'\nRetraining KNN (K={best_k}, {best_metric}) on all {len(train_labels)} training samples...')
final_knn        = KnnClassifier(train_features, train_labels)
test_predictions = final_knn.classify_batch(
    test_features, num_neighbors=best_k, metric=best_metric
)

print(f'Prediction distribution: {np.bincount(test_predictions)}')

submit_path = os.path.join(SUBMIT_DIR, 'v5_knn_hist_best.csv')
with open(submit_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'label'])
    for fname, label in zip(test_filenames, test_predictions):
        writer.writerow([fname, label])

print(f'Submission saved -> {submit_path}  ({len(test_predictions)} rows)')
print('\nDone!')
