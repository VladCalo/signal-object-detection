import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.utils import shuffle

from feature_extraction import load_dataset
from knn_classifier import confusion_matrix, compute_accuracy

# ============================================================
# PATHS
# ============================================================

DATA_DIR    = 'signal-object-detection'
TRAIN_DIR   = os.path.join(DATA_DIR, 'train')
TEST_DIR    = os.path.join(DATA_DIR, 'test')
TRAIN_CSV   = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV    = os.path.join(DATA_DIR, 'test.csv')
CACHE_DIR   = 'cache'
RESULTS_DIR = 'results/v6_svm_hist'
SUBMIT_DIR  = 'submissions'

os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUBMIT_DIR,  exist_ok=True)

# ============================================================
# PARAMETERS
# ============================================================

NUM_BINS     = 64   # histogram bins per channel -> 4 * 64 = 256 total features
RANDOM_STATE = 0

C_VALUES = [0.1, 1, 10]
KERNELS  = ['linear', 'rbf']

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

# normalize — StandardScaler fit on train only
scaler = preprocessing.StandardScaler()
scaler.fit(tr_feat)
tr_norm  = scaler.transform(tr_feat)
val_norm = scaler.transform(val_feat)

# ============================================================
# 3. SVM HYPERPARAMETER SEARCH  (Lab 4 style)
# ============================================================

print('\n' + '=' * 55)
print('MODEL: Support Vector Machine  (v6 — color histogram features)')
print('=' * 55)

svm_results = {}

for kernel in KERNELS:
    accuracies = []
    for C in C_VALUES:
        print(f'\n  kernel={kernel}, C={C}')
        model = svm.SVC(C=C, kernel=kernel, gamma='scale')
        model.fit(tr_norm, tr_labels)
        acc = model.score(val_norm, val_labels)
        accuracies.append(acc)
        print(f'  Validation accuracy: {acc:.4f}')
    svm_results[kernel] = accuracies

# print results table
print('\n--- Results Table ---')
print(f'{"Kernel":<10}', end='')
for C in C_VALUES:
    print(f'  C={C:<8}', end='')
print()
for kernel in KERNELS:
    print(f'{kernel:<10}', end='')
    for acc in svm_results[kernel]:
        print(f'  {acc:<10.4f}', end='')
    print()

# save results
results_path = os.path.join(RESULTS_DIR, 'svm_results.txt')
with open(results_path, 'w') as f:
    f.write('SVM Validation Results (features: 256-dim color histograms)\n')
    f.write(f'Train: {len(tr_labels)} samples | Val: {len(val_labels)} samples\n\n')
    f.write(f'{"Kernel":<10}')
    for C in C_VALUES:
        f.write(f'  C={C:<8}')
    f.write('\n')
    for kernel in KERNELS:
        f.write(f'{kernel:<10}')
        for acc in svm_results[kernel]:
            f.write(f'  {acc:<10.4f}')
        f.write('\n')
print(f'\nResults saved -> {results_path}')

# find best
best_acc    = 0
best_C      = C_VALUES[0]
best_kernel = KERNELS[0]
for kernel in KERNELS:
    for i, C in enumerate(C_VALUES):
        if svm_results[kernel][i] > best_acc:
            best_acc    = svm_results[kernel][i]
            best_C      = C
            best_kernel = kernel

print(f'\nBest SVM: kernel={best_kernel}, C={best_C}, val accuracy={best_acc:.4f}')

best_model = svm.SVC(C=best_C, kernel=best_kernel, gamma='scale')
best_model.fit(tr_norm, tr_labels)
best_preds = best_model.predict(val_norm)
cm = confusion_matrix(val_labels, best_preds)
print('\nConfusion Matrix (validation set):')
print(cm)

cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_svm.txt')
np.savetxt(cm_path, cm, fmt='%d')
print(f'Confusion matrix saved -> {cm_path}')

# ============================================================
# 4. GENERATE SUBMISSION
# ============================================================

print('\n' + '=' * 55)
print('GENERATING SUBMISSION')
print('=' * 55)

scaler_final = preprocessing.StandardScaler()
scaler_final.fit(train_features)
train_full_norm = scaler_final.transform(train_features)
test_full_norm  = scaler_final.transform(test_features)

print(f'\nRetraining SVM (kernel={best_kernel}, C={best_C}) on all {len(train_labels)} training samples...')
final_model = svm.SVC(C=best_C, kernel=best_kernel, gamma='scale')
final_model.fit(train_full_norm, train_labels)
test_predictions = final_model.predict(test_full_norm)

print(f'Prediction distribution: {np.bincount(test_predictions)}')

submit_path = os.path.join(SUBMIT_DIR, 'v6_svm_hist_best.csv')
with open(submit_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'label'])
    for fname, label in zip(test_filenames, test_predictions):
        writer.writerow([fname, label])

print(f'Submission saved -> {submit_path}  ({len(test_predictions)} rows)')
print('\nDone!')
