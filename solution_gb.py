import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle

from feature_extraction import load_dataset
from knn_classifier import confusion_matrix

# ============================================================
# PATHS
# ============================================================

DATA_DIR    = 'signal-object-detection'
TRAIN_DIR   = os.path.join(DATA_DIR, 'train')
TEST_DIR    = os.path.join(DATA_DIR, 'test')
TRAIN_CSV   = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV    = os.path.join(DATA_DIR, 'test.csv')
CACHE_DIR   = 'cache'
RESULTS_DIR = 'results/v8_gb_histograms'
SUBMIT_DIR  = 'submissions'

os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUBMIT_DIR,  exist_ok=True)

# ============================================================
# PARAMETERS
# ============================================================

NUM_BINS     = 64   # histogram bins per channel -> 4 * 64 = 256 total features
RANDOM_STATE = 0

N_ESTIMATORS   = [100, 200]
LEARNING_RATES = [0.1, 0.05]
MAX_DEPTH      = 3

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

# ============================================================
# 3. GRADIENT BOOSTING HYPERPARAMETER SEARCH
# ============================================================

col_header = 'lr \\ n'

print('\n' + '=' * 55)
print('MODEL: Gradient Boosting  (v8 — color histogram features)')
print(f'       max_depth={MAX_DEPTH} fixed')
print('=' * 55)

gb_results = {}

for lr in LEARNING_RATES:
    accuracies = []
    for n in N_ESTIMATORS:
        print(f'\n  n_estimators={n}, learning_rate={lr}, max_depth={MAX_DEPTH}')
        model = GradientBoostingClassifier(
            n_estimators=n, learning_rate=lr,
            max_depth=MAX_DEPTH, random_state=RANDOM_STATE
        )
        model.fit(tr_feat, tr_labels)
        acc = model.score(val_feat, val_labels)
        accuracies.append(acc)
        print(f'  Validation accuracy: {acc:.4f}')
    gb_results[lr] = accuracies

# print results table
print('\n--- Results Table ---')
print(f'{col_header:<10}', end='')
for n in N_ESTIMATORS:
    print(f'  n={n:<8}', end='')
print()
for lr in LEARNING_RATES:
    print(f'{lr:<10}', end='')
    for acc in gb_results[lr]:
        print(f'  {acc:<10.4f}', end='')
    print()

# save results
results_path = os.path.join(RESULTS_DIR, 'gb_results.txt')
with open(results_path, 'w') as f:
    f.write('Gradient Boosting Validation Results (features: 256-dim color histograms)\n')
    f.write(f'max_depth={MAX_DEPTH} (fixed) | Train: {len(tr_labels)} | Val: {len(val_labels)}\n\n')
    f.write(f'{col_header:<10}')
    for n in N_ESTIMATORS:
        f.write(f'  n={n:<8}')
    f.write('\n')
    for lr in LEARNING_RATES:
        f.write(f'{lr:<10}')
        for acc in gb_results[lr]:
            f.write(f'  {acc:<10.4f}')
        f.write('\n')
print(f'\nResults saved -> {results_path}')

# find best
best_acc = 0
best_n   = N_ESTIMATORS[0]
best_lr  = LEARNING_RATES[0]
for lr in LEARNING_RATES:
    for i, n in enumerate(N_ESTIMATORS):
        if gb_results[lr][i] > best_acc:
            best_acc = gb_results[lr][i]
            best_n   = n
            best_lr  = lr

print(f'\nBest GB: n_estimators={best_n}, lr={best_lr}, max_depth={MAX_DEPTH}')
print(f'Val accuracy: {best_acc:.4f}')

best_model = GradientBoostingClassifier(
    n_estimators=best_n, learning_rate=best_lr,
    max_depth=MAX_DEPTH, random_state=RANDOM_STATE
)
best_model.fit(tr_feat, tr_labels)
best_preds = best_model.predict(val_feat)
cm = confusion_matrix(val_labels, best_preds)
print('\nConfusion Matrix (validation set):')
print(cm)

cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_gb.txt')
np.savetxt(cm_path, cm, fmt='%d')
print(f'Confusion matrix saved -> {cm_path}')

# ============================================================
# 4. GENERATE SUBMISSION
# ============================================================

print('\n' + '=' * 55)
print('GENERATING SUBMISSION')
print('=' * 55)

print(f'\nRetraining GB (n={best_n}, lr={best_lr}) on all {len(train_labels)} training samples...')
final_model = GradientBoostingClassifier(
    n_estimators=best_n, learning_rate=best_lr,
    max_depth=MAX_DEPTH, random_state=RANDOM_STATE
)
final_model.fit(train_features, train_labels)
test_predictions = final_model.predict(test_features)

print(f'Prediction distribution: {np.bincount(test_predictions)}')

submit_path = os.path.join(SUBMIT_DIR, 'v8_gb_hist_best.csv')
with open(submit_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'label'])
    for fname, label in zip(test_filenames, test_predictions):
        writer.writerow([fname, label])

print(f'Submission saved -> {submit_path}  ({len(test_predictions)} rows)')
print('\nDone!')
