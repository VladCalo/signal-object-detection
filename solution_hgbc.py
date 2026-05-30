import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import shuffle

from feature_extraction import load_dataset_comprehensive
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
RESULTS_DIR = 'results/v9_hgbc_comprehensive'
SUBMIT_DIR  = 'submissions'

os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUBMIT_DIR,  exist_ok=True)

# ============================================================
# PARAMETERS
# ============================================================

NUM_BINS     = 64
RANDOM_STATE = 0

# HGBC hyperparameter grid
HGBC_DEPTHS = [5, 7]
HGBC_LRS    = [0.03, 0.05, 0.1]
HGBC_ITERS  = [500]

# ============================================================
# 1. LOAD DATA  (with caching)
# ============================================================

train_feat_cache  = os.path.join(CACHE_DIR, 'train_comprehensive_features.npy')
train_label_cache = os.path.join(CACHE_DIR, 'train_comprehensive_labels.npy')
train_fname_cache = os.path.join(CACHE_DIR, 'train_comprehensive_filenames.txt')
test_feat_cache   = os.path.join(CACHE_DIR, 'test_comprehensive_features.npy')
test_fname_cache  = os.path.join(CACHE_DIR, 'test_comprehensive_filenames.txt')

print('=' * 60)
print('v9 — HistGradientBoosting on comprehensive features')
print('     (histograms + HOG + channel stats + texture + CC + projections)')
print('=' * 60)

print('\nLoading training data...')
if os.path.exists(train_feat_cache):
    print('  (loading from cache)')
    train_features = np.load(train_feat_cache)
    train_labels   = np.load(train_label_cache)
    with open(train_fname_cache) as f:
        train_filenames = f.read().splitlines()
else:
    train_features, train_labels, train_filenames = load_dataset_comprehensive(
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
    test_features, _, test_filenames = load_dataset_comprehensive(
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
# 3. HGBC HYPERPARAMETER SEARCH
# ============================================================

print('\n' + '=' * 60)
print('MODEL 1: HistGradientBoosting  (comprehensive features)')
print('=' * 60)

hgbc_results = {}

for md in HGBC_DEPTHS:
    for lr in HGBC_LRS:
        accuracies = []
        for n in HGBC_ITERS:
            print(f'\n  max_depth={md}, lr={lr}, max_iter={n}')
            model = HistGradientBoostingClassifier(
                max_iter=n, learning_rate=lr, max_depth=md,
                min_samples_leaf=5, random_state=RANDOM_STATE
            )
            model.fit(tr_feat, tr_labels)
            acc = model.score(val_feat, val_labels)
            accuracies.append(acc)
            print(f'  Validation accuracy: {acc:.4f}')
        hgbc_results[(md, lr)] = accuracies

# ============================================================
# 4. RF FOR COMPARISON
# ============================================================

print('\n' + '=' * 60)
print('MODEL 2: Random Forest 1000 trees  (comprehensive features)')
print('=' * 60)

rf_model = RandomForestClassifier(
    n_estimators=1000, max_features='sqrt',
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_model.fit(tr_feat, tr_labels)
rf_acc = rf_model.score(val_feat, val_labels)
print(f'  RF 1000 trees -> {rf_acc:.4f}')

# ============================================================
# 5. FIND BEST MODEL
# ============================================================

# print results table
print('\n--- HGBC Results Table ---')
print(f'{"depth":<7} {"lr":<6}', end='')
for n in HGBC_ITERS:
    print(f'  n={n:<8}', end='')
print()
for (md, lr) in hgbc_results:
    print(f'{md:<7} {lr:<6}', end='')
    for acc in hgbc_results[(md, lr)]:
        print(f'  {acc:<10.4f}', end='')
    print()
print(f'\nRF 1000: {rf_acc:.4f}')

# find overall best
best_acc  = rf_acc
best_type = 'RF'
best_cfg  = {'n': 1000, 'mf': 'sqrt'}

for (md, lr) in hgbc_results:
    for i, n in enumerate(HGBC_ITERS):
        if hgbc_results[(md, lr)][i] > best_acc:
            best_acc  = hgbc_results[(md, lr)][i]
            best_type = 'HGBC'
            best_cfg  = {'n': n, 'lr': lr, 'md': md}

print(f'\n*** BEST: {best_type}, val accuracy = {best_acc:.4f} ***')
print(f'    Config: {best_cfg}')

# confusion matrix for best
if best_type == 'HGBC':
    best_model = HistGradientBoostingClassifier(
        max_iter=best_cfg['n'], learning_rate=best_cfg['lr'],
        max_depth=best_cfg['md'], min_samples_leaf=5,
        random_state=RANDOM_STATE
    )
else:
    best_model = RandomForestClassifier(
        n_estimators=best_cfg['n'], max_features=best_cfg['mf'],
        random_state=RANDOM_STATE, n_jobs=-1
    )

best_model.fit(tr_feat, tr_labels)
best_preds = best_model.predict(val_feat)
cm = confusion_matrix(val_labels, best_preds)
print('\nConfusion Matrix (validation set):')
print(cm)

cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_best.txt')
np.savetxt(cm_path, cm, fmt='%d')
print(f'Confusion matrix saved -> {cm_path}')

# save results
results_path = os.path.join(RESULTS_DIR, 'results.txt')
with open(results_path, 'w') as f:
    f.write(f'v9 — Comprehensive features ({train_features.shape[1]}-dim)\n')
    f.write(f'Train: {len(tr_labels)} | Val: {len(val_labels)}\n\n')
    f.write(f'Best: {best_type} {best_cfg} -> {best_acc:.4f}\n\n')
    f.write('HGBC Results:\n')
    for (md, lr) in hgbc_results:
        for i, n in enumerate(HGBC_ITERS):
            f.write(f'  depth={md} lr={lr} n={n} -> {hgbc_results[(md, lr)][i]:.4f}\n')
    f.write(f'\nRF 1000: {rf_acc:.4f}\n')
print(f'Results saved -> {results_path}')

# ============================================================
# 6. GENERATE SUBMISSION (retrain on ALL training data)
# ============================================================

print('\n' + '=' * 60)
print('GENERATING SUBMISSION')
print('=' * 60)

if best_type == 'HGBC':
    print(f'\nRetraining HGBC (depth={best_cfg["md"]}, lr={best_cfg["lr"]}, '
          f'n={best_cfg["n"]}) on all {len(train_labels)} samples...')
    final_model = HistGradientBoostingClassifier(
        max_iter=best_cfg['n'], learning_rate=best_cfg['lr'],
        max_depth=best_cfg['md'], min_samples_leaf=5,
        random_state=RANDOM_STATE
    )
else:
    print(f'\nRetraining RF (n={best_cfg["n"]}) on all {len(train_labels)} samples...')
    final_model = RandomForestClassifier(
        n_estimators=best_cfg['n'], max_features=best_cfg['mf'],
        random_state=RANDOM_STATE, n_jobs=-1
    )

final_model.fit(train_features, train_labels)
test_predictions = final_model.predict(test_features)

print(f'Prediction distribution: {np.bincount(test_predictions)}')

submit_path = os.path.join(SUBMIT_DIR, 'v9_hgbc_comprehensive_best.csv')
with open(submit_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'label'])
    for fname, label in zip(test_filenames, test_predictions):
        writer.writerow([fname, label])

print(f'Submission saved -> {submit_path}  ({len(test_predictions)} rows)')
print('\nDone!')
