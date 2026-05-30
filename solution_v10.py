import numpy as np
import os
import csv
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import shuffle

from feature_extraction import load_dataset_v10
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
RESULTS_DIR = 'results/v10_rcf'
SUBMIT_DIR  = 'submissions'

os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUBMIT_DIR,  exist_ok=True)

# ============================================================
# PARAMETERS
# ============================================================

RANDOM_STATE = 0

# ============================================================
# 1. LOAD DATA  (with caching)
# ============================================================

train_feat_cache  = os.path.join(CACHE_DIR, 'train_v10_features.npy')
train_label_cache = os.path.join(CACHE_DIR, 'train_v10_labels.npy')
train_fname_cache = os.path.join(CACHE_DIR, 'train_v10_filenames.txt')
test_feat_cache   = os.path.join(CACHE_DIR, 'test_v10_features.npy')
test_fname_cache  = os.path.join(CACHE_DIR, 'test_v10_filenames.txt')

print('=' * 60)
print('v10 — Comprehensive + Random Convolutional Features')
print('     (1366 comprehensive + 450 RCF = 1816 total)')
print('=' * 60)

print('\nLoading training data...')
if os.path.exists(train_feat_cache):
    print('  (loading from cache)')
    train_features = np.load(train_feat_cache)
    train_labels   = np.load(train_label_cache)
    with open(train_fname_cache) as f:
        train_filenames = f.read().splitlines()
else:
    train_features, train_labels, train_filenames = load_dataset_v10(
        TRAIN_DIR, TRAIN_CSV
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
    test_features, _, test_filenames = load_dataset_v10(
        TEST_DIR, TEST_CSV
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
print('HGBC on v10 features (comprehensive + RCF)')
print('=' * 60)

results = {}

for md in [7, 10, 12]:
    for lr in [0.02, 0.03, 0.05]:
        print(f'\n  max_depth={md}, lr={lr}, max_iter=500')
        model = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=lr, max_depth=md,
            min_samples_leaf=5, random_state=RANDOM_STATE
        )
        model.fit(tr_feat, tr_labels)
        acc = model.score(val_feat, val_labels)
        results[(md, lr)] = acc
        print(f'  Validation accuracy: {acc:.4f}')

# also test RF
print(f'\n  RF 1000 trees')
rf = RandomForestClassifier(
    n_estimators=1000, max_features='sqrt',
    random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(tr_feat, tr_labels)
rf_acc = rf.score(val_feat, val_labels)
print(f'  Validation accuracy: {rf_acc:.4f}')

# ============================================================
# 4. RESULTS SUMMARY
# ============================================================

print('\n--- Results Table ---')
print(f'{"depth":<7} {"lr":<6} {"accuracy":<10}')
for (md, lr), acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f'{md:<7} {lr:<6} {acc:<10.4f}')
print(f'{"RF":7s} {"1000":6s} {rf_acc:<10.4f}')

# find best
best_acc  = rf_acc
best_type = 'RF'
best_cfg  = {'n': 1000, 'mf': 'sqrt'}

for (md, lr), acc in results.items():
    if acc > best_acc:
        best_acc  = acc
        best_type = 'HGBC'
        best_cfg  = {'n': 500, 'lr': lr, 'md': md}

print(f'\n*** BEST: {best_type}, val accuracy = {best_acc:.4f} ***')
print(f'    Config: {best_cfg}')

# confusion matrix
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

# save results
results_path = os.path.join(RESULTS_DIR, 'results.txt')
with open(results_path, 'w') as f:
    f.write(f'v10 — Comprehensive + RCF features ({train_features.shape[1]}-dim)\n')
    f.write(f'Train: {len(tr_labels)} | Val: {len(val_labels)}\n\n')
    f.write(f'Best: {best_type} {best_cfg} -> {best_acc:.4f}\n\n')
    for (md, lr), acc in sorted(results.items(), key=lambda x: -x[1]):
        f.write(f'  HGBC d={md} lr={lr} -> {acc:.4f}\n')
    f.write(f'  RF 1000 -> {rf_acc:.4f}\n')
print(f'Results saved -> {results_path}')

# ============================================================
# 5. GENERATE SUBMISSION
# ============================================================

print('\n' + '=' * 60)
print('GENERATING SUBMISSION')
print('=' * 60)

if best_type == 'HGBC':
    print(f'\nRetraining HGBC (depth={best_cfg["md"]}, lr={best_cfg["lr"]}) '
          f'on all {len(train_labels)} samples...')
    final_model = HistGradientBoostingClassifier(
        max_iter=best_cfg['n'], learning_rate=best_cfg['lr'],
        max_depth=best_cfg['md'], min_samples_leaf=5,
        random_state=RANDOM_STATE
    )
else:
    print(f'\nRetraining RF on all {len(train_labels)} samples...')
    final_model = RandomForestClassifier(
        n_estimators=best_cfg['n'], max_features=best_cfg['mf'],
        random_state=RANDOM_STATE, n_jobs=-1
    )

final_model.fit(train_features, train_labels)
test_predictions = final_model.predict(test_features)

print(f'Prediction distribution: {np.bincount(test_predictions)}')

submit_path = os.path.join(SUBMIT_DIR, 'v10_rcf_best.csv')
with open(submit_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'label'])
    for fname, lbl in zip(test_filenames, test_predictions):
        writer.writerow([fname, lbl])

print(f'Submission saved -> {submit_path}  ({len(test_predictions)} rows)')
print('\nDone!')
