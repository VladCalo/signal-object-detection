import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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
RESULTS_DIR = 'results/v7_rf_histograms'
SUBMIT_DIR  = 'submissions'

os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUBMIT_DIR,  exist_ok=True)

# ============================================================
# PARAMETERS
# ============================================================

NUM_BINS     = 64   # histogram bins per channel -> 4 * 64 = 256 total features
RANDOM_STATE = 0

N_ESTIMATORS = [100, 200, 300]
MAX_FEATURES = ['sqrt', 'log2']

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
# 3. RANDOM FOREST HYPERPARAMETER SEARCH
# ============================================================

print('\n' + '=' * 55)
print('MODEL: Random Forest  (v7 — color histogram features)')
print('=' * 55)

rf_results = {}

for mf in MAX_FEATURES:
    accuracies = []
    for n in N_ESTIMATORS:
        print(f'\n  n_estimators={n}, max_features={mf}')
        model = RandomForestClassifier(
            n_estimators=n, max_features=mf,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(tr_feat, tr_labels)
        acc = model.score(val_feat, val_labels)
        accuracies.append(acc)
        print(f'  Validation accuracy: {acc:.4f}')
    rf_results[mf] = accuracies

# print results table
print('\n--- Results Table ---')
print(f'{"max_features":<14}', end='')
for n in N_ESTIMATORS:
    print(f'  n={n:<8}', end='')
print()
for mf in MAX_FEATURES:
    print(f'{mf:<14}', end='')
    for acc in rf_results[mf]:
        print(f'  {acc:<10.4f}', end='')
    print()

# save results
results_path = os.path.join(RESULTS_DIR, 'rf_results.txt')
with open(results_path, 'w') as f:
    f.write('RF Validation Results (features: 256-dim color histograms)\n')
    f.write(f'Train: {len(tr_labels)} samples | Val: {len(val_labels)} samples\n\n')
    f.write(f'{"max_features":<14}')
    for n in N_ESTIMATORS:
        f.write(f'  n={n:<8}')
    f.write('\n')
    for mf in MAX_FEATURES:
        f.write(f'{mf:<14}')
        for acc in rf_results[mf]:
            f.write(f'  {acc:<10.4f}')
        f.write('\n')
print(f'\nResults saved -> {results_path}')

# plot
plt.figure(figsize=(8, 5))
markers = {'sqrt': 'o', 'log2': 's'}
for mf in MAX_FEATURES:
    plt.plot(N_ESTIMATORS, rf_results[mf], marker=markers[mf], label=f'max_features={mf}')
plt.xlabel('n_estimators')
plt.ylabel('Validation Accuracy')
plt.title('RF: Accuracy vs n_estimators (color histogram features)')
plt.legend()
plt.grid(True)
plot_path = os.path.join(RESULTS_DIR, 'rf_accuracy_vs_n.png')
plt.savefig(plot_path)
plt.close()
print(f'Plot saved    -> {plot_path}')

# find best
best_acc = 0
best_n   = N_ESTIMATORS[0]
best_mf  = MAX_FEATURES[0]
for mf in MAX_FEATURES:
    for i, n in enumerate(N_ESTIMATORS):
        if rf_results[mf][i] > best_acc:
            best_acc = rf_results[mf][i]
            best_n   = n
            best_mf  = mf

print(f'\nBest RF: n_estimators={best_n}, max_features={best_mf}, val accuracy={best_acc:.4f}')

best_model = RandomForestClassifier(
    n_estimators=best_n, max_features=best_mf,
    random_state=RANDOM_STATE, n_jobs=-1
)
best_model.fit(tr_feat, tr_labels)
best_preds = best_model.predict(val_feat)
cm = confusion_matrix(val_labels, best_preds)
print('\nConfusion Matrix (validation set):')
print(cm)

cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_rf.txt')
np.savetxt(cm_path, cm, fmt='%d')
print(f'Confusion matrix saved -> {cm_path}')

# ============================================================
# 4. GENERATE SUBMISSION
# ============================================================

print('\n' + '=' * 55)
print('GENERATING SUBMISSION')
print('=' * 55)

print(f'\nRetraining RF (n={best_n}, max_features={best_mf}) on all {len(train_labels)} training samples...')
final_model = RandomForestClassifier(
    n_estimators=best_n, max_features=best_mf,
    random_state=RANDOM_STATE, n_jobs=-1
)
final_model.fit(train_features, train_labels)
test_predictions = final_model.predict(test_features)

print(f'Prediction distribution: {np.bincount(test_predictions)}')

submit_path = os.path.join(SUBMIT_DIR, 'v7_rf_hist_best.csv')
with open(submit_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'label'])
    for fname, label in zip(test_filenames, test_predictions):
        writer.writerow([fname, label])

print(f'Submission saved -> {submit_path}  ({len(test_predictions)} rows)')
print('\nDone!')
