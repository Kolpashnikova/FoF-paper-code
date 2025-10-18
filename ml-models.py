
import os
import glob
import pandas as pd
import numpy as np
import gc
import logging
import locale
import json
import uuid
import datetime
from collections import defaultdict

from scipy.stats import ttest_ind
from scipy.signal import butter, filtfilt
from scipy.special import expit
import statsmodels.api as sm
from statsmodels.formula.api import ols

import cudf
import cupy as cp
from cuml.preprocessing import StandardScaler, LabelEncoder
from cuml.model_selection import train_test_split
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.neighbors import KNeighborsClassifier as cuKNN, KNeighborsClassifier
from cuml.linear_model import LogisticRegression as cuLR, LogisticRegression
from cuml.svm import SVC as cuSVM, LinearSVC as cuLinearSVM

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, auc, f1_score, precision_score, 
    recall_score, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from catboost import CatBoostClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import optuna
import joblib

locale.getpreferredencoding = lambda: "UTF-8"

df_clean = cudf.read_parquet("drive/MyDrive/Colab Notebooks/data/model_data_final.parquet")

def clear_memory():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    if hasattr(cp.cuda, 'Device'):
        cp.cuda.Device().synchronize()

X = df_clean[['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
              'filtered_eda_values', 'filtered_temperature_values',
              'filtered_steps_values', 'filtered_azimuth',
              'filtered_duration_fixation', 'filtered_duration_blink']]
y = df_clean['Potential_FoF']

X = X.astype('float32')

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


clear_memory()

y_train_cpu = y_train.values_host
neg_count = (y_train_cpu == 0).sum()
pos_count = (y_train_cpu == 1).sum()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

del X, X_train, X_test
clear_memory()

X_train_cpu = X_train_scaled.values.get().astype(np.float32)
X_test_cpu = X_test_scaled.values.get().astype(np.float32)
y_train_cpu = y_train.values.get().astype(np.int32)
y_test_cpu = y_test.values.get().astype(np.int32)

del X_train_scaled, X_test_scaled, y_train, y_test
clear_memory()

catboost_clf = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    verbose=False,
    task_type='GPU' if cp.cuda.is_available() else 'CPU',
    devices='0' if cp.cuda.is_available() else None
)

catboost_clf.fit(X_train_cpu, y_train_cpu)

y_pred = catboost_clf.predict(X_test_cpu)
y_pred_proba = catboost_clf.predict_proba(X_test_cpu)[:, 1]

accuracy = accuracy_score(y_test_cpu, y_pred)

del catboost_clf
clear_memory()

def objective(trial):
    iterations = trial.suggest_categorical('iterations', [50, 100, 200, 300])
    depth = trial.suggest_categorical('depth', [4, 6, 8, 10])
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2])
    l2_leaf_reg = trial.suggest_categorical('l2_leaf_reg', [1, 3, 5, 7, 9])
    border_count = trial.suggest_categorical('border_count', [32, 64, 128])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_cpu, y_train_cpu)):
        try:
            X_tr_fold = X_train_cpu[train_idx]
            X_val_fold = X_train_cpu[val_idx]
            y_tr_fold = y_train_cpu[train_idx]
            y_val_fold = y_train_cpu[val_idx]

            fold_catboost = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                border_count=border_count,
                random_seed=42,
                verbose=False,
                task_type='GPU' if cp.cuda.is_available() else 'CPU',
                devices='0' if cp.cuda.is_available() else None
            )

            fold_catboost.fit(X_tr_fold, y_tr_fold)

            val_proba = fold_catboost.predict_proba(X_val_fold)[:, 1]

            auc = roc_auc_score(y_val_fold, val_proba)
            auc_scores.append(auc)

            del fold_catboost, val_proba
            del X_tr_fold, X_val_fold
            clear_memory()

        except Exception as e:
            try:
                del fold_catboost
            except:
                pass
            clear_memory()
            return 0.0

    if len(auc_scores) == 0:
        return 0.0

    return np.mean(auc_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

clear_memory()

best_params = study.best_trial.params
final_catboost_clf = CatBoostClassifier(
    iterations=best_params['iterations'],
    depth=best_params['depth'],
    learning_rate=best_params['learning_rate'],
    l2_leaf_reg=best_params['l2_leaf_reg'],
    border_count=best_params['border_count'],
    random_seed=42,
    verbose=False,
    task_type='GPU' if cp.cuda.is_available() else 'CPU',
    devices='0' if cp.cuda.is_available() else None
)

final_catboost_clf.fit(X_train_cpu, y_train_cpu)

final_proba = final_catboost_clf.predict_proba(X_test_cpu)[:, 1]

final_auc = roc_auc_score(y_test_cpu, final_proba)

precision, recall, thresholds = precision_recall_curve(y_test_cpu, final_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]

y_pred_best = (final_proba >= optimal_threshold).astype(int)
accuracy_best = accuracy_score(y_test_cpu, y_pred_best)

feature_names = ['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
              'filtered_eda_values', 'filtered_temperature_values',
              'filtered_steps_values', 'filtered_azimuth',
              'filtered_duration_fixation', 'filtered_duration_blink']

feature_importance = final_catboost_clf.get_feature_importance()



FEATURES = [
    'filtered_acc_x','filtered_acc_y','filtered_acc_z',
    'filtered_eda_values','filtered_temperature_values',
    'filtered_steps_values','filtered_azimuth',
    'filtered_duration_fixation','filtered_duration_blink'
]
TARGET = 'Potential_FoF'

X = df_clean[FEATURES].astype('float32')
y = df_clean[TARGET]
ids = df_clean['id'] if 'id' in df_clean.columns else df_clean.index.to_series()

le = LabelEncoder()
y_enc = le.fit_transform(y)

y_np = y_enc.to_pandas().values
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
(train_idx, test_idx), = sss.split(np.zeros_like(y_np), y_np)

X_tr = X.iloc[train_idx]; X_te = X.iloc[test_idx]
y_tr = y_enc.iloc[train_idx]; y_te = y_enc.iloc[test_idx]
ids_te = ids.iloc[test_idx] if hasattr(ids, "iloc") else ids.take(test_idx)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

def to_cpu_np(gobj, dtype=None):
    arr = gobj.values if hasattr(gobj, "values") else gobj
    arr = arr.get() if hasattr(arr, "get") else arr
    return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)

X_train_cpu = to_cpu_np(X_tr_sc, np.float32)
X_test_cpu  = to_cpu_np(X_te_sc,  np.float32)
y_train_cpu = to_cpu_np(y_tr,     np.int32)
y_test_cpu  = to_cpu_np(y_te,     np.int32)
ids_test    = np.asarray(ids_te.to_pandas() if hasattr(ids_te, "to_pandas") else ids_te)

best_params = dict(
    iterations=300,
    depth=10,
    learning_rate=0.2,
    l2_leaf_reg=1,
    border_count=32,
    random_seed=42,
    verbose=False,
    task_type='GPU' if cp.cuda.is_available() else 'CPU',
    devices='0' if cp.cuda.is_available() else None
)
model = CatBoostClassifier(**best_params)
model.fit(X_train_cpu, y_train_cpu)

proba = model.predict_proba(X_test_cpu)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_cpu, proba, pos_label=1)
roc_auc = auc(fpr, tpr)

best_thresh = 0.3131
y_pred = (proba >= best_thresh).astype(int)

accuracy = accuracy_score(y_test_cpu, y_pred)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test_cpu, y_pred, average="macro", zero_division=0
)
f1_minority = f1_score(y_test_cpu, y_pred, pos_label=1, zero_division=0)

save_dir = "drive/MyDrive/Colab Notebooks/data"
os.makedirs(save_dir, exist_ok=True)

np.savez(os.path.join(save_dir, "roc_data_catboost_best.npz"),
         fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc,
         accuracy=accuracy,
         precision_macro=prec_macro,
         f1_macro=f1_macro,
         f1_minority=f1_minority,
         threshold_used=best_thresh,
         ids=ids_test)


model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "catboost_fof_2.cbm")
final_catboost_clf.save_model(model_path)

scaler_path = os.path.join(model_dir, "scaler_catboost_2.joblib")
joblib.dump(scaler, scaler_path)

le_path = os.path.join(model_dir, "label_encoder_2.joblib")
joblib.dump(le, le_path)


fpr, tpr, _ = roc_curve(y_test_cpu, final_proba)
roc_auc = auc(fpr, tpr)


"""# RF """


def clear_memory():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    if hasattr(cp.cuda, 'Device'):
        cp.cuda.Device().synchronize()

X = df_clean[['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
              'filtered_eda_values', 'filtered_temperature_values',
              'filtered_steps_values', 'filtered_azimuth',
              'filtered_duration_fixation', 'filtered_duration_blink']]
y = df_clean['Potential_FoF']

X = X.astype('float32')

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

clear_memory()

y_train_cpu = y_train.values_host
neg_count = (y_train_cpu == 0).sum()
pos_count = (y_train_cpu == 1).sum()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

del X, X_train, X_test
clear_memory()

rf_clf = cuRF(
    n_estimators=100,
    max_depth=10,
    max_features=1.0,
    n_streams=1,
    random_state=42
)

rf_clf.fit(X_train_scaled, y_train)

y_pred = rf_clf.predict(X_test_scaled)
y_pred_cpu = y_pred.to_pandas().values
y_test_cpu = y_test.to_pandas().values

del y_pred
clear_memory()

accuracy = accuracy_score(y_test_cpu, y_pred_cpu)

del rf_clf
clear_memory()

X_train_np = X_train_scaled.values.get().astype(np.float32)
y_train_np = y_train.values.get().astype(np.int32)

del X_train_scaled, y_train
clear_memory()

def objective(trial):
    n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 150, 200])
    max_depth = trial.suggest_categorical('max_depth', [5, 10, 15, 20]) 
    max_features = trial.suggest_categorical('max_features', [0.5, 0.7, 0.8, 1.0])
    min_samples_split = trial.suggest_categorical('min_samples_split', [2, 5])
    min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [1, 2])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        try:
            X_tr_fold = X_train_np[train_idx].astype(np.float32)
            X_val_fold = X_train_np[val_idx].astype(np.float32)
            y_tr_fold = y_train_np[train_idx]
            y_val_fold = y_train_np[val_idx]

            X_tr_cudf = cudf.DataFrame(X_tr_fold)
            y_tr_cudf = cudf.Series(y_tr_fold)
            X_val_cudf = cudf.DataFrame(X_val_fold)

            fold_rf = cuRF(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_streams=1,
                random_state=42
            )

            fold_rf.fit(X_tr_cudf, y_tr_cudf)

            val_proba = fold_rf.predict_proba(X_val_cudf)
            val_proba_cpu = val_proba.to_pandas().values[:, 1]

            auc = roc_auc_score(y_val_fold, val_proba_cpu)
            auc_scores.append(auc)

            del fold_rf, val_proba, X_tr_cudf, y_tr_cudf, X_val_cudf
            del X_tr_fold, X_val_fold, val_proba_cpu
            clear_memory()

        except Exception as e:
            try:
                del fold_rf, X_tr_cudf, y_tr_cudf, X_val_cudf
            except:
                pass
            clear_memory()
            return 0.0

    if len(auc_scores) == 0:
        return 0.0

    return np.mean(auc_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)


clear_memory()

best_params = study.best_trial.params
final_rf_clf = cuRF(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    max_features=best_params['max_features'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    n_streams=1,
    random_state=42
)

X_train_final = cudf.DataFrame(X_train_np)
y_train_final = cudf.Series(y_train_np)

final_rf_clf.fit(X_train_final, y_train_final)

final_proba = final_rf_clf.predict_proba(X_test_scaled)
final_proba_cpu = final_proba.to_pandas().values

del final_proba, X_train_final, y_train_final
clear_memory()

final_auc = roc_auc_score(y_test_cpu, final_proba_cpu[:, 1])

precision, recall, thresholds = precision_recall_curve(y_test_cpu, final_proba_cpu[:, 1])
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]

y_pred_best = (final_proba_cpu[:, 1] >= optimal_threshold).astype(int)
accuracy_best = accuracy_score(y_test_cpu, y_pred_best)



FEATURES = [
    'filtered_acc_x','filtered_acc_y','filtered_acc_z',
    'filtered_eda_values','filtered_temperature_values',
    'filtered_steps_values','filtered_azimuth',
    'filtered_duration_fixation','filtered_duration_blink'
]
TARGET = 'Potential_FoF'

X = df_clean[FEATURES].astype('float32')
y = df_clean[TARGET]
ids = (df_clean['id'] if 'id' in df_clean.columns else df_clean.index.to_series())

le = LabelEncoder()
y_enc = le.fit_transform(y)
y_np = y_enc.to_pandas().values

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
(train_idx, test_idx), = sss.split(np.zeros_like(y_np), y_np)

X_tr = X.iloc[train_idx]; X_te = X.iloc[test_idx]
y_tr = y_enc.iloc[train_idx]; y_te = y_enc.iloc[test_idx]
ids_te = ids.iloc[test_idx] if hasattr(ids, "iloc") else ids.take(test_idx)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

rf = cuRF(
    n_estimators=150,
    max_depth=20,
    max_features=0.8,
    min_samples_split=2,
    min_samples_leaf=1,
    n_streams=1,
    random_state=42
)
rf.fit(X_tr_sc, y_tr)

proba = rf.predict_proba(X_te_sc).to_pandas().values[:, 1]
y_true = y_te.to_pandas().values

fpr, tpr, thr = roc_curve(y_true, proba, pos_label=1)
roc_auc = auc(fpr, tpr)

best_thresh = 0.1249
y_pred = (proba >= best_thresh).astype(int)

acc = accuracy_score(y_true, y_pred)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
f1_minority = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

np.savez("drive/MyDrive/Colab Notebooks/data/roc_data_rf_best.npz",
         fpr=fpr, tpr=tpr, thresholds=thr, auc=roc_auc,
         accuracy=acc,
         precision_macro=prec_macro,
         f1_macro=f1_macro,
         f1_minority=f1_minority,
         threshold_used=best_thresh,
         ids=np.asarray(ids_te.to_pandas() if hasattr(ids_te, "to_pandas") else ids_te))

d = np.load("drive/MyDrive/Colab Notebooks/data/roc_data_rf_best.npz")


d = np.load("drive/MyDrive/Colab Notebooks/data/roc_data_rf_best.npz")
fpr, tpr, auc_val = d["fpr"], d["tpr"], float(d["auc"])


model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

rf_path = os.path.join(model_dir, "rf_model.pkl")
joblib.dump(final_rf_clf, rf_path)

scaler_path = os.path.join(model_dir, "scaler_rf.joblib")
joblib.dump(scaler, scaler_path)

le_path = os.path.join(model_dir, "label_encoder_rf.joblib")
joblib.dump(le, le_path)


fpr, tpr, _ = roc_curve(y_test_cpu, final_proba_cpu[:, 1])
roc_auc = auc(fpr, tpr)

"""# LR"""


def clear_memory():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    if hasattr(cp.cuda, 'Device'):
        cp.cuda.Device().synchronize()

X = df_clean[['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
              'filtered_eda_values', 'filtered_temperature_values',
              'filtered_steps_values', 'filtered_azimuth',
              'filtered_duration_fixation', 'filtered_duration_blink']]
y = df_clean['Potential_FoF']

X = X.astype('float32')

le = LabelEncoder()
y_encoded = le.fit_transform(y)

y_cpu = y_encoded.values_host
class_weights = compute_class_weight('balanced', classes=np.unique(y_cpu), y=y_cpu)
class_weight_dict = dict(zip(np.unique(y_cpu), class_weights))

 with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

clear_memory()

y_train_cpu = y_train.values_host
neg_count = (y_train_cpu == 0).sum()
pos_count = (y_train_cpu == 1).sum()
imbalance_ratio = neg_count / pos_count

 (very important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

del X, X_train, X_test
clear_memory()

lr_clf = cuLR(
    C=1.0,
    penalty='l2',
    solver='qn',
    max_iter=1000,
    tol=1e-4,
    fit_intercept=True,
    class_weight='balanced'
)

lr_clf.fit(X_train_scaled, y_train)

y_proba = lr_clf.predict_proba(X_test_scaled)
y_proba_cpu = y_proba.to_pandas().values

precision, recall, thresholds = precision_recall_curve(y_test.to_pandas().values, y_proba_cpu[:, 1])
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_threshold = thresholds[np.argmax(f1_scores)]

y_pred_optimal = (y_proba_cpu[:, 1] >= optimal_threshold).astype(int)
y_test_cpu = y_test.to_pandas().values

accuracy = accuracy_score(y_test_cpu, y_pred_optimal)
auc = roc_auc_score(y_test_cpu, y_proba_cpu[:, 1])

del lr_clf, y_proba
clear_memory()

X_train_np = X_train_scaled.values.get().astype(np.float32)
y_train_np = y_train.values.get().astype(np.int32)

del X_train_scaled, y_train
clear_memory()

def objective(trial):
    C = trial.suggest_float('C', 0.001, 100.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l2', 'elasticnet'])
    solver = 'qn'
    max_iter = trial.suggest_categorical('max_iter', [1000, 2000, 3000])
    tol = trial.suggest_float('tol', 1e-6, 1e-3, log=True)

    class_weight_strategy = trial.suggest_categorical('class_weight', ['balanced', 'custom'])

    if class_weight_strategy == 'custom':
        minority_weight = trial.suggest_float('minority_weight', 1.0, 20.0)
        class_weight_custom = {0: 1.0, 1: minority_weight}
    else:
        class_weight_custom = 'balanced'

    if penalty == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
    else:
        l1_ratio = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        try:
            X_tr_fold = X_train_np[train_idx].astype(np.float32)
            X_val_fold = X_train_np[val_idx].astype(np.float32)
            y_tr_fold = y_train_np[train_idx]
            y_val_fold = y_train_np[val_idx]

            X_tr_cudf = cudf.DataFrame(X_tr_fold)
            y_tr_cudf = cudf.Series(y_tr_fold)
            X_val_cudf = cudf.DataFrame(X_val_fold)

            fold_lr = cuLR(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=max_iter,
                tol=tol,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                class_weight=class_weight_custom
            )

            fold_lr.fit(X_tr_cudf, y_tr_cudf)

            val_proba = fold_lr.predict_proba(X_val_cudf)
            val_proba_cpu = val_proba.to_pandas().values[:, 1]

            fold_precision, fold_recall, fold_thresholds = precision_recall_curve(y_val_fold, val_proba_cpu)
            fold_f1_scores = 2 * (fold_precision * fold_recall) / (fold_precision + fold_recall + 1e-8)
            fold_optimal_threshold = fold_thresholds[np.argmax(fold_f1_scores)]

            val_pred = (val_proba_cpu >= fold_optimal_threshold).astype(int)

            f1 = f1_score(y_val_fold, val_pred)
            f1_scores.append(f1)

            del fold_lr, val_proba, X_tr_cudf, y_tr_cudf, X_val_cudf
            del X_tr_fold, X_val_fold, val_proba_cpu
            clear_memory()

        except Exception as e:
            try:
                del fold_lr, X_tr_cudf, y_tr_cudf, X_val_cudf
            except:
                pass
            clear_memory()
            return 0.0

    if len(f1_scores) == 0:
        return 0.0

    return np.mean(f1_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

clear_memory()

best_params = study.best_trial.params

if best_params['class_weight'] == 'custom':
    final_class_weight = {0: 1.0, 1: best_params['minority_weight']}
else:
    final_class_weight = 'balanced'

final_lr_clf = cuLR(
    C=best_params['C'],
    penalty=best_params['penalty'],
    solver='qn',
    max_iter=best_params['max_iter'],
    tol=best_params['tol'],
    l1_ratio=best_params.get('l1_ratio', None),
    fit_intercept=True,
    class_weight=final_class_weight
)

X_train_final = cudf.DataFrame(X_train_np)
y_train_final = cudf.Series(y_train_np)

final_lr_clf.fit(X_train_final, y_train_final)

 with multiple metrics
final_proba = final_lr_clf.predict_proba(X_test_scaled)
final_proba_cpu = final_proba.to_pandas().values

final_precision, final_recall, final_thresholds = precision_recall_curve(y_test_cpu, final_proba_cpu[:, 1])
final_f1_scores = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-8)
final_optimal_threshold = final_thresholds[np.argmax(final_f1_scores)]

 with optimal threshold
y_pred_final = (final_proba_cpu[:, 1] >= final_optimal_threshold).astype(int)

final_accuracy = accuracy_score(y_test_cpu, y_pred_final)
final_auc = roc_auc_score(y_test_cpu, final_proba_cpu[:, 1])
final_f1 = f1_score(y_test_cpu, y_pred_final)
final_precision = precision_score(y_test_cpu, y_pred_final)
final_recall = recall_score(y_test_cpu, y_pred_final)

cm = confusion_matrix(y_test_cpu, y_pred_final)

feature_names = ['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
                'filtered_eda_values', 'filtered_temperature_values',
                'filtered_steps_values', 'filtered_azimuth',
                'filtered_duration_fixation', 'filtered_duration_blink']

coefficients = final_lr_clf.coef_.to_pandas().values[0]
feature_importance = list(zip(feature_names, coefficients, np.abs(coefficients)))
feature_importance.sort(key=lambda x: x[2], reverse=True)

for name, coef, abs_coef in feature_importance:
    direction = "↑ positive" if coef > 0 else "↓ negative"

thresholds_to_test = [0.3, 0.4, 0.5, final_optimal_threshold, 0.6, 0.7]
for thresh in thresholds_to_test:
    pred_thresh = (final_proba_cpu[:, 1] >= thresh).astype(int)
    acc_thresh = accuracy_score(y_test_cpu, pred_thresh)
    f1_thresh = f1_score(y_test_cpu, pred_thresh)
    precision_thresh = precision_score(y_test_cpu, pred_thresh, zero_division=0)
    recall_thresh = recall_score(y_test_cpu, pred_thresh)
    marker = " ← OPTIMAL" if abs(thresh - final_optimal_threshold) < 0.01 else ""


FEATURES = [
    'filtered_acc_x','filtered_acc_y','filtered_acc_z',
    'filtered_eda_values','filtered_temperature_values',
    'filtered_steps_values','filtered_azimuth',
    'filtered_duration_fixation','filtered_duration_blink'
]
TARGET = 'Potential_FoF'
SAVE_DIR = "drive/MyDrive/Colab Notebooks/data"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_THRESH_OVERRIDE = None

X = df_clean[FEATURES].astype('float32')
y = df_clean[TARGET]
ids = df_clean['id'] if 'id' in df_clean.columns else df_clean.index.to_series()

 on GPU
le = LabelEncoder()
y_enc = le.fit_transform(y)

 (seed=42)
y_np = y_enc.to_pandas().values
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
(train_idx, test_idx), = sss.split(np.zeros_like(y_np), y_np)

X_tr = X.iloc[train_idx]; X_te = X.iloc[test_idx]
y_tr = y_enc.iloc[train_idx]; y_te = y_enc.iloc[test_idx]
ids_te = ids.iloc[test_idx] if hasattr(ids, "iloc") else ids.take(test_idx)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)


minority_weight = 4.72696510922015
class_weight_dict = {0: 1.0, 1: float(minority_weight)}

lr = cuLR(
    C=0.0020408142278042275,
    penalty='l2',
    solver='qn',
    max_iter=2000,
    tol=7.470393304994062e-06,
    fit_intercept=True,
    class_weight=class_weight_dict
)
lr.fit(X_tr_sc, y_tr)

proba = lr.predict_proba(X_te_sc).to_pandas().values[:, 1]
y_true = y_te.to_pandas().values
ids_test = np.asarray(ids_te.to_pandas() if hasattr(ids_te, "to_pandas") else ids_te)

fpr, tpr, thresholds = roc_curve(y_true, proba, pos_label=1)
roc_auc = auc(fpr, tpr)

best_thresh = 0.3628

y_pred = (proba >= best_thresh).astype(int)

acc = accuracy_score(y_true, y_pred)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
f1_minority = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

np.savez(os.path.join(SAVE_DIR, "roc_data_logreg_best.npz"),
         fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc,
         accuracy=acc,
         precision_macro=prec_macro,
         f1_macro=f1_macro,
         f1_minority=f1_minority,
         threshold_used=best_thresh,
         ids=ids_test)


model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

lr_path = os.path.join(model_dir, "lr_model.pkl")
joblib.dump(final_lr_clf, lr_path)

scaler_path = os.path.join(model_dir, "scaler_lr.joblib")
joblib.dump(scaler, scaler_path)
le_path = os.path.join(model_dir, "label_encoder_lr.joblib")
joblib.dump(le, le_path)

probs = final_proba_cpu[:, 1]

fpr, tpr, _ = roc_curve(y_test_cpu, probs)
roc_auc = auc(fpr, tpr)

"""# KNN"""


def clear_memory():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    if hasattr(cp.cuda, 'Device'):
        cp.cuda.Device().synchronize()

X = df_clean[['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
              'filtered_eda_values', 'filtered_temperature_values',
              'filtered_steps_values', 'filtered_azimuth',
              'filtered_duration_fixation', 'filtered_duration_blink']]
y = df_clean['Potential_FoF']

X = X.astype('float32')

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

clear_memory()

y_train_cpu = y_train.values_host
neg_count = (y_train_cpu == 0).sum()
pos_count = (y_train_cpu == 1).sum()

 (CRITICAL for KNN - all features must be on same scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

del X, X_train, X_test
clear_memory()

knn_clf = cuKNN(
    n_neighbors=5,
    weights='uniform',
    metric='euclidean'
)

knn_clf.fit(X_train_scaled, y_train)

y_pred = knn_clf.predict(X_test_scaled)
y_pred_cpu = y_pred.to_pandas().values
y_test_cpu = y_test.to_pandas().values

y_pred_proba = knn_clf.predict_proba(X_test_scaled)
y_pred_proba_cpu = y_pred_proba.to_pandas().values[:, 1]

del y_pred, y_pred_proba
clear_memory()

accuracy = accuracy_score(y_test_cpu, y_pred_cpu)

del knn_clf
clear_memory()

X_train_np = X_train_scaled.values.get().astype(np.float32)
y_train_np = y_train.values.get().astype(np.int32)

del X_train_scaled, y_train
clear_memory()

def objective(trial):
    n_neighbors = trial.suggest_categorical('n_neighbors', [3, 5, 7, 9, 11, 15, 21])
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])

    if metric == 'minkowski':
        p = trial.suggest_categorical('p', [1, 2, 3])
    else:
        p = 2

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        try:
            X_tr_fold = X_train_np[train_idx].astype(np.float32)
            X_val_fold = X_train_np[val_idx].astype(np.float32)
            y_tr_fold = y_train_np[train_idx]
            y_val_fold = y_train_np[val_idx]

            X_tr_cudf = cudf.DataFrame(X_tr_fold)
            y_tr_cudf = cudf.Series(y_tr_fold)
            X_val_cudf = cudf.DataFrame(X_val_fold)

            fold_knn = cuKNN(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
                p=p if metric == 'minkowski' else 2
            )

            fold_knn.fit(X_tr_cudf, y_tr_cudf)

            val_proba = fold_knn.predict_proba(X_val_cudf)
            val_proba_cpu = val_proba.to_pandas().values[:, 1]

            auc = roc_auc_score(y_val_fold, val_proba_cpu)
            auc_scores.append(auc)

            del fold_knn, val_proba, X_tr_cudf, y_tr_cudf, X_val_cudf
            del X_tr_fold, X_val_fold, val_proba_cpu
            clear_memory()

        except Exception as e:
            try:
                del fold_knn, X_tr_cudf, y_tr_cudf, X_val_cudf
            except:
                pass
            clear_memory()
            return 0.0

    if len(auc_scores) == 0:
        return 0.0

    return np.mean(auc_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, show_progress_bar=True)

clear_memory()

best_params = study.best_trial.params
final_knn_clf = cuKNN(
    n_neighbors=best_params['n_neighbors'],
    weights=best_params['weights'],
    metric=best_params['metric'],
    p=best_params.get('p', 2)
)

X_train_final = cudf.DataFrame(X_train_np)
y_train_final = cudf.Series(y_train_np)

final_knn_clf.fit(X_train_final, y_train_final)

final_proba = final_knn_clf.predict_proba(X_test_scaled)
final_proba_cpu = final_proba.to_pandas().values

del final_proba, X_train_final, y_train_final
clear_memory()

final_auc = roc_auc_score(y_test_cpu, final_proba_cpu[:, 1])

precision, recall, thresholds = precision_recall_curve(y_test_cpu, final_proba_cpu[:, 1])
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]

y_pred_best = (final_proba_cpu[:, 1] >= optimal_threshold).astype(int)
accuracy_best = accuracy_score(y_test_cpu, y_pred_best)

feature_names = ['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
                'filtered_eda_values', 'filtered_temperature_values',
                'filtered_steps_values', 'filtered_azimuth',
                'filtered_duration_fixation', 'filtered_duration_blink']


FEATURES = [
    'filtered_acc_x','filtered_acc_y','filtered_acc_z',
    'filtered_eda_values','filtered_temperature_values',
    'filtered_steps_values','filtered_azimuth',
    'filtered_duration_fixation','filtered_duration_blink'
]
TARGET = 'Potential_FoF'
SAVE_DIR = "drive/MyDrive/Colab Notebooks/data"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_THRESH_OVERRIDE = None

X = df_clean[FEATURES].astype('float32')
y = df_clean[TARGET]
ids = df_clean['id'] if 'id' in df_clean.columns else df_clean.index.to_series()

 on GPU
le = LabelEncoder()
y_enc = le.fit_transform(y)

y_np = y_enc.to_pandas().values
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
(train_idx, test_idx), = sss.split(np.zeros_like(y_np), y_np)

X_tr = X.iloc[train_idx]; X_te = X.iloc[test_idx]
y_tr = y_enc.iloc[train_idx]; y_te = y_enc.iloc[test_idx]
ids_te = ids.iloc[test_idx] if hasattr(ids, "iloc") else ids.take(test_idx)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

best_params = {'n_neighbors': 7, 'weights': 'uniform', 'metric': 'euclidean'}

knn = cuKNN(
    n_neighbors=best_params['n_neighbors'],
    weights=best_params['weights'],
    metric=best_params['metric']
)
knn.fit(X_tr_sc, y_tr)

proba = knn.predict_proba(X_te_sc).to_pandas().values[:, 1]
y_true = y_te.to_pandas().values
ids_test = np.asarray(ids_te.to_pandas() if hasattr(ids_te, "to_pandas") else ids_te)

fpr, tpr, thresholds = roc_curve(y_true, proba, pos_label=1)
roc_auc = auc(fpr, tpr)

best_thresh = 0.5714

y_pred = (proba >= best_thresh).astype(int)

acc = accuracy_score(y_true, y_pred)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
f1_minority = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

np.savez(os.path.join(SAVE_DIR, "roc_data_knn_best.npz"),
         fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc,
         accuracy=acc,
         precision_macro=prec_macro,
         f1_macro=f1_macro,
         f1_minority=f1_minority,
         threshold_used=best_thresh,
         ids=ids_test)


model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

knn_path = os.path.join(model_dir, "knn_model.pkl")
joblib.dump(final_knn_clf, knn_path)

scaler_path = os.path.join(model_dir, "scaler_knn.joblib")
joblib.dump(scaler, scaler_path)

le_path = os.path.join(model_dir, "label_encoder_knn.joblib")
joblib.dump(le, le_path)

probs = final_proba_cpu[:, 1]

fpr, tpr, _ = roc_curve(y_test_cpu, probs)
roc_auc = auc(fpr, tpr)

"""# SVM"""


def clear_memory():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    if hasattr(cp.cuda, 'Device'):
        cp.cuda.Device().synchronize()

X = df_clean[['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
              'filtered_eda_values', 'filtered_temperature_values',
              'filtered_steps_values', 'filtered_azimuth',
              'filtered_duration_fixation', 'filtered_duration_blink']]
y = df_clean['Potential_FoF']

X = X.astype('float32')

le = LabelEncoder()
y_encoded = le.fit_transform(y)

y_cpu = y_encoded.values_host
class_weights = compute_class_weight('balanced', classes=np.unique(y_cpu), y=y_cpu)
class_weight_dict = dict(zip(np.unique(y_cpu), class_weights))

 with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

clear_memory()

y_train_cpu = y_train.values_host
neg_count = (y_train_cpu == 0).sum()
pos_count = (y_train_cpu == 1).sum()
imbalance_ratio = neg_count / pos_count

 (CRITICAL for SVM - all features must be on same scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

del X, X_train, X_test
clear_memory()

y_train_cpu = y_train.values.get().astype(np.int32)
y_test_cpu = y_test.values.get().astype(np.int32)

del y_train, y_test
clear_memory()

baseline_svm = cuSVM(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    tol=1e-3,
    max_iter=1000,
    class_weight='balanced'
)

baseline_svm.fit(X_train_scaled, cudf.Series(y_train_cpu))

y_pred_baseline = baseline_svm.predict(X_test_scaled)
y_pred_baseline_cpu = y_pred_baseline.to_pandas().values

y_decision_baseline = baseline_svm.decision_function(X_test_scaled)
y_decision_baseline_cpu = y_decision_baseline.to_pandas().values
y_proba_baseline = expit(y_decision_baseline_cpu)

precision_base, recall_base, thresholds_base = precision_recall_curve(y_test_cpu, y_proba_baseline)
f1_scores_base = 2 * (precision_base * recall_base) / (precision_base + recall_base + 1e-8)
optimal_threshold_base = thresholds_base[np.argmax(f1_scores_base)]

y_pred_baseline_optimal = (y_proba_baseline >= optimal_threshold_base).astype(int)

baseline_accuracy = accuracy_score(y_test_cpu, y_pred_baseline_optimal)
baseline_auc = roc_auc_score(y_test_cpu, y_proba_baseline)
baseline_f1 = f1_score(y_test_cpu, y_pred_baseline_optimal)

del baseline_svm, y_pred_baseline, y_decision_baseline
clear_memory()

X_train_np = X_train_scaled.values.get().astype(np.float32)
y_train_np = y_train_cpu.copy()

del X_train_scaled
clear_memory()

def objective(trial):
    C = trial.suggest_float('C', 0.01, 1000.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])

    if kernel in ['rbf', 'poly']:
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        degree = trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3
    else:
        gamma = 'scale'
        degree = 3

    tol = trial.suggest_float('tol', 1e-5, 1e-2, log=True)
    max_iter = trial.suggest_categorical('max_iter', [1000, 2000, 3000])

    class_weight_strategy = trial.suggest_categorical('class_weight', ['balanced', 'custom'])

    if class_weight_strategy == 'custom':
        minority_weight_multiplier = trial.suggest_float('minority_weight', 1.0, 20.0)
        sample_weights = np.ones(len(y_train_np))
        sample_weights[y_train_np == 1] = minority_weight_multiplier
    else:
        sample_weights = np.ones(len(y_train_np))
        sample_weights[y_train_np == 0] = class_weight_dict[0]
        sample_weights[y_train_np == 1] = class_weight_dict[1]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        try:
            X_tr_fold = X_train_np[train_idx]
            X_val_fold = X_train_np[val_idx]
            y_tr_fold = y_train_np[train_idx]
            y_val_fold = y_train_np[val_idx]
            weights_fold = sample_weights[train_idx]

            X_tr_cudf = cudf.DataFrame(X_tr_fold)
            y_tr_cudf = cudf.Series(y_tr_fold)
            X_val_cudf = cudf.DataFrame(X_val_fold)
            weights_cudf = cudf.Series(weights_fold)

            if kernel == 'linear':
                fold_svm = cuLinearSVM(
                    C=C,
                    tol=tol,
                    max_iter=max_iter,
                    loss='hinge',
                    fit_intercept=True
                )
            else:
                fold_svm = cuSVM(
                    C=C,
                    kernel=kernel,
                    gamma=gamma,
                    degree=degree,
                    tol=tol,
                    max_iter=max_iter
                )

            fold_svm.fit(X_tr_cudf, y_tr_cudf, sample_weight=weights_cudf)

            val_decision = fold_svm.decision_function(X_val_cudf)
            val_decision_cpu = val_decision.to_pandas().values
            val_proba = expit(val_decision_cpu)

            fold_precision, fold_recall, fold_thresholds = precision_recall_curve(y_val_fold, val_proba)
            fold_f1_scores = 2 * (fold_precision * fold_recall) / (fold_precision + fold_recall + 1e-8)
            fold_optimal_threshold = fold_thresholds[np.argmax(fold_f1_scores)]

            val_pred = (val_proba >= fold_optimal_threshold).astype(int)

            f1 = f1_score(y_val_fold, val_pred)
            f1_scores.append(f1)

            del fold_svm, val_decision, X_tr_cudf, y_tr_cudf, X_val_cudf, weights_cudf
            del X_tr_fold, X_val_fold, val_decision_cpu, val_proba, weights_fold
            clear_memory()

        except Exception as e:
            try:
                del fold_svm
            except:
                pass
            clear_memory()
            return 0.0

    if len(f1_scores) == 0:
        return 0.0

    return np.mean(f1_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

clear_memory()

best_params = study.best_trial.params

if best_params.get('class_weight', 'balanced') == 'custom':
    final_sample_weights = np.ones(len(y_train_np))
    final_sample_weights[y_train_np == 1] = best_params['minority_weight']
else:
    final_sample_weights = np.ones(len(y_train_np))
    final_sample_weights[y_train_np == 0] = class_weight_dict[0]
    final_sample_weights[y_train_np == 1] = class_weight_dict[1]

final_svm_clf = cuSVM(
    C=best_params['C'],
    kernel=best_params['kernel'],
    gamma=best_params.get('gamma', 'scale'),
    degree=best_params.get('degree', 3),
    tol=best_params['tol'],
    max_iter=best_params['max_iter']
)

X_train_final = cudf.DataFrame(X_train_np)
y_train_final = cudf.Series(y_train_np)
weights_final = cudf.Series(final_sample_weights)

final_svm_clf.fit(X_train_final, y_train_final, sample_weight=weights_final)

 with comprehensive metrics
final_decision = final_svm_clf.decision_function(X_test_scaled)
final_decision_cpu = final_decision.to_pandas().values
final_proba = expit(final_decision_cpu)

final_precision, final_recall, final_thresholds = precision_recall_curve(y_test_cpu, final_proba)
final_f1_scores = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-8)
final_optimal_threshold = final_thresholds[np.argmax(final_f1_scores)]

y_pred_final = (final_proba >= final_optimal_threshold).astype(int)

final_accuracy = accuracy_score(y_test_cpu, y_pred_final)
final_auc = roc_auc_score(y_test_cpu, final_proba)
final_f1 = f1_score(y_test_cpu, y_pred_final)
final_precision_score = precision_score(y_test_cpu, y_pred_final)
final_recall_score = recall_score(y_test_cpu, y_pred_final)

cm = confusion_matrix(y_test_cpu, y_pred_final)

feature_names = ['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
                'filtered_eda_values', 'filtered_temperature_values',
                'filtered_steps_values', 'filtered_azimuth',
                'filtered_duration_fixation', 'filtered_duration_blink']

kernel_type = best_params['kernel']
model_desc = 'LinearSVC (optimized for linear)' if kernel_type == 'linear' else f'SVC with {kernel_type} kernel'

class_weight_info = best_params.get('class_weight', 'balanced')

try:
    n_support = final_svm_clf.n_support_
except:
    print("Support vector count not available in cuML SVM")

thresholds_to_test = [0.3, 0.4, 0.5, final_optimal_threshold, 0.6, 0.7]
for thresh in thresholds_to_test:
    pred_thresh = (final_proba >= thresh).astype(int)
    acc_thresh = accuracy_score(y_test_cpu, pred_thresh)
    f1_thresh = f1_score(y_test_cpu, pred_thresh)
    precision_thresh = precision_score(y_test_cpu, pred_thresh, zero_division=0)
    recall_thresh = recall_score(y_test_cpu, pred_thresh)
    marker = " ← OPTIMAL" if abs(thresh - final_optimal_threshold) < 0.01 else ""
kernel_type = best_params['kernel']
model_type = 'LinearSVC (specialized linear solver)' if kernel_type == 'linear' else f'SVC with {kernel_type} kernel'
boundary_type = 'Linear (hyperplane)' if kernel_type == 'linear' else 'Non-linear'
optimization_note = 'LinearSVC for faster linear training' if kernel_type == 'linear' else 'Standard SVC for non-linear kernels'


model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

svm_path = os.path.join(model_dir, "svm_model.pkl")
joblib.dump(final_svm_clf, svm_path)

scaler_path = os.path.join(model_dir, "scaler_svm.joblib")
joblib.dump(scaler, scaler_path)

le_path = os.path.join(model_dir, "label_encoder_svm.joblib")
joblib.dump(le, le_path)


probs = final_proba

fpr, tpr, _ = roc_curve(y_test_cpu, probs)
roc_auc = auc(fpr, tpr)

"""# XGBoost"""


def clear_memory():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    if hasattr(cp.cuda, 'Device'):
        cp.cuda.Device().synchronize()

X = df_clean[['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
              'filtered_eda_values', 'filtered_temperature_values',
              'filtered_steps_values', 'filtered_azimuth',
              'filtered_duration_fixation', 'filtered_duration_blink']]
y = df_clean['Potential_FoF']

X = X.astype('float32')

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

clear_memory()

y_train_cpu = y_train.values_host
neg_count = (y_train_cpu == 0).sum()
pos_count = (y_train_cpu == 1).sum()

 (beneficial but not critical for tree-based methods like XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

del X, X_train, X_test
clear_memory()

X_train_cpu = X_train_scaled.values.get().astype(np.float32)
X_test_cpu = X_test_scaled.values.get().astype(np.float32)
y_train_cpu = y_train.values.get().astype(np.int32)
y_test_cpu = y_test.values.get().astype(np.int32)

del X_train_scaled, X_test_scaled, y_train, y_test
clear_memory()

scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    tree_method='gpu_hist' if cp.cuda.is_available() else 'hist',
    gpu_id=0 if cp.cuda.is_available() else None
)

xgb_clf.fit(X_train_cpu, y_train_cpu)

y_pred = xgb_clf.predict(X_test_cpu)
y_pred_proba = xgb_clf.predict_proba(X_test_cpu)[:, 1]

accuracy = accuracy_score(y_test_cpu, y_pred)

del xgb_clf
clear_memory()

def objective(trial):
    n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 200, 300])
    max_depth = trial.suggest_categorical('max_depth', [3, 4, 6, 8, 10])
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2, 0.3])
    subsample = trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1.0])
    colsample_bytree = trial.suggest_categorical('colsample_bytree', [0.6, 0.7, 0.8, 0.9, 1.0])
    reg_alpha = trial.suggest_categorical('reg_alpha', [0, 0.01, 0.1, 1, 10])
    reg_lambda = trial.suggest_categorical('reg_lambda', [0, 0.01, 0.1, 1, 10])
    min_child_weight = trial.suggest_categorical('min_child_weight', [1, 3, 5, 7])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_cpu, y_train_cpu)):
        try:
            X_tr_fold = X_train_cpu[train_idx]
            X_val_fold = X_train_cpu[val_idx]
            y_tr_fold = y_train_cpu[train_idx]
            y_val_fold = y_train_cpu[val_idx]

            fold_xgb = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                min_child_weight=min_child_weight,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss',
                tree_method='gpu_hist' if cp.cuda.is_available() else 'hist',
                gpu_id=0 if cp.cuda.is_available() else None,
                verbosity=0
            )

            fold_xgb.fit(X_tr_fold, y_tr_fold)

            val_proba = fold_xgb.predict_proba(X_val_fold)[:, 1]

            auc = roc_auc_score(y_val_fold, val_proba)
            auc_scores.append(auc)

            del fold_xgb, val_proba
            del X_tr_fold, X_val_fold
            clear_memory()

        except Exception as e:
            try:
                del fold_xgb
            except:
                pass
            clear_memory()
            return 0.0

    if len(auc_scores) == 0:
        return 0.0

    return np.mean(auc_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

clear_memory()

best_params = study.best_trial.params
final_xgb_clf = xgb.XGBClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    min_child_weight=best_params['min_child_weight'],
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    tree_method='gpu_hist' if cp.cuda.is_available() else 'hist',
    gpu_id=0 if cp.cuda.is_available() else None
)

final_xgb_clf.fit(X_train_cpu, y_train_cpu)

final_proba = final_xgb_clf.predict_proba(X_test_cpu)[:, 1]

final_auc = roc_auc_score(y_test_cpu, final_proba)

precision, recall, thresholds = precision_recall_curve(y_test_cpu, final_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]

y_pred_best = (final_proba >= optimal_threshold).astype(int)
accuracy_best = accuracy_score(y_test_cpu, y_pred_best)

feature_names = ['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
                'filtered_eda_values', 'filtered_temperature_values',
                'filtered_steps_values', 'filtered_azimuth',
                'filtered_duration_fixation', 'filtered_duration_blink']

feature_importance = final_xgb_clf.feature_importances_

feature_importance_pairs = list(zip(feature_names, feature_importance))
feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)


FEATURES = [
    'filtered_acc_x','filtered_acc_y','filtered_acc_z',
    'filtered_eda_values','filtered_temperature_values',
    'filtered_steps_values','filtered_azimuth',
    'filtered_duration_fixation','filtered_duration_blink'
]
TARGET = 'Potential_FoF'
SAVE_DIR = "drive/MyDrive/Colab Notebooks/data"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_THRESH_OVERRIDE = None 

X = df_clean[FEATURES].astype('float32')
y = df_clean[TARGET]
ids = df_clean['id'] if 'id' in df_clean.columns else df_clean.index.to_series()

 on GPU
le = LabelEncoder()
y_enc = le.fit_transform(y)

y_np = y_enc.to_pandas().values
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
(train_idx, test_idx), = sss.split(np.zeros_like(y_np), y_np)

X_tr = X.iloc[train_idx]; X_te = X.iloc[test_idx]
y_tr = y_enc.iloc[train_idx]; y_te = y_enc.iloc[test_idx]
ids_te = ids.iloc[test_idx] if hasattr(ids, "iloc") else ids.take(test_idx)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

def to_cpu_np(gobj, dtype=None):
    arr = gobj.values if hasattr(gobj, "values") else gobj
    arr = arr.get() if hasattr(arr, "get") else arr
    return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)

X_train = to_cpu_np(X_tr_sc, np.float32)
X_test  = to_cpu_np(X_te_sc,  np.float32)
y_train = to_cpu_np(y_tr,     np.int32)
y_test  = to_cpu_np(y_te,     np.int32)
ids_test = np.asarray(ids_te.to_pandas() if hasattr(ids_te, "to_pandas") else ids_te)

neg = (y_train == 0).sum(); pos = (y_train == 1).sum()
scale_pos_weight = float(neg / max(pos, 1))


best_params = {
    "n_estimators": 200,
    "max_depth": 10,
    "learning_rate": 0.3,
    "subsample": 0.9,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "min_child_weight": 1
}

xgb_clf = xgb.XGBClassifier(
    **best_params,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    tree_method='gpu_hist' if cp.cuda.is_available() else 'hist',
    gpu_id=0 if cp.cuda.is_available() else None,
    verbosity=0
)
xgb_clf.fit(X_train, y_train)

proba = xgb_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, proba, pos_label=1)
roc_auc = auc(fpr, tpr)

best_thresh = 0.7076

y_pred = (proba >= best_thresh).astype(int)

acc = accuracy_score(y_test, y_pred)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)
f1_minority = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

np.savez(os.path.join(SAVE_DIR, "roc_data_xgb_best.npz"),
         fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc,
         accuracy=acc,
         precision_macro=prec_macro,
         f1_macro=f1_macro,
         f1_minority=f1_minority,
         threshold_used=best_thresh,
         ids=ids_test)


model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

xgb_path = os.path.join(model_dir, "xgb_model.joblib")
joblib.dump(final_xgb_clf, xgb_path)

scaler_path = os.path.join(model_dir, "scaler_xgb.joblib")
joblib.dump(scaler, scaler_path)

le_path = os.path.join(model_dir, "label_encoder_xgb.joblib")
joblib.dump(le, le_path)

probs = final_proba

fpr, tpr, _ = roc_curve(y_test_cpu, probs)
roc_auc = auc(fpr, tpr)

"""# CB ensemble"""


def to_numpy_if_needed(proba):
    if hasattr(proba, 'to_pandas'):
        return proba.to_pandas().values
    return proba

def optimize_model(model_class, X, y, model_name, trial):
    if model_name == 'rf':
        n_estimators = trial.suggest_categorical('n_estimators', [50, 100])
        max_depth = trial.suggest_categorical('max_depth', [10, 20])
        model = cuRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=1.0,
            n_streams=1,
            random_state=42
        )
        model.fit(X, y)
        return model, n_estimators, max_depth, None, None, None

    elif model_name == 'catboost':
        iterations = trial.suggest_categorical('iterations', [50, 100, 200, 300])
        depth = trial.suggest_categorical('depth', [4, 6, 8, 10])
        learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2])
        l2_leaf_reg = trial.suggest_categorical('l2_leaf_reg', [1, 3, 5, 7, 9])
        border_count = trial.suggest_categorical('border_count', [32, 64, 128])

        X_cpu = X.to_pandas().values
        y_cpu = y.to_pandas().values
        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=42,
            loss_function='Logloss',
            eval_metric='F1',
            verbose=False,
            task_type='GPU' if cp.cuda.is_available() else 'CPU',
            devices='0' if cp.cuda.is_available() else None
        )
        model.fit(X_cpu, y_cpu)
        return model, None, None, depth, learning_rate, None

    else:
        n_neighbors = trial.suggest_categorical('n_neighbors', [3,5,7])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
        model.fit(X, y)
        return model, None, None, None, None, n_neighbors

def find_best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


X = df_clean[['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
              'filtered_eda_values', 'filtered_temperature_values',
              'filtered_steps_values', 'filtered_azimuth',
              'filtered_duration_fixation', 'filtered_duration_blink']]
y = df_clean['Potential_FoF']

X = X.astype('float32')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

y_test_cpu = y_test.to_pandas().values

X_train_np = X_train.values.get()
y_train_np = y_train.values.get()

X_train_pd = cudf.DataFrame(X_train_np).to_pandas()
y_train_pd = cudf.Series(y_train_np).to_pandas()

sm = SMOTE(random_state=42)
X_resampled_pd, y_resampled_pd = sm.fit_resample(X_train_pd, y_train_pd)

X_res_cudf = cudf.DataFrame(X_resampled_pd)
y_res_cudf = cudf.Series(y_resampled_pd)

X_res_np = X_res_cudf.values.get()
y_res_np = y_res_cudf.values.get()

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_preds = []
cv_trues = []

def objective(trial):
    train_idx, val_idx = next(skf.split(X_res_np, y_res_np))
    X_tr_fold = X_res_np[train_idx]
    y_tr_fold = y_res_np[train_idx]
    X_val_fold = X_res_np[val_idx]
    y_val_fold = y_res_np[val_idx]

    X_tr_cudf = cudf.DataFrame(X_tr_fold)
    y_tr_cudf = cudf.Series(y_tr_fold)
    X_val_cudf = cudf.DataFrame(X_val_fold)

    rf_model, rf_n_estimators, rf_max_depth, _, _, _ = optimize_model(cuRF, X_tr_cudf, y_tr_cudf, 'rf', trial)
    cat_model, _, _, cat_depth, cat_learning_rate, _ = optimize_model(CatBoostClassifier, X_tr_cudf, y_tr_cudf, 'catboost', trial)
    knn_model, _, _, _, _, knn_neighbors = optimize_model(KNeighborsClassifier, X_tr_cudf, y_tr_cudf, 'knn', trial)

    rf_proba = to_numpy_if_needed(rf_model.predict_proba(X_val_cudf))
    X_val_cpu = X_val_cudf.to_pandas().values
    cat_proba = cat_model.predict_proba(X_val_cpu)
    knn_proba = to_numpy_if_needed(knn_model.predict_proba(X_val_cudf))

    avg_proba = (rf_proba[:,1] + cat_proba[:,1] + knn_proba[:,1]) / 3.0
    y_pred_fold = (avg_proba >= 0.5).astype(int)

    return f1_score(y_val_fold, y_pred_fold, pos_label=1)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5, show_progress_bar=False)

def get_final_models(params):
    rf_n_estimators = params['n_estimators']
    rf_max_depth = params['max_depth']
    cat_depth = params['depth']
    cat_learning_rate = params['learning_rate']
    knn_neighbors = params['n_neighbors']

    rf_model = cuRF(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        max_features=1.0,
        n_streams=1,
        random_state=42
    )
    cat_model = CatBoostClassifier(
        iterations=200,
        depth=cat_depth,
        learning_rate=cat_learning_rate,
        loss_function='Logloss',
        eval_metric='F1',
        verbose=False,
        task_type='GPU'
    )
    knn_model = KNeighborsClassifier(n_neighbors=knn_neighbors, metric='euclidean')

    return rf_model, cat_model, knn_model, cat_depth, cat_learning_rate

final_rf, final_cat, final_knn, cat_depth, cat_learning_rate = get_final_models(study.best_params)

cv_accuracies = []
proba_for_threshold = []
y_for_threshold = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_res_np, y_res_np)):
    X_tr_fold = X_res_np[train_idx]
    y_tr_fold = y_res_np[train_idx]
    X_val_fold = X_res_np[val_idx]
    y_val_fold = y_res_np[val_idx]

    X_tr_cudf = cudf.DataFrame(X_tr_fold)
    y_tr_cudf = cudf.Series(y_tr_fold)
    X_val_cudf = cudf.DataFrame(X_val_fold)

    rf_clf = cuRF(
        n_estimators=final_rf.n_estimators,
        max_depth=final_rf.max_depth,
        max_features=1.0,
        n_streams=1,
        random_state=42
    )
    rf_clf.fit(X_tr_cudf, y_tr_cudf)

    X_tr_cpu = X_tr_cudf.to_pandas().values
    y_tr_cpu = y_tr_cudf.to_pandas().values
    cat_clf = CatBoostClassifier(
        iterations=200,
        depth=cat_depth,
        learning_rate=cat_learning_rate,
        loss_function='Logloss',
        eval_metric='F1',
        verbose=False,
        task_type='GPU'
    )
    cat_clf.fit(X_tr_cpu, y_tr_cpu)

    knn_clf = KNeighborsClassifier(n_neighbors=final_knn.n_neighbors, metric='euclidean')
    knn_clf.fit(X_tr_cudf, y_tr_cudf)

    rf_proba = to_numpy_if_needed(rf_clf.predict_proba(X_val_cudf))
    X_val_cpu = X_val_cudf.to_pandas().values
    cat_proba = cat_clf.predict_proba(X_val_cpu)
    knn_proba = to_numpy_if_needed(knn_clf.predict_proba(X_val_cudf))

    avg_proba = (rf_proba[:,1] + cat_proba[:,1] + knn_proba[:,1]) / 3.0
    y_pred_fold = (avg_proba >= 0.5).astype(int)

    acc = accuracy_score(y_val_fold, y_pred_fold)
    cv_accuracies.append(acc)

    cv_preds.extend(y_pred_fold)
    cv_trues.extend(y_val_fold)

    proba_for_threshold.extend(avg_proba)
    y_for_threshold.extend(y_val_fold)

mean_acc = np.mean(cv_accuracies)

conf_matrix_cv = confusion_matrix(cv_trues, cv_preds)
class_labels = le.inverse_transform([0, 1]).to_pandas().tolist()

proba_for_threshold = np.array(proba_for_threshold)
y_for_threshold = np.array(y_for_threshold)

best_thresh, best_f1 = find_best_threshold(y_for_threshold, proba_for_threshold)

X_res_cudf = cudf.DataFrame(X_res_np)
y_res_cudf = cudf.Series(y_res_np)

rf_final = cuRF(
    n_estimators=final_rf.n_estimators,
    max_depth=final_rf.max_depth,
    max_features=1.0,
    n_streams=1,
    random_state=42
)
rf_final.fit(X_res_cudf, y_res_cudf)

X_res_cpu = X_res_cudf.to_pandas().values
y_res_cpu = y_res_cudf.to_pandas().values
cat_final = CatBoostClassifier(
    iterations=200,
    depth=cat_depth,
    learning_rate=cat_learning_rate,
    loss_function='Logloss',
    eval_metric='F1',
    verbose=False,
    task_type='GPU'
)
cat_final.fit(X_res_cpu, y_res_cpu)

knn_final = KNeighborsClassifier(n_neighbors=final_knn.n_neighbors, metric='euclidean')
knn_final.fit(X_res_cudf, y_res_cudf)

rf_proba_test = to_numpy_if_needed(rf_final.predict_proba(X_test))
X_test_cpu = X_test.to_pandas().values
cat_proba_test = cat_final.predict_proba(X_test_cpu)
knn_proba_test = to_numpy_if_needed(knn_final.predict_proba(X_test))

avg_proba_test = (rf_proba_test[:,1] + cat_proba_test[:,1] + knn_proba_test[:,1]) / 3.0
y_pred_ensemble_test = (avg_proba_test >= best_thresh).astype(int)

accuracy_test = accuracy_score(y_test_cpu, y_pred_ensemble_test)

conf_matrix_test = confusion_matrix(y_test_cpu, y_pred_ensemble_test)

precisions, recalls, _ = precision_recall_curve(y_test_cpu, avg_proba_test)
pr_auc = auc(recalls, precisions)


model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

rf_path   = os.path.join(model_dir, "rf_final_cben.pkl")
joblib.dump(rf_final, rf_path)

cat_path  = os.path.join(model_dir, "cat_final_cben.pkl")
joblib.dump(cat_final, cat_path)

knn_path  = os.path.join(model_dir, "knn_final_cben.pkl")
joblib.dump(knn_final, knn_path)

scaler_path = os.path.join(model_dir, "scaler_ensemble_cben.joblib")
joblib.dump(scaler, scaler_path)

le_path = os.path.join(model_dir, "label_encoder_ensemble.joblib")
joblib.dump(le, le_path)



fpr, tpr, _ = roc_curve(y_test_cpu, avg_proba_test)
roc_auc = auc(fpr, tpr)


 (from first script)
def clear_memory():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    if hasattr(cp.cuda, 'Device'):
        cp.cuda.Device().synchronize()

def to_numpy_if_needed(proba):
    if hasattr(proba, 'to_pandas'):
        return proba.to_pandas().values
    return proba

def find_best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def catboost_objective(trial, X_train_cpu, y_train_cpu):
    iterations = trial.suggest_categorical('iterations', [50, 100, 200, 300])
    depth = trial.suggest_categorical('depth', [4, 6, 8, 10])
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2])
    l2_leaf_reg = trial.suggest_categorical('l2_leaf_reg', [1, 3, 5, 7, 9])
    border_count = trial.suggest_categorical('border_count', [32, 64, 128])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_cpu, y_train_cpu)):
        try:
            X_tr_fold = X_train_cpu[train_idx]
            X_val_fold = X_train_cpu[val_idx]
            y_tr_fold = y_train_cpu[train_idx]
            y_val_fold = y_train_cpu[val_idx]

            fold_catboost = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                border_count=border_count,
                random_seed=42,
                verbose=False,
                task_type='GPU' if cp.cuda.is_available() else 'CPU',
                devices='0' if cp.cuda.is_available() else None
            )

            fold_catboost.fit(X_tr_fold, y_tr_fold)

            val_proba = fold_catboost.predict_proba(X_val_fold)[:, 1]

            auc = roc_auc_score(y_val_fold, val_proba)
            auc_scores.append(auc)

            del fold_catboost, val_proba
            del X_tr_fold, X_val_fold
            clear_memory()

        except Exception as e:
            try:
                del fold_catboost
            except:
                pass
            clear_memory()
            return 0.0

    if len(auc_scores) == 0:
        return 0.0

    return np.mean(auc_scores)

def knn_objective(trial, X_train_np, y_train_np):
    n_neighbors = trial.suggest_categorical('n_neighbors', [3, 5, 7, 9, 11, 15, 21])
    weights = trial.suggest_categorical('weights', ['uniform'])
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])

    if metric == 'minkowski':
        p = trial.suggest_categorical('p', [1, 2, 3])
    else:
        p = 2

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        try:
            X_tr_fold = X_train_np[train_idx].astype(np.float32)
            X_val_fold = X_train_np[val_idx].astype(np.float32)
            y_tr_fold = y_train_np[train_idx]
            y_val_fold = y_train_np[val_idx]

            X_tr_cudf = cudf.DataFrame(X_tr_fold)
            y_tr_cudf = cudf.Series(y_tr_fold)
            X_val_cudf = cudf.DataFrame(X_val_fold)

            fold_knn = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
                p=p if metric == 'minkowski' else 2
            )

            fold_knn.fit(X_tr_cudf, y_tr_cudf)

            val_proba = fold_knn.predict_proba(X_val_cudf)
            val_proba_cpu = val_proba.to_pandas().values[:, 1]

            auc = roc_auc_score(y_val_fold, val_proba_cpu)
            auc_scores.append(auc)

            del fold_knn, val_proba, X_tr_cudf, y_tr_cudf, X_val_cudf
            del X_tr_fold, X_val_fold, val_proba_cpu
            clear_memory()

        except Exception as e:
            try:
                del fold_knn, X_tr_cudf, y_tr_cudf, X_val_cudf
            except:
                pass
            clear_memory()
            return 0.0

    if len(auc_scores) == 0:
        return 0.0

    return np.mean(auc_scores)

def rf_objective(trial, X_train_np, y_train_np):
    n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 150, 200]) 
    max_depth = trial.suggest_categorical('max_depth', [5, 10, 15, 20])
    max_features = trial.suggest_categorical('max_features', [0.5, 0.7, 0.8, 1.0])
    min_samples_split = trial.suggest_categorical('min_samples_split', [2, 5])
    min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [1, 2])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        try:
            X_tr_fold = X_train_np[train_idx].astype(np.float32)
            X_val_fold = X_train_np[val_idx].astype(np.float32)
            y_tr_fold = y_train_np[train_idx]
            y_val_fold = y_train_np[val_idx]

            X_tr_cudf = cudf.DataFrame(X_tr_fold)
            y_tr_cudf = cudf.Series(y_tr_fold)
            X_val_cudf = cudf.DataFrame(X_val_fold)

            fold_rf = cuRF(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_streams=1,
                random_state=42
            )

            fold_rf.fit(X_tr_cudf, y_tr_cudf)

            val_proba = fold_rf.predict_proba(X_val_cudf)
            val_proba_cpu = val_proba.to_pandas().values[:, 1]

            auc = roc_auc_score(y_val_fold, val_proba_cpu)
            auc_scores.append(auc)

            del fold_rf, val_proba, X_tr_cudf, y_tr_cudf, X_val_cudf
            del X_tr_fold, X_val_fold, val_proba_cpu
            clear_memory()

        except Exception as e:
            try:
                del fold_rf, X_tr_cudf, y_tr_cudf, X_val_cudf
            except:
                pass
            clear_memory()
            return 0.0

    if len(auc_scores) == 0:
        return 0.0

    return np.mean(auc_scores)


X = df_clean[['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
              'filtered_eda_values', 'filtered_temperature_values',
              'filtered_steps_values', 'filtered_azimuth',
              'filtered_duration_fixation', 'filtered_duration_blink']]
y = df_clean['Potential_FoF']

 (from first script)
X = X.astype('float32')

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

clear_memory()

 (from first script)
y_train_cpu = y_train.values_host
neg_count = (y_train_cpu == 0).sum()
pos_count = (y_train_cpu == 1).sum()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

del X, X_train, X_test
clear_memory()

 and SMOTE (from first script)
X_train_cpu = X_train_scaled.values.get().astype(np.float32)
X_test_cpu = X_test_scaled.values.get().astype(np.float32)
y_train_cpu = y_train.values.get().astype(np.int32)
y_test_cpu = y_test.values.get().astype(np.int32)

del X_train_scaled, X_test_scaled, y_train, y_test
clear_memory()

catboost_study = optuna.create_study(direction="maximize")
catboost_study.optimize(
    lambda trial: catboost_objective(trial, X_train_cpu, y_train_cpu),
    n_trials=30,
    show_progress_bar=True
)

clear_memory()

knn_study = optuna.create_study(direction="maximize")
knn_study.optimize(
    lambda trial: knn_objective(trial, X_train_cpu, y_train_cpu),
    n_trials=30,
    show_progress_bar=True
)

clear_memory()

rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(
    lambda trial: rf_objective(trial, X_train_cpu, y_train_cpu),
    n_trials=30,
    show_progress_bar=True
)

clear_memory()

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_preds = []
cv_trues = []
cv_accuracies = []
proba_for_threshold = []
y_for_threshold = []

best_cat_params = catboost_study.best_trial.params
best_knn_params = knn_study.best_trial.params
best_rf_params = rf_study.best_params

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_cpu, y_train_cpu)):
    X_tr_fold_cpu = X_train_cpu[train_idx]
    y_tr_fold_cpu = y_train_cpu[train_idx]
    X_val_fold_cpu = X_train_cpu[val_idx]
    y_val_fold_cpu = y_train_cpu[val_idx]

    X_tr_cudf = cudf.DataFrame(X_tr_fold_cpu)
    y_tr_cudf = cudf.Series(y_tr_fold_cpu)
    X_val_cudf = cudf.DataFrame(X_val_fold_cpu)

    rf_clf = cuRF(
        n_estimators=best_rf_params['n_estimators'],
        max_depth=best_rf_params['max_depth'],
        max_features=best_rf_params['max_features'],
        min_samples_split=best_rf_params['min_samples_split'],
        min_samples_leaf=best_rf_params['min_samples_leaf'],
        n_streams=1,
        random_state=42
    )
    rf_clf.fit(X_tr_cudf, y_tr_cudf)

    cat_clf = CatBoostClassifier(
        iterations=best_cat_params['iterations'],
        depth=best_cat_params['depth'],
        learning_rate=best_cat_params['learning_rate'],
        l2_leaf_reg=best_cat_params['l2_leaf_reg'],
        border_count=best_cat_params['border_count'],
        random_seed=42,
        verbose=False,
        task_type='GPU' if cp.cuda.is_available() else 'CPU',
        devices='0' if cp.cuda.is_available() else None
    )
    cat_clf.fit(X_tr_fold_cpu, y_tr_fold_cpu)

    knn_clf = KNeighborsClassifier(
        n_neighbors=best_knn_params['n_neighbors'],
        weights=best_knn_params['weights'],
        metric=best_knn_params['metric'],
        p=best_knn_params.get('p', 2)
    )
    knn_clf.fit(X_tr_cudf, y_tr_cudf)

    rf_proba = to_numpy_if_needed(rf_clf.predict_proba(X_val_cudf))
    cat_proba = cat_clf.predict_proba(X_val_fold_cpu)
    knn_proba = to_numpy_if_needed(knn_clf.predict_proba(X_val_cudf))

    avg_proba = (rf_proba[:,1] + cat_proba[:,1] + knn_proba[:,1]) / 3.0
    y_pred_fold = (avg_proba >= 0.5).astype(int)

    acc = accuracy_score(y_val_fold_cpu, y_pred_fold)
    cv_accuracies.append(acc)

    cv_preds.extend(y_pred_fold)
    cv_trues.extend(y_val_fold_cpu)

    proba_for_threshold.extend(avg_proba)
    y_for_threshold.extend(y_val_fold_cpu)

    del rf_clf, cat_clf, knn_clf
    clear_memory()

mean_acc = np.mean(cv_accuracies)

proba_for_threshold = np.array(proba_for_threshold)
y_for_threshold = np.array(y_for_threshold)

best_thresh, best_f1 = find_best_threshold(y_for_threshold, proba_for_threshold)

X_train_cudf = cudf.DataFrame(X_train_cpu)
y_train_cudf = cudf.Series(y_train_cpu)

rf_final = cuRF(
    n_estimators=best_rf_params['n_estimators'],
    max_depth=best_rf_params['max_depth'],
    max_features=best_rf_params['max_features'],
    min_samples_split=best_rf_params['min_samples_split'],
    min_samples_leaf=best_rf_params['min_samples_leaf'],
    n_streams=1,
    random_state=42
)
rf_final.fit(X_train_cudf, y_train_cudf)

cat_final = CatBoostClassifier(
    iterations=best_cat_params['iterations'],
    depth=best_cat_params['depth'],
    learning_rate=best_cat_params['learning_rate'],
    l2_leaf_reg=best_cat_params['l2_leaf_reg'],
    border_count=best_cat_params['border_count'],
    random_seed=42,
    verbose=False,
    task_type='GPU' if cp.cuda.is_available() else 'CPU',
    devices='0' if cp.cuda.is_available() else None
)
cat_final.fit(X_train_cpu, y_train_cpu)

knn_final = KNeighborsClassifier(
    n_neighbors=best_knn_params['n_neighbors'],
    weights=best_knn_params['weights'],
    metric=best_knn_params['metric'],
    p=best_knn_params.get('p', 2)
)
knn_final.fit(X_train_cudf, y_train_cudf)

X_test_cudf = cudf.DataFrame(X_test_cpu)

rf_proba_test = to_numpy_if_needed(rf_final.predict_proba(X_test_cudf))
cat_proba_test = cat_final.predict_proba(X_test_cpu)
knn_proba_test = to_numpy_if_needed(knn_final.predict_proba(X_test_cudf))

avg_proba_test = (rf_proba_test[:,1] + cat_proba_test[:,1] + knn_proba_test[:,1]) / 3.0
y_pred_ensemble_test = (avg_proba_test >= best_thresh).astype(int)

accuracy_test = accuracy_score(y_test_cpu, y_pred_ensemble_test)

conf_matrix_test = confusion_matrix(y_test_cpu, y_pred_ensemble_test)
class_labels = le.inverse_transform([0, 1]).to_pandas().tolist()

precisions, recalls, _ = precision_recall_curve(y_test_cpu, avg_proba_test)
pr_auc = auc(recalls, precisions)

precisions, recalls, _ = precision_recall_curve(y_test_cpu, avg_proba_test)
pr_auc = auc(recalls, precisions)

feature_names = ['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
                'filtered_eda_values', 'filtered_temperature_values',
                'filtered_steps_values', 'filtered_azimuth',
                'filtered_duration_fixation', 'filtered_duration_blink']

feature_importance = cat_final.get_feature_importance()


save_dir = "drive/MyDrive/ensemble_results"
os.makedirs(save_dir, exist_ok=True)

fpr, tpr, roc_thr = roc_curve(y_test_cpu, avg_proba_test, pos_label=1)
roc_auc = auc(fpr, tpr)

prec_curve, rec_curve, _ = precision_recall_curve(y_test_cpu, avg_proba_test)
pr_auc = auc(rec_curve, prec_curve)

y_pred_thresh = (avg_proba_test >= best_thresh).astype(int)
acc = accuracy_score(y_test_cpu, y_pred_thresh)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test_cpu, y_pred_thresh, average="macro", zero_division=0
)
f1_minority = f1_score(y_test_cpu, y_pred_thresh, pos_label=1, zero_division=0)

np.savez(os.path.join(save_dir, "ensemble_test_metrics.npz"),
         fpr=fpr, tpr=tpr, thresholds=roc_thr, auc=roc_auc,
         pr_auc=pr_auc,
         accuracy=acc,
         precision_macro=prec_macro,
         f1_macro=f1_macro,
         f1_minority=f1_minority,
         threshold_used=float(best_thresh))

joblib.dump(rf_final, os.path.join(save_dir, "rf_final.pkl"))
joblib.dump(cat_final, os.path.join(save_dir, "cat_final.pkl"))
joblib.dump(knn_final, os.path.join(save_dir, "knn_final.pkl"))
joblib.dump(scaler,   os.path.join(save_dir, "scaler.joblib"))
joblib.dump(le,       os.path.join(save_dir, "label_encoder.joblib"))

metrics = np.load("drive/MyDrive/ensemble_results/ensemble_test_metrics.npz")

fpr = metrics["fpr"]
tpr = metrics["tpr"]
roc_auc = metrics["auc"]

"""# LR ensemble"""


def to_numpy_if_needed(proba):
    if hasattr(proba, 'to_pandas'):
        return proba.to_pandas().values
    return proba

def optimize_model(model_class, X, y, model_name, trial):
    if model_name == 'rf':
        n_estimators = trial.suggest_categorical('n_estimators', [50, 100])
        max_depth = trial.suggest_categorical('max_depth', [10, 20])
        model = cuRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=1.0,
            n_streams=1,
            random_state=42
        )
    elif model_name == 'logreg':
        max_iter = trial.suggest_int('max_iter', 500, 1000, step=500)
        model = LogisticRegression(verbose=0, max_iter=max_iter)
    else:
        n_neighbors = trial.suggest_categorical('n_neighbors', [3,5,7])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    model.fit(X, y)
    return model

def find_best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

X = df_clean[['filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z',
              'filtered_eda_values', 'filtered_temperature_values',
              'filtered_steps_values', 'filtered_azimuth',
              'filtered_duration_fixation', 'filtered_duration_blink']]
y = df_clean['Potential_FoF']

X = X.astype('float32')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

y_test_cpu = y_test.to_pandas().values

X_train_np = X_train.values.get()
y_train_np = y_train.values.get()

X_train_pd = cudf.DataFrame(X_train_np).to_pandas()
y_train_pd = cudf.Series(y_train_np).to_pandas()

sm = SMOTE(random_state=42)
X_resampled_pd, y_resampled_pd = sm.fit_resample(X_train_pd, y_train_pd)

X_res_cudf = cudf.DataFrame(X_resampled_pd)
y_res_cudf = cudf.Series(y_resampled_pd)

X_res_np = X_res_cudf.values.get()
y_res_np = y_res_cudf.values.get()

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_preds = []
cv_trues = []

def objective(trial):
    train_idx, val_idx = next(skf.split(X_res_np, y_res_np))
    X_tr_fold = X_res_np[train_idx]
    y_tr_fold = y_res_np[train_idx]
    X_val_fold = X_res_np[val_idx]
    y_val_fold = y_res_np[val_idx]

    X_tr_cudf = cudf.DataFrame(X_tr_fold)
    y_tr_cudf = cudf.Series(y_tr_fold)
    X_val_cudf = cudf.DataFrame(X_val_fold)

    rf_model = optimize_model(cuRF, X_tr_cudf, y_tr_cudf, 'rf', trial)
    logreg_model = optimize_model(LogisticRegression, X_tr_cudf, y_tr_cudf, 'logreg', trial)
    knn_model = optimize_model(KNeighborsClassifier, X_tr_cudf, y_tr_cudf, 'knn', trial)

    rf_proba = to_numpy_if_needed(rf_model.predict_proba(X_val_cudf))
    logreg_proba = to_numpy_if_needed(logreg_model.predict_proba(X_val_cudf))
    knn_proba = to_numpy_if_needed(knn_model.predict_proba(X_val_cudf))

    avg_proba = (rf_proba[:,1] + logreg_proba[:,1] + knn_proba[:,1]) / 3.0
    y_pred_fold = (avg_proba >= 0.5).astype(int)

    return f1_score(y_val_fold, y_pred_fold, pos_label=1)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5, show_progress_bar=False)

def get_final_models(params):
    rf_model = cuRF(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        max_features=1.0,
        n_streams=1,
        random_state=42
    )
    logreg_model = LogisticRegression(verbose=0, max_iter=params['max_iter'])
    knn_model = KNeighborsClassifier(n_neighbors=params['n_neighbors'], metric='euclidean')
    return rf_model, logreg_model, knn_model

final_rf, final_logreg, final_knn = get_final_models(study.best_params)

cv_accuracies = []
proba_for_threshold = []
y_for_threshold = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_res_np, y_res_np)):
    X_tr_fold = X_res_np[train_idx]
    y_tr_fold = y_res_np[train_idx]
    X_val_fold = X_res_np[val_idx]
    y_val_fold = y_res_np[val_idx]

    X_tr_cudf = cudf.DataFrame(X_tr_fold)
    y_tr_cudf = cudf.Series(y_tr_fold)
    X_val_cudf = cudf.DataFrame(X_val_fold)

    rf_clf = cuRF(
        n_estimators=final_rf.n_estimators,
        max_depth=final_rf.max_depth,
        max_features=1.0,
        n_streams=1,
        random_state=42
    )
    log_reg_clf = LogisticRegression(verbose=0, max_iter=final_logreg.max_iter)
    knn_clf = KNeighborsClassifier(n_neighbors=final_knn.n_neighbors, metric='euclidean')

    rf_clf.fit(X_tr_cudf, y_tr_cudf)
    log_reg_clf.fit(X_tr_cudf, y_tr_cudf)
    knn_clf.fit(X_tr_cudf, y_tr_cudf)

    rf_proba = to_numpy_if_needed(rf_clf.predict_proba(X_val_cudf))
    log_reg_proba = to_numpy_if_needed(log_reg_clf.predict_proba(X_val_cudf))
    knn_proba = to_numpy_if_needed(knn_clf.predict_proba(X_val_cudf))

    avg_proba = (rf_proba[:,1] + log_reg_proba[:,1] + knn_proba[:,1]) / 3.0
    y_pred_fold = (avg_proba >= 0.5).astype(int)

    acc = accuracy_score(y_val_fold, y_pred_fold)
    cv_accuracies.append(acc)

    cv_preds.extend(y_pred_fold)
    cv_trues.extend(y_val_fold)

    proba_for_threshold.extend(avg_proba)
    y_for_threshold.extend(y_val_fold)

mean_acc = np.mean(cv_accuracies)

conf_matrix_cv = confusion_matrix(cv_trues, cv_preds)
class_labels = le.inverse_transform([0, 1]).to_pandas().tolist()

proba_for_threshold = np.array(proba_for_threshold)
y_for_threshold = np.array(y_for_threshold)

best_thresh, best_f1 = find_best_threshold(y_for_threshold, proba_for_threshold)

X_res_cudf = cudf.DataFrame(X_res_np)
y_res_cudf = cudf.Series(y_res_np)

rf_final = cuRF(
    n_estimators=final_rf.n_estimators,
    max_depth=final_rf.max_depth,
    max_features=1.0,
    n_streams=1,
    random_state=42
)
log_reg_final = LogisticRegression(verbose=0, max_iter=final_logreg.max_iter)
knn_final = KNeighborsClassifier(n_neighbors=final_knn.n_neighbors, metric='euclidean')

rf_final.fit(X_res_cudf, y_res_cudf)
log_reg_final.fit(X_res_cudf, y_res_cudf)
knn_final.fit(X_res_cudf, y_res_cudf)

rf_proba_test = to_numpy_if_needed(rf_final.predict_proba(X_test))
log_reg_proba_test = to_numpy_if_needed(log_reg_final.predict_proba(X_test))
knn_proba_test = to_numpy_if_needed(knn_final.predict_proba(X_test))

avg_proba_test = (rf_proba_test[:,1] + log_reg_proba_test[:,1] + knn_proba_test[:,1]) / 3.0
y_pred_ensemble_test = (avg_proba_test >= best_thresh).astype(int)

accuracy_test = accuracy_score(y_test_cpu, y_pred_ensemble_test)

conf_matrix_test = confusion_matrix(y_test_cpu, y_pred_ensemble_test)

precisions, recalls, _ = precision_recall_curve(y_test_cpu, avg_proba_test)
pr_auc = auc(recalls, precisions)


save_dir = "drive/MyDrive/ensemble_second"
os.makedirs(save_dir, exist_ok=True)

fpr, tpr, roc_thr = roc_curve(y_test_cpu, avg_proba_test, pos_label=1)
roc_auc = auc(fpr, tpr)

prec_curve, rec_curve, _ = precision_recall_curve(y_test_cpu, avg_proba_test)
pr_auc = auc(rec_curve, prec_curve)

y_pred_thresh = (avg_proba_test >= best_thresh).astype(int)
acc = accuracy_score(y_test_cpu, y_pred_thresh)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test_cpu, y_pred_thresh, average="macro", zero_division=0
)
f1_minority = f1_score(y_test_cpu, y_pred_thresh, pos_label=1, zero_division=0)

np.savez(os.path.join(save_dir, "ensemble_test_metrics.npz"),
         fpr=fpr,
         tpr=tpr,
         thresholds=roc_thr,
         auc=roc_auc,
         pr_auc=pr_auc,
         accuracy=acc,
         precision_macro=prec_macro,
         f1_macro=f1_macro,
         f1_minority=f1_minority,
         threshold_used=float(best_thresh))

joblib.dump(rf_final,       os.path.join(save_dir, "rf_final.pkl"))
joblib.dump(log_reg_final,  os.path.join(save_dir, "logreg_final.pkl"))
joblib.dump(knn_final,      os.path.join(save_dir, "knn_final.pkl"))
joblib.dump(scaler,         os.path.join(save_dir, "scaler.joblib"))
joblib.dump(le,             os.path.join(save_dir, "label_encoder.joblib"))


run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:8]
model_dir = os.path.join("models", f"ensemble_rf_logreg_knn_{run_id}")
os.makedirs(model_dir, exist_ok=True)

paths = {
    "rf":       os.path.join(model_dir, f"rf_final_{run_id}.pkl"),
    "logreg":   os.path.join(model_dir, f"logreg_final_{run_id}.pkl"),
    "knn":      os.path.join(model_dir, f"knn_final_{run_id}.pkl"),
    "scaler":   os.path.join(model_dir, f"scaler_{run_id}.joblib"),
    "encoder":  os.path.join(model_dir, f"label_encoder_{run_id}.joblib"),
    "roc_png":  os.path.join(model_dir, f"roc_curve_{run_id}.png"),
    "manifest": os.path.join(model_dir, f"manifest_{run_id}.json"),
}

joblib.dump(rf_final,      paths["rf"])
joblib.dump(log_reg_final, paths["logreg"])
joblib.dump(knn_final,     paths["knn"])
joblib.dump(scaler,        paths["scaler"])
joblib.dump(le,            paths["encoder"])

fpr, tpr, _ = roc_curve(y_test_cpu, avg_proba_test)
roc_auc = auc(fpr, tpr)


prec, rec, _ = precision_recall_curve(y_test_cpu, avg_proba_test)
pr_auc = auc(rec, prec)
manifest = {
    "run_id": run_id,
    "files": paths,
    "best_threshold": float(best_thresh),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "n_test": int(len(y_test_cpu)),
    "notes": "RF+LogReg+KNN ensemble with SMOTE + scaling"
}
with open(paths["manifest"], "w") as f:
    json.dump(manifest, f, indent=2)

