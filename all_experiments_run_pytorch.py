import os
import glob
import numpy as np
import pandas as pd
from helper_functions import data_mixer,X_set,y_set
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold,StratifiedKFold
from scipy.stats import pearsonr
from skorch import NeuralNetRegressor, NeuralNetClassifier
import torch
from models_pytorch import (
    CNNClassifier, CNNRegressor, LSTMClassifier, LSTMRegressor, MLPClassifier, MLPRegressor,svc,random_forest_clf,random_forest_reg,linear_regression
)
from joblib import Parallel, delayed
from collections import Counter

#########---PATHS---#############
balanced = r'C:\projects\thesis\all_datasets\Balanced_Data'
test = r'C:\projects\thesis\all_datasets\test_classification'
random = r'C:\projects\thesis\all_datasets\random_data'

#########---INPUTS---#############
n_points_list = [375, 460, 750]
transformation_list = ['none', 'fourier']
noise_levels = [2, 5, 10]
damage_percentage = [0.25, 0.5, 0.75]


def p_val(y_true, y_pred):
    return pearsonr(y_true, y_pred)[1]

def run_regression_fold(name, model_fn, X, X_dl, y, train_idx, test_idx):
    if name == 'MLP' or name =='LR' or name =='RF':
        X_train = X[train_idx].astype(np.float32)
        X_test = X[test_idx].astype(np.float32)
    elif name == 'CNN':
        X_train = np.transpose(X_dl[train_idx], (0, 2, 1)).astype(np.float32)
        X_test = np.transpose(X_dl[test_idx], (0, 2, 1)).astype(np.float32)
    elif name == 'LSTM':
        X_train = X_dl[train_idx].astype(np.float32)
        X_test = X_dl[test_idx].astype(np.float32)

    y_train = y[train_idx].astype(np.float32)
    y_test = y[test_idx].astype(np.float32)

    model = model_fn()
    model.fit(X_train, y_train)
    preds = model.predict(X_test).reshape(-1)
    mape = mean_absolute_percentage_error(y_test, preds)
    pval = p_val(y_test, preds)
    return mape, pval, preds, y_test



def regression_experiment_run():
    for n_points in n_points_list:
        all_results = []
        for transformation in transformation_list:
            y_random = y_set(random)['dmg'].values
            y_data = y_set(balanced)['dmg'].values

            for noise_percent in noise_levels:
                X_random = X_set(random, transformation, n_points, noise_percent=noise_percent)[0]
                X_data = X_set(balanced, transformation, n_points, noise_percent=noise_percent)[0]

                for perc in damage_percentage:
                    X_mixed, y_mixed = data_mixer(X_data, y_data, X_random, y_random, perc, perc)

                    scaler = StandardScaler()
                    X = scaler.fit_transform(X_mixed)

                    X_dl = np.expand_dims(X, axis=-1)  # shape (N, features, 1)

                    y = y_mixed.astype(np.float32)

                    input_shape = X.shape[1]

                    model_fns = {
                        'LR' :lambda:linear_regression(),
                        'RF':lambda:random_forest_reg(),
                        'MLP': lambda: NeuralNetRegressor(
                            module=MLPRegressor,
                            module__input_size=input_shape,
                            max_epochs=150,
                            batch_size=64,
                            optimizer=torch.optim.Adam,
                            lr=0.001,
                            device='cuda',
                            verbose = 0,
                            train_split=None  # Disable internal split
                        )
                        ,
                        'CNN': lambda: NeuralNetRegressor(
                            module=CNNRegressor,
                            module__input_shape=(1, input_shape),
                            max_epochs=150,
                            batch_size=64,
                            optimizer=torch.optim.Adam,
                            lr=0.001,
                            device='cuda',
                            verbose = 0,
                            train_split=None  # Disable internal split
                        ),
                        'LSTM': lambda: NeuralNetRegressor(
                            module=LSTMRegressor,
                            module__input_size=1,
                            max_epochs=150,
                            batch_size=64,
                            optimizer=torch.optim.Adam,
                            lr=0.001,
                            device='cuda',
                            verbose = 0,
                            train_split=None  # Disable internal split
                        ),
                    }

                    cv = KFold(n_splits=5, shuffle=True, random_state=1)  # Align with script (2)
                    for name, model_fn in model_fns.items():
                        tasks = [
                            delayed(run_regression_fold)(name, model_fn, X, X_dl, y, train_idx, test_idx)
                            for train_idx, test_idx in cv.split(X)
                        ]
                        results_fold = Parallel(n_jobs=2, backend='loky')(tasks)
                        mape_scores, pval_scores, preds_list, y_true_list = zip(*results_fold)
                        all_results.append({
                            'n_points': n_points,
                            'transformation': transformation,
                            'noise_percent': noise_percent,
                            'data_percentage': perc,
                            'model': name,
                            'mean_mape': np.mean(mape_scores),
                            'std_mape': np.std(mape_scores),
                            'pval': np.mean(pval_scores),
                            'last_fold_preds': preds_list[-1].tolist(),
                            'last_fold_true': y_true_list[-1].tolist()
                        })
        df = pd.DataFrame(all_results)
        df.to_csv(f'regression_results_n{n_points}.csv', index=False)

def run_classification_fold(name, model_fn, X, X_dl, y, train_idx, test_idx):
    # Choose input shape depending on model
    if name in ('MLP', 'SVC', 'RF'):
        X_train = X[train_idx].astype(np.float32)          # (N, features)
        X_test = X[test_idx].astype(np.float32)
    elif name == 'CNN':
        # CNN expects (batch, channels, length)
        X_train = np.transpose(X_dl[train_idx], (0, 2, 1)).astype(np.float32)
        X_test = np.transpose(X_dl[test_idx], (0, 2, 1)).astype(np.float32)
    elif name == 'LSTM':
        # LSTM expects (batch, seq_len, features)
        X_train = X_dl[train_idx].astype(np.float32)
        X_test = X_dl[test_idx].astype(np.float32)
    else:
        X_train = X[train_idx].astype(np.float32)
        X_test = X[test_idx].astype(np.float32)

    y_train = y[train_idx].astype(np.int64)
    y_test = y[test_idx].astype(np.int64)

    model = model_fn()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    return acc, f1, preds, y_test

def classification_experiment_run():
    for n_points in n_points_list:
        all_results = []
        for transformation in transformation_list:
            y_data_raw = y_set(balanced)['defect']
            y_test_raw = y_set(test)['defect']

            # label map built from union of labels in train and test as in TF script
            label_map = {label: i for i, label in enumerate(set(y_data_raw) | set(y_test_raw))}
            num_classes = len(label_map)

            y_data = np.array([label_map[label] for label in y_data_raw], dtype=np.int64)
            y_test_labels = np.array([label_map[label] for label in y_test_raw], dtype=np.int64)

            for noise_percent in noise_levels:
                X_data = X_set(balanced, transformation, n_points, noise_percent=noise_percent)[0]
                X_test_data = X_set(test, transformation, n_points, noise_percent=noise_percent)[0]

                for perc in damage_percentage:
                    X_mixed, y_mixed = data_mixer(X_data, y_data, X_test_data, y_test_labels, perc, perc)

                    scaler = StandardScaler()
                    X = scaler.fit_transform(X_mixed).astype(np.float32)
                    y = y_mixed.astype(np.int64)

                    # sequence variants
                    X_dl = np.expand_dims(X, axis=-1).astype(np.float32)  # (N, seq_len, features=1)
                    input_shape = X.shape[1]

                    model_fns = {
                        'SVC': lambda: svc(),
                        'RF': lambda: random_forest_clf(),
                        'MLP': lambda: NeuralNetClassifier(
                            module=MLPClassifier,
                            module__input_size=input_shape,
                            module__num_classes=num_classes,
                            max_epochs=150,
                            batch_size=64,
                            optimizer=torch.optim.Adam,
                            lr=0.001,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            verbose=0,
                            train_split=None,
                            criterion=torch.nn.CrossEntropyLoss
                        ),
                        'CNN': lambda: NeuralNetClassifier(
                            module=CNNClassifier,
                            module__input_shape=(1, input_shape),
                            module__num_classes=num_classes,
                            max_epochs=150,
                            batch_size=64,
                            optimizer=torch.optim.Adam,
                            lr=0.001,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            verbose=0,
                            train_split=None,
                            criterion=torch.nn.CrossEntropyLoss
                        ),
                        'LSTM': lambda: NeuralNetClassifier(
                            module=LSTMClassifier,
                            module__input_size=1,
                            module__num_classes=num_classes,
                            max_epochs=150,
                            batch_size=64,
                            optimizer=torch.optim.Adam,
                            lr=0.001,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            verbose=0,
                            train_split=None,
                            criterion=torch.nn.CrossEntropyLoss
                        ),
                    }

                    # Choose CV type: Stratified if enough samples per class, else plain KFold
                    class_counts = Counter(y)
                    min_class = min(class_counts.values()) if class_counts else 1
                    if min_class >= 5:
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
                        split_args = (X, y)
                    else:
                        cv = KFold(n_splits=5, shuffle=True, random_state=1)
                        split_args = (X,)

                    for name, model_fn in model_fns.items():
                        # Build tasks with correct cv.split invocation
                        tasks = [
                            delayed(run_classification_fold)(name, model_fn, X, X_dl, y, train_idx, test_idx)
                            for train_idx, test_idx in cv.split(*split_args)
                        ]
                        results_fold = Parallel(n_jobs=2, backend='loky')(tasks)
                        acc_scores, f1_scores, preds_list, y_true_list = zip(*results_fold)
                        all_results.append({
                            'n_points': n_points,
                            'transformation': transformation,
                            'noise_percent': noise_percent,
                            'data_percentage': perc,
                            'model': name,
                            'mean_acc': np.mean(acc_scores),
                            'std_acc': np.std(acc_scores),
                            'f1_macro': np.mean(f1_scores),
                            'last_fold_preds': preds_list[-1].tolist(),
                            'last_fold_true': y_true_list[-1].tolist()
                        })
        df = pd.DataFrame(all_results)
        df.to_csv(f'classification_results_n{n_points}.csv', index=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#regression_experiment_run()
classification_experiment_run()