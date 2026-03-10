import numpy as np
from pprint import pprint
from collections import defaultdict
import matplotlib.pyplot as plt

import xgboost
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from itertools import product

from utils._npy_manipulation import *


def evaluate_random_forest(data, metadata, train_split_target, validation_split_target=None, n_splits=5):
    accs, f1s = [], []

    if validation_split_target is None:
        print(f'train split {train_split_target:.3g}')
        print(f'test split {1-train_split_target:.3g}')

        for seed in range(n_splits):
            print(f'\n      Split {seed+1}/{n_splits}')

            X_train, X_test, y_train, y_test = train_test_split_by_plotid(
                data, metadata, train_split_target=train_split_target, seed=seed
            )

            clf = RandomForestClassifier(
                random_state=seed, 
                n_jobs=-1)
            
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'accuracy: {acc:.3f} | f1: {f1:.3f}')

            accs.append(acc)
            f1s.append(f1)
        
    else:
        print(f'train split {train_split_target:.3g}')
        print(f'validation split {validation_split_target:.3g}')
        print(f'test split {1-train_split_target-validation_split_target:.3g}')

        # val_scores = defaultdict(list)
        # param_grid = list(product(range(50, 501, 50), [5, 10, 15, 20, None]))  # Adjust `max_depth` values as needed
        
        # for seed in range(n_splits):
        #     X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split_by_gid(
        #         data, metadata, train_split_target=train_split_target, validation_split_target=validation_split_target, seed=seed
        #     )
            
        #     min_estimators = 50
        #     max_estimators = 500
        #     step = 50 
            
        #     for n in range(min_estimators, max_estimators + 1, step):
            #     clf = RandomForestClassifier(
        #               random_state=seed, 
        #               n_jobs=-1)
        #         clf.set_params(n_estimators=n)
        #         clf.fit(X_train, y_train)
                
        #         val_preds = clf.predict(X_val)
        #         val_acc = accuracy_score(y_val, val_preds)
        #         val_scores[n].append((seed, np.round((val_acc), 4)))

        # mean_val_acc = {
        #     n: np.round(np.mean([val_acc for _, val_acc in tuples]), 4)
        #     for n, tuples in val_scores.items()
        # }

        # best_n = max(mean_val_acc, key=mean_val_acc.get)
        # best_val_acc = mean_val_acc[best_n]

        # print(f"Best n: {best_n} with mean val_acc: {best_val_acc:.4f}")

        val_scores = defaultdict(list)
        param_grid = list(product(range(50, 501, 50), [5, 10, 15, 20, None]))  # Adjust `max_depth` values as needed

        for seed in range(n_splits):
            X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split_by_plotid(
                data, metadata, train_split_target=train_split_target, validation_split_target=validation_split_target, seed=seed
            )

            for n_estimators, max_depth in param_grid:
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=seed,
                    n_jobs=-1
                )
                clf.fit(X_train, y_train)

                val_preds = clf.predict(X_val)
                val_acc = accuracy_score(y_val, val_preds)
                val_scores[(n_estimators, max_depth)].append((seed, np.round(val_acc, 4)))

        mean_val_acc = {
            params: np.round(np.mean([score for _, score in results]), 4)
            for params, results in val_scores.items()
        }

        best_params = max(mean_val_acc, key=mean_val_acc.get)
        best_n, best_depth = best_params
        best_val_acc = mean_val_acc[best_params]
        print(f"\nBest (n_estimators, max_depth): ({best_n}, {best_depth}) with mean val_acc: {best_val_acc:.4f}")

        for seed in range(n_splits):
            print(f'\n      Split {seed+1}/{n_splits}')

            X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split_by_plotid(
                data, metadata, train_split_target=train_split_target, validation_split_target=validation_split_target, seed=seed
            )

            clf = RandomForestClassifier(
                n_estimators=best_n,
                random_state=seed, 
                n_jobs=-1
            )

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'accuracy: {acc:.3f} | f1: {f1:.3f}')

            accs.append(acc)
            f1s.append(f1)

    seed = range(1, n_splits+1)
    plt.figure(figsize=(8, 5))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.plot(seed, accs, marker='o', color='r', label='accuracy')
    plt.plot(seed, f1s, marker='o', color='g', label='f1 Score')
    plt.title('RandomForest')
    plt.xlabel('splits')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print('RandomForest')
    print(f'mean accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}')
    print(f'mean f1 score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}')


def evaluate_xgboost(data, metadata, train_split_target, validation_split_target=None, n_splits=5):
    accs, f1s = [], []

    if validation_split_target is None:
        print(f'train split {train_split_target:.3g}')
        print(f'test split {1-train_split_target:.3g}')

        for seed in range(n_splits):
            print(f'\n      Split {seed+1}/{n_splits}')

            X_train, X_test, y_train, y_test = train_test_split_by_plotid(
                data, metadata, train_split_target=train_split_target, seed=seed
            )

            clf = XGBClassifier(
                n_estimators=100,
                random_state=seed,
                eval_metric='logloss',
                n_jobs=-1
            )

            clf.fit(
                X_train, y_train
            )

            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'accuracy: {acc:.3f} | f1: {f1:.3f}')

            accs.append(acc)
            f1s.append(f1)

    else:
        print(f'train split {train_split_target:.3g}')
        print(f'validation split {validation_split_target:.3g}')
        print(f'test split {1-train_split_target-validation_split_target:.3g}')

        # val_scores = defaultdict(list)
        
        # for seed in range(n_splits):
        #     X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split_by_gid(
        #         data, metadata, train_split_target=train_split_target, validation_split_target=validation_split_target, seed=seed
        #     )
   
        #     min_estimators = 50
        #     max_estimators = 500
        #     step = 50

        #     for n in range(min_estimators, max_estimators + 1, step):
            #     clf = XGBClassifier(
            #         random_state=seed,
            #         eval_metric='logloss',
            #         n_jobs=-1
            #     )
        #         clf.set_params(n_estimators=n)
        #         clf.fit(X_train, y_train)
                
        #         val_preds = clf.predict(X_val)
        #         val_acc = accuracy_score(y_val, val_preds)
        #         val_scores[n].append((seed, np.round((val_acc), 4)))

        # mean_val_acc = {
        #     n: np.round(np.mean([val_acc for _, val_acc in tuples]), 4)
        #     for n, tuples in val_scores.items()
        # }

        # best_n = max(mean_val_acc, key=mean_val_acc.get)
        # best_val_acc = mean_val_acc[best_n]
        # print(f"Best n: {best_n} with mean val_acc: {best_val_acc:.4f}")

        val_scores = defaultdict(list)

        n_estimators_range = range(50, 501, 50)
        max_depth_range = [3, 5, 7, 10, 15]
        param_grid = list(product(n_estimators_range, max_depth_range))

        for seed in range(n_splits):
            X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split_by_plotid(
                data, metadata, train_split_target=train_split_target, validation_split_target=validation_split_target, seed=seed
            )

            for n_estimators, max_depth in param_grid:
                clf = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=seed,
                    eval_metric='logloss',
                    n_jobs=-1
                )

                clf.fit(X_train, y_train)

                val_preds = clf.predict(X_val)
                val_acc = accuracy_score(y_val, val_preds)
                val_scores[(n_estimators, max_depth)].append((seed, np.round(val_acc, 4)))

        mean_val_acc = {
            params: np.round(np.mean([acc for _, acc in scores]), 4)
            for params, scores in val_scores.items()
        }

        best_params = max(mean_val_acc, key=mean_val_acc.get)
        best_n, best_depth = best_params
        best_val_acc = mean_val_acc[best_params]

        print(f"\nBest (n_estimators, max_depth): ({best_n}, {best_depth}) with mean val_acc: {best_val_acc:.4f}")


        for seed in range(n_splits):
            print(f'\n      Split {seed+1}/{n_splits}')

            X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split_by_plotid(
                data, metadata, train_split_target=train_split_target, validation_split_target=validation_split_target, seed=seed
            )

            clf = XGBClassifier(
                n_estimators=best_n,
                random_state=seed,
                eval_metric='logloss',
                n_jobs=-1,
                early_stopping_rounds=10
            )

            clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'accuracy: {acc:.3f} | f1: {f1:.3f}')

            accs.append(acc)
            f1s.append(f1)

    seed = range(1, n_splits+1)
    plt.figure(figsize=(8, 5))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.plot(seed, accs, marker='o', color='r', label='accuracy')
    plt.plot(seed, f1s, marker='o', color='g', label='f1 Score')
    plt.title('XGBoost')
    plt.xlabel('splits')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print('XGBoost')
    print(f'mean accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}')
    print(f'mean f1 score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}')
