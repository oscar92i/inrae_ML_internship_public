from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from utils._npy_manipulation import finetune_split, flatten_data
import numpy as np
import torch

def evaluate_RF_baseline(data, metadata, n_splits=5):
    results = {subset: {"acc": [], "f1": []} for subset in [5, 10, 20, 40]}

    for len_subset in [5, 10, 20, 40]:
        for seed in range(n_splits):
            print(f"\n===== RF Split {seed+1}/{n_splits} | subset={len_subset} =====")
            torch.manual_seed(seed)
            np.random.seed(seed)

            X_train, y_train, X_test, y_test = finetune_split(data, metadata, len_subset, seed=seed)

            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=seed,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred)

            results[len_subset]["acc"].append(acc)
            results[len_subset]["f1"].append(f1)

            print(f"[RF Test] accuracy: {acc:.3f} | f1: {f1:.3f}")

    print("\n=== Random Forest Baseline Results ===")
    for subset, metrics in results.items():
        print(f"Subset {subset:>2}: "
              f"Accuracy {np.mean(metrics['acc']):.3f} ± {np.std(metrics['acc']):.3f} | "
              f"F1 {np.mean(metrics['f1']):.3f} ± {np.std(metrics['f1']):.3f}")

    return results
