import os
import json
import argparse
from datetime import datetime

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_iter", type=int, default=20)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    if "ProdTaken" not in train_df.columns:
        raise ValueError("ProdTaken not found in train.csv")
    if "ProdTaken" not in test_df.columns:
        raise ValueError("ProdTaken not found in test.csv")

    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"].astype(int)

    X_test = test_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"].astype(int)

    # Save feature columns for inference alignment (one-hot)
    os.makedirs(args.out_dir, exist_ok=True)
    feature_cols = list(X_train.columns)
    with open(os.path.join(args.out_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Model + RandomizedSearchCV
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 8, 12, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )

    search.fit(X_train, y_train)

    # Log all tuned parameters + results
    cv_results_df = pd.DataFrame(search.cv_results_)
    cv_results_df.to_csv(os.path.join(args.out_dir, "cv_results.csv"), index=False)

    with open(os.path.join(args.out_dir, "best_params.json"), "w") as f:
        json.dump(search.best_params_, f, indent=2)

    with open(os.path.join(args.out_dir, "best_cv_summary.json"), "w") as f:
        json.dump(
            {
                "best_cv_score_roc_auc": float(search.best_score_),
                "n_iter": int(args.n_iter),
                "cv_folds": int(cv.get_n_splits()),
                "run_timestamp": datetime.now().isoformat()
            },
            f,
            indent=2
        )

    best_model = search.best_estimator_

    # Test evaluation
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }

    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(confusion_matrix(y_test, y_pred)).to_csv(
        os.path.join(args.out_dir, "confusion_matrix.csv"), index=False
    )

    # Save best model artifact
    model_path = os.path.join(args.out_dir, "best_random_forest_model.joblib")
    joblib.dump(best_model, model_path)

    print("✅ Training complete")
    print("Best CV ROC-AUC:", search.best_score_)
    print("Best Params:", search.best_params_)
    print("Saved model:", model_path)
    print("Saved artifacts in:", args.out_dir)


if __name__ == "__main__":
    main()
