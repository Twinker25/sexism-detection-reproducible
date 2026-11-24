import os
import random
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def normalize_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()

def load_and_prepare_data(data_dir: Path):
    raw_train = pd.read_csv(data_dir / "train.csv")
    raw_dev = pd.read_csv(data_dir / "dev.csv")
    raw_test = pd.read_csv(data_dir / "test.csv")

    df = raw_train.copy()
    df["key"] = normalize_key(df["text"])

    conflict_keys = df.groupby("key")["label"].nunique()
    conflict_keys = set(conflict_keys[conflict_keys > 1].index)
    df = df[~df["key"].isin(conflict_keys)]
    df = df.drop_duplicates(subset="key", keep="first")
    df = df[~df["key"].isin(set(normalize_key(raw_dev["text"])) | set(normalize_key(raw_test["text"])))]
    df = df.reset_index(drop=True)

    X_all, y_all = df["text"], df["label"].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=SEED
    )

    test_df = raw_test.drop_duplicates(subset="text").reset_index(drop=True)
    X_test, y_test = test_df["text"], test_df["label"].to_numpy()

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_train,
        y_val,
        X_test,
        y_test,
    )

def build_pipeline(k=None, C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("select", SelectKBest(chi2, k=k) if k else "passthrough"),
        ("clf", LogisticRegression(C=C, max_iter=1000, random_state=SEED)),
    ])

def run_experiment(run_name: str, k, C: float, data_tuple) -> None:
    X_train, X_val, y_train, y_val, X_test, y_test = data_tuple

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "TF-IDF + LogisticRegression")
        mlflow.log_param("k_features", "all" if k is None else k)
        mlflow.log_param("C", C)
        mlflow.log_param("random_state", SEED)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))

        pipe = build_pipeline(k=k, C=C)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        val_pred = pipe.predict(X_val)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        test_pred = pipe.predict(X_test)
        test_prob = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "train_time_sec": round(train_time, 4),
            "val_f1_sexist": f1_score(y_val, val_pred, pos_label=1),
            "val_macro_f1": f1_score(y_val, val_pred, average="macro"),
            "val_precision": precision_score(y_val, val_pred, pos_label=1),
            "val_recall": recall_score(y_val, val_pred, pos_label=1),
            "val_accuracy": accuracy_score(y_val, val_pred),
            "val_roc_auc": roc_auc_score(y_val, val_prob),
            "test_f1_sexist": f1_score(y_test, test_pred, pos_label=1),
            "test_macro_f1": f1_score(y_test, test_pred, average="macro"),
            "test_roc_auc": roc_auc_score(y_test, test_prob),
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print(
            f"[{run_name}] T_train={train_time:.3f}s | "
            f"Val F1={metrics['val_f1_sexist']:.4f} | "
            f"Test F1={metrics['test_f1_sexist']:.4f}"
        )

if __name__ == "__main__":
    set_seed(SEED)
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("sexism-detection-reproducibility")

    data = load_and_prepare_data(data_dir)
    run_experiment("baseline_M0", k=None, C=1.0, data_tuple=data)
    run_experiment("optimized_M1", k=2000, C=10.0, data_tuple=data)