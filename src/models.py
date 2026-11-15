"""Modele bazowe, Naive Bayes, SVM oraz strojenie hiperparametrów."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from src.features import CombinedURLFeatures, StructuralTransformer, build_tfidf

RANDOM_STATE = 42
PROCESSED_DATA_PATH = Path("data/processed")
MODELS_DIR = Path("models")
BEST_PARAMS_PATH = Path("reports/best_params.json")

MODEL_NAMES = (
    "baseline_dummy",
    "baseline_structural_lr",
    "nb_char",
    "nb_word",
    "svm_char",
    "svm_word",
    "svm_combined",
)


def get_models() -> dict[str, Pipeline]:
    """Zwróć słownik pipeline'ów modeli (fabryka)."""
    return {
        "baseline_dummy": Pipeline(
            [
                ("structural", StructuralTransformer()),
                (
                    "clf",
                    DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "baseline_structural_lr": Pipeline(
            [
                ("structural", StructuralTransformer()),
                ("scaler", MaxAbsScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "nb_char": Pipeline(
            [
                ("tfidf", build_tfidf("char")),
                ("clf", MultinomialNB()),
            ]
        ),
        "nb_word": Pipeline(
            [
                ("tfidf", build_tfidf("word")),
                ("clf", MultinomialNB()),
            ]
        ),
        "svm_char": Pipeline(
            [
                ("tfidf", build_tfidf("char")),
                (
                    "clf",
                    LinearSVC(random_state=RANDOM_STATE, max_iter=5000, dual="auto"),
                ),
            ]
        ),
        "svm_word": Pipeline(
            [
                ("tfidf", build_tfidf("word")),
                (
                    "clf",
                    LinearSVC(random_state=RANDOM_STATE, max_iter=5000, dual="auto"),
                ),
            ]
        ),
        "svm_combined": Pipeline(
            [
                ("features", CombinedURLFeatures(kind="char")),
                (
                    "clf",
                    LinearSVC(random_state=RANDOM_STATE, max_iter=5000, dual="auto"),
                ),
            ]
        ),
    }


PARAM_GRIDS: dict[str, dict] = {
    "baseline_dummy": {},
    "baseline_structural_lr": {"clf__C": [0.1, 1.0, 10.0]},
    "nb_char": {
        "tfidf__ngram_range": [(3, 4), (3, 5)],
        "clf__alpha": [0.1, 0.5, 1.0],
    },
    "nb_word": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__alpha": [0.1, 0.5, 1.0],
    },
    "svm_char": {
        "tfidf__ngram_range": [(3, 4), (3, 5)],
        "clf__C": [0.1, 1.0, 10.0],
    },
    "svm_word": {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    },
    "svm_combined": {
        "features__ngram_range": [(3, 4), (3, 5)],
        "clf__C": [0.1, 1.0, 10.0],
    },
}


def load_splits(
    processed_dir: Path = PROCESSED_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wczytaj zapisane zbiory train/val/test."""
    train = pd.read_csv(processed_dir / "train.csv")
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    return train, val, test


def sample_split(
    df: pd.DataFrame, n: int | None, seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """Losowa próbka stratyfikowana (lub cały zbiór, gdy n jest None / za duże)."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    sampled, _ = train_test_split(
        df,
        train_size=n,
        random_state=seed,
        stratify=df["Label"],
    )
    return sampled.reset_index(drop=True)


def tune(
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    n_jobs: int = -1,
) -> tuple[Pipeline, dict]:
    """Dobierz hiperparametry na val (PredefinedSplit), dopasuj finalny model na train."""
    if model_name not in get_models():
        raise ValueError(f"Nieznany model: {model_name}")

    pipeline = get_models()[model_name]
    param_grid = PARAM_GRIDS[model_name]
    x_train = train["URL"].astype(str)
    y_train = train["Label"].astype(int)
    x_val = val["URL"].astype(str)
    y_val = val["Label"].astype(int)

    if not param_grid:
        model = clone(pipeline)
        model.fit(x_train, y_train)
        return model, {}

    x_all = pd.concat([x_train, x_val], ignore_index=True)
    y_all = pd.concat([y_train, y_val], ignore_index=True)
    test_fold = np.array([-1] * len(x_train) + [0] * len(x_val))
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=PredefinedSplit(test_fold),
        n_jobs=n_jobs,
        refit=False,
    )
    search.fit(x_all, y_all)

    best_params = dict(search.best_params_)
    model = clone(pipeline).set_params(**best_params)
    model.fit(x_train, y_train)
    return model, best_params


def save_model(model: Pipeline, model_name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Zapisz model do models/<name>.joblib."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{model_name}.joblib"
    joblib.dump(model, path)
    return path


def save_best_params(
    all_params: dict[str, dict], path: Path = BEST_PARAMS_PATH
) -> None:
    """Zapisz najlepsze hiperparametry do JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(value):
        if isinstance(value, dict):
            return {str(key): _serialize(val) for key, val in value.items()}
        if isinstance(value, tuple):
            return list(value)
        return value

    path.write_text(
        json.dumps(_serialize(all_params), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def train_all(
    model_names: list[str] | None = None,
    *,
    sample: int | None = 50_000,
    n_jobs: int = -1,
) -> dict[str, dict]:
    """Wytreinuj wybrane modele, zapisz artefakty i zwróć najlepsze parametry."""
    train, val, _test = load_splits()
    train_s = sample_split(train, sample)
    val_s = sample_split(val, None if sample is None else max(1, sample // 4))

    names = list(model_names) if model_names else list(MODEL_NAMES)
    print(f"Trening na sample train={len(train_s)}, val={len(val_s)}")
    print(f"Modele: {', '.join(names)}")

    all_params: dict[str, dict] = {}
    for name in names:
        print(f"\n=== {name} ===")
        model, best_params = tune(name, train_s, val_s, n_jobs=n_jobs)
        path = save_model(model, name)
        all_params[name] = best_params
        print(f"best_params={best_params}")
        print(f"zapisano: {path}")

    save_best_params(all_params)
    print(f"\nParametry: {BEST_PARAMS_PATH}")
    return all_params


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening i strojenie modeli phishingu.")
    parser.add_argument(
        "--sample",
        type=int,
        default=50_000,
        help="Liczba wierszy train (stratyfikowana). 0 = cały zbiór.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODEL_NAMES),
        help="Podzbiór modeli (domyślnie wszystkie).",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Wątki GridSearchCV.")
    return parser.parse_args()


def main() -> None:
    """Punkt wejścia CLI (osobna funkcja, żeby joblib widział moduł src.models)."""
    args = _parse_args()
    sample = None if args.sample == 0 else args.sample
    train_all(model_names=args.models, sample=sample, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
