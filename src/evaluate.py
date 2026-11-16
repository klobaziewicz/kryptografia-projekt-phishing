"""Ewaluacja modeli: metryki, dobór progu na walidacji (FPR ≤ 2%) i wykresy."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.models import MODEL_NAMES, MODELS_DIR, load_splits

MAX_FPR = 0.02
MIN_F1 = 0.92
RESULTS_PATH = Path("reports/results.csv")
FIGURES_DIR = Path("reports/figures")


def fpr_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Oblicz FPR = FP / (FP + TN); 0.0, gdy brak próbek negatywnych."""
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) else 0.0


def get_scores(model, urls: pd.Series) -> np.ndarray:
    """Zwróć ciągły wynik klasy phishing (predict_proba lub decision_function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(urls)[:, 1]
    return model.decision_function(urls)


def compute_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Policz F1, precision, recall, FPR i ROC-AUC dla reguły `score >= threshold`."""
    y_pred = (scores >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "fpr": fpr_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
    }


def select_threshold(
    y_val: np.ndarray, scores_val: np.ndarray, max_fpr: float = MAX_FPR
) -> float:
    """Wybierz próg o najlepszym F1 na walidacji spośród progów z FPR ≤ max_fpr.

    Próg `inf` (nic nie jest phishingiem) zawsze spełnia limit, więc wynik
    zawsze istnieje. Nigdy nie wywołuj tej funkcji na zbiorze testowym.
    """
    positives = int(np.sum(y_val))
    fpr, tpr, thresholds = roc_curve(y_val, scores_val, drop_intermediate=False)
    tp = tpr * positives
    fp = fpr * (len(y_val) - positives)
    f1 = 2 * tp / np.maximum(tp + fp + positives, 1)  # 2TP / (2TP + FP + FN)
    return float(thresholds[np.argmax(np.where(fpr <= max_fpr, f1, -1.0))])


def evaluate_model(
    model, val: pd.DataFrame, test: pd.DataFrame, max_fpr: float = MAX_FPR
) -> tuple[dict[str, float], np.ndarray]:
    """Dobierz próg na val, potem jednorazowo oceń model na teście.

    Zwraca wiersz wyników oraz wyniki modelu na teście (do wykresów).
    """
    y_val, y_test = val["Label"].to_numpy(), test["Label"].to_numpy()
    scores_val = get_scores(model, val["URL"].astype(str))
    threshold = select_threshold(y_val, scores_val, max_fpr)
    val_metrics = compute_metrics(y_val, scores_val, threshold)

    scores_test = get_scores(model, test["URL"].astype(str))
    test_metrics = compute_metrics(y_test, scores_test, threshold)

    row = {
        "threshold": threshold,
        "val_f1": val_metrics["f1"],
        "val_fpr": val_metrics["fpr"],
    }
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    row["meets_criteria"] = (
        test_metrics["f1"] > MIN_F1 and test_metrics["fpr"] < max_fpr
    )
    return row, scores_test


def save_plots(
    results: pd.DataFrame,
    y_test: np.ndarray,
    test_scores: dict[str, np.ndarray],
    figures_dir: Path,
    max_fpr: float = MAX_FPR,
) -> None:
    """Zapisz krzywe ROC, słupki F1/FPR i macierz pomyłek najlepszego modelu."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Krzywe ROC: pełny zakres i przybliżenie okolic progu FPR.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, x_max in zip(axes, (1.0, 0.05), strict=True):
        for name, scores in test_scores.items():
            RocCurveDisplay.from_predictions(y_test, scores, name=name, ax=ax)
        ax.axvline(max_fpr, color="gray", linestyle="--")
        ax.set(xlim=(0, x_max), aspect="auto")
        ax.set_title("Krzywe ROC (test)" if x_max == 1.0 else "Przybliżenie: FPR ≤ 5%")
    axes[1].get_legend().remove()
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    # F1 i FPR na teście - osobne panele, bo mają różne skale.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, column, target in zip(
        axes, ("test_f1", "test_fpr"), (MIN_F1, max_fpr), strict=True
    ):
        results[column].plot.barh(ax=ax)
        ax.axvline(target, color="black", linestyle="--")
        ax.set_title(f"{column} (kryterium: {target})")
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(figures_dir / "metrics_by_model.png", dpi=150)
    plt.close(fig)

    # Macierz pomyłek modelu wybranego po F1 na walidacji (nie na teście).
    best = str(results["val_f1"].idxmax())
    y_pred = (test_scores[best] >= results.loc[best, "threshold"]).astype(int)
    display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Legalne", "Phishing"], cmap="Blues"
    )
    display.ax_.set_title(f"{best} - test")
    display.figure_.savefig(figures_dir / "confusion_matrix_best.png", dpi=150)
    plt.close(display.figure_)


def evaluate_all(
    model_names: list[str] | None = None,
    models_dir: Path = MODELS_DIR,
    results_path: Path = RESULTS_PATH,
    figures_dir: Path = FIGURES_DIR,
    max_fpr: float = MAX_FPR,
) -> pd.DataFrame:
    """Oceń zapisane modele, zapisz reports/results.csv i wykresy."""
    names = [
        name
        for name in (model_names or MODEL_NAMES)
        if (models_dir / f"{name}.joblib").exists()
    ]
    if not names:
        raise FileNotFoundError(
            f"Brak modeli w {models_dir}. Uruchom: python -m src.models"
        )

    _train, val, test = load_splits()
    rows, test_scores = {}, {}
    for name in names:
        model = joblib.load(models_dir / f"{name}.joblib")
        rows[name], test_scores[name] = evaluate_model(model, val, test, max_fpr)
        print(f"{name:<24} " + " ".join(f"{k}={v:.4g}" for k, v in rows[name].items()))

    results = pd.DataFrame.from_dict(rows, orient="index").rename_axis("model")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path)
    save_plots(results, test["Label"].to_numpy(), test_scores, figures_dir, max_fpr)
    print(f"Zapisano {results_path} i wykresy w {figures_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ewaluacja modeli phishingu.")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None)
    parser.add_argument("--max-fpr", type=float, default=MAX_FPR)
    args = parser.parse_args()
    evaluate_all(model_names=args.models, max_fpr=args.max_fpr)
