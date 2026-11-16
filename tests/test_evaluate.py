"""Testy metryk, doboru progu na walidacji i pełnej ewaluacji na małych danych."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import evaluate
from src.evaluate import (
    compute_metrics,
    evaluate_model,
    fpr_score,
    get_scores,
    select_threshold,
)
from src.models import get_models, save_model

PHISHING = [
    "http://secure-login.bank.example/account/verify",
    "http://10.0.0.7/paypal/confirm/password",
    "http://signin.account.example/update",
    "http://verify-bank.example/login.php",
]
LEGIT = [
    "https://news.example/article/1",
    "https://docs.example/guide/intro",
    "https://shop.example/products/42",
    "https://wiki.example/page/history",
]


def _toy_frame(repeat: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "URL": (PHISHING + LEGIT) * repeat,
            "Label": ([1] * len(PHISHING) + [0] * len(LEGIT)) * repeat,
        }
    )


def test_fpr_score_counts_false_positives_over_negatives() -> None:
    """FPR = FP / (FP + TN)."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 0, 1, 0])
    assert fpr_score(y_true, y_pred) == pytest.approx(0.25)


def test_fpr_score_without_negatives_is_zero() -> None:
    """Brak próbek negatywnych nie powoduje dzielenia przez zero."""
    assert fpr_score(np.array([1, 1]), np.array([1, 0])) == 0.0


def test_compute_metrics_uses_threshold() -> None:
    """Metryki liczone są dla reguły score >= threshold."""
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    metrics = compute_metrics(y_true, scores, threshold=0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["fpr"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)


def test_select_threshold_respects_max_fpr() -> None:
    """Próg z val trzyma FPR ≤ limitu, nawet kosztem F1."""
    rng = np.random.default_rng(42)
    y = np.array([0] * 900 + [1] * 100)
    scores = np.concatenate([rng.normal(0.0, 1.0, 900), rng.normal(2.0, 1.0, 100)])

    threshold = select_threshold(y, scores, max_fpr=0.02)
    metrics = compute_metrics(y, scores, threshold)
    assert metrics["fpr"] <= 0.02
    assert metrics["recall"] > 0

    loose = compute_metrics(y, scores, select_threshold(y, scores, max_fpr=1.0))
    assert loose["f1"] >= metrics["f1"]


def test_select_threshold_perfect_separation() -> None:
    """Przy idealnym rozdzieleniu klas próg daje F1 = 1 i FPR = 0."""
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([-2.0, -1.0, -0.5, 1.0, 3.0])
    metrics = compute_metrics(y, scores, select_threshold(y, scores, max_fpr=0.0))
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["fpr"] == 0.0


def test_get_scores_supports_svm_and_nb() -> None:
    """LinearSVC daje decision_function, NB — prawdopodobieństwa."""
    train = _toy_frame(5)
    urls = train["URL"].astype(str)
    for name in ("svm_word", "nb_word"):
        model = get_models()[name].fit(urls, train["Label"])
        scores = get_scores(model, urls)
        assert scores.shape == (len(train),)
        assert np.isfinite(scores).all()


def test_evaluate_model_returns_val_threshold_and_test_metrics() -> None:
    """Wiersz wyników zawiera próg z val oraz metryki testu."""
    train, val, test = _toy_frame(5), _toy_frame(2), _toy_frame(2)
    model = get_models()["svm_char"].fit(train["URL"], train["Label"])
    row, scores = evaluate_model(model, val, test)
    assert len(scores) == len(test)
    for key in (
        "threshold",
        "val_f1",
        "test_f1",
        "test_fpr",
        "test_roc_auc",
        "meets_criteria",
    ):
        assert key in row
    assert row["val_fpr"] <= evaluate.MAX_FPR


def test_evaluate_all_writes_results_and_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pełna ewaluacja zapisuje results.csv i trzy wykresy; pomija brakujące modele."""
    train, val, test = _toy_frame(5), _toy_frame(2), _toy_frame(2)
    models_dir = tmp_path / "models"
    for name in ("baseline_dummy", "nb_word", "svm_word"):
        model = get_models()[name].fit(train["URL"], train["Label"])
        save_model(model, name, models_dir)
    monkeypatch.setattr(evaluate, "load_splits", lambda: (train, val, test))

    results = evaluate.evaluate_all(
        models_dir=models_dir,
        results_path=tmp_path / "results.csv",
        figures_dir=tmp_path / "figures",
    )

    assert list(results.index) == ["baseline_dummy", "nb_word", "svm_word"]
    saved = pd.read_csv(tmp_path / "results.csv", index_col="model")
    assert set(saved.index) == set(results.index)
    for figure in (
        "roc_curves.png",
        "metrics_by_model.png",
        "confusion_matrix_best.png",
    ):
        assert (tmp_path / "figures" / figure).stat().st_size > 0


def test_evaluate_all_without_models_raises(tmp_path: Path) -> None:
    """Brak plików .joblib kończy się czytelnym błędem."""
    with pytest.raises(FileNotFoundError):
        evaluate.evaluate_all(models_dir=tmp_path / "empty")
