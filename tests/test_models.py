"""Testy fabryki modeli i strojenia na małej próbce."""

import json
from pathlib import Path

import pandas as pd

from src.models import MODEL_NAMES, get_models, save_best_params, tune


def _toy_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "URL": [
                "https://secure-login.bank.example/account",
                "https://verify.paypal.example/confirm",
                "https://news.example/article/1",
                "https://docs.example/guide",
                "http://192.168.0.1/password/update",
                "https://shop.example/products",
                "https://signin.account.example/secure",
                "https://wiki.example/page",
            ]
            * 5,
            "Label": [1, 1, 0, 0, 1, 0, 1, 0] * 5,
        }
    )
    val = pd.DataFrame(
        {
            "URL": [
                "https://bank.example/login",
                "https://example.org/about",
                "http://10.0.0.2/secure/update",
                "https://blog.example/post",
            ]
            * 3,
            "Label": [1, 0, 1, 0] * 3,
        }
    )
    return train, val


def test_get_models_contains_expected_pipelines() -> None:
    """Fabryka zwraca wszystkie wymagane modele jako Pipeline."""
    models = get_models()
    assert tuple(models) == MODEL_NAMES
    for name, pipeline in models.items():
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict")
        assert name in models


def test_tune_dummy_and_nb_on_toy_data(tmp_path: Path) -> None:
    """Strojenie działa na małym zbiorze; finalny model przewiduje etykiety."""
    train, val = _toy_splits()

    dummy, dummy_params = tune("baseline_dummy", train, val, n_jobs=1)
    nb, nb_params = tune("nb_word", train, val, n_jobs=1)

    assert dummy_params == {}
    assert "clf__alpha" in nb_params or "tfidf__ngram_range" in nb_params
    preds = nb.predict(val["URL"].astype(str))
    assert len(preds) == len(val)
    assert set(preds).issubset({0, 1})

    save_best_params({"baseline_dummy": dummy_params, "nb_word": nb_params}, tmp_path / "p.json")
    loaded = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert "nb_word" in loaded
