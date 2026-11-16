"""Testy cech strukturalnych i TF-IDF."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from src.features import (
    STRUCTURAL_COLUMNS,
    CombinedURLFeatures,
    build_tfidf,
    combine_structural_and_tfidf,
    structural_features,
    tokenize_url_words,
)


@pytest.fixture
def sample_urls() -> pd.Series:
    """Przykładowe URL-e pokrywające kluczowe cechy."""
    return pd.Series(
        [
            "https://secure-login.bank.example.com/account/verify?user=1",
            "http://192.168.0.1/paypal/password",
            "http://user@evil.example/path",
            "example.com/safe",
        ]
    )


def test_structural_features_handles_malformed_ipv6_like_url() -> None:
    """Uszkodzony URL nie przerywa ekstrakcji cech."""
    features = structural_features(pd.Series(["http://[::broken/path", "https://ok.example/"]))
    assert len(features) == 2
    assert features.loc[1, "is_https"] == 1


def test_structural_features_columns_and_values(sample_urls: pd.Series) -> None:
    """Cechy strukturalne mają oczekiwane kolumny i poprawne wartości."""
    features = structural_features(sample_urls)

    assert list(features.columns) == list(STRUCTURAL_COLUMNS)
    assert len(features) == len(sample_urls)

    first = features.iloc[0]
    assert first["url_length"] == len(sample_urls.iloc[0])
    assert first["is_https"] == 1
    assert first["has_at"] == 0
    assert first["is_ip_host"] == 0
    assert first["n_subdomains"] == 2  # secure-login.bank.example.com
    assert first["n_keywords"] >= 4  # secure, login, bank, account, verify

    second = features.iloc[1]
    assert second["is_ip_host"] == 1
    assert second["is_https"] == 0
    assert second["n_subdomains"] == 0
    assert second["n_keywords"] >= 2  # paypal, password

    third = features.iloc[2]
    assert third["has_at"] == 1


def test_tokenize_url_words_splits_on_non_alnum() -> None:
    """Tokenizer słowny dzieli URL po znakach niealfanumerycznych."""
    assert tokenize_url_words("https://Bank-Login.example/path?x=1") == [
        "https",
        "bank",
        "login",
        "example",
        "path",
        "x",
        "1",
    ]


def test_build_tfidf_char_and_word_fit_transform(sample_urls: pd.Series) -> None:
    """Oba warianty TF-IDF fitują się na train i transformują bez błędu."""
    char_vectorizer = build_tfidf("char")
    word_vectorizer = build_tfidf("word")

    char_matrix = char_vectorizer.fit_transform(sample_urls)
    word_matrix = word_vectorizer.fit_transform(sample_urls)

    assert issparse(char_matrix)
    assert issparse(word_matrix)
    assert char_matrix.shape[0] == len(sample_urls)
    assert word_matrix.shape[0] == len(sample_urls)
    assert char_matrix.shape[1] > 0
    assert word_matrix.shape[1] > 0


def test_build_tfidf_rejects_unknown_kind() -> None:
    """Nieznany kind TF-IDF kończy się błędem."""
    with pytest.raises(ValueError, match="Nieznany kind"):
        build_tfidf("bytes")


def test_combined_url_features_fit_only_on_train() -> None:
    """CombinedURLFeatures łączy skaler + TF-IDF; transform działa na nowych URL-ach."""
    train = pd.Series(
        [
            "https://login.bank.example/secure",
            "https://news.example/article",
            "http://192.168.1.10/update",
        ]
    )
    test = pd.Series(["http://verify.account.example/confirm", "safe.example/home"])

    transformer = CombinedURLFeatures(kind="char")
    train_matrix = transformer.fit_transform(train)
    test_matrix = transformer.transform(test)

    assert issparse(train_matrix)
    assert issparse(test_matrix)
    assert train_matrix.shape[0] == len(train)
    assert test_matrix.shape[0] == len(test)
    assert train_matrix.shape[1] == test_matrix.shape[1]
    assert train_matrix.shape[1] > len(STRUCTURAL_COLUMNS)


def test_combine_structural_and_tfidf_helper(sample_urls: pd.Series) -> None:
    """Funkcja pomocnicza hstackuje przeskalowane cechy z TF-IDF."""
    structural = structural_features(sample_urls)
    tfidf = build_tfidf("word").fit_transform(sample_urls)

    combined, scaler = combine_structural_and_tfidf(structural, tfidf, fit_scaler=True)
    again, _ = combine_structural_and_tfidf(
        structural, tfidf, scaler=scaler, fit_scaler=False
    )

    assert issparse(combined)
    assert combined.shape == again.shape
    assert combined.shape[0] == len(sample_urls)
    np.testing.assert_allclose(combined.toarray(), again.toarray())
