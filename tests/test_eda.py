"""Testy statystyk przygotowywanych do eksploracyjnej analizy danych."""

import pandas as pd
import pytest

from src.eda import (
    class_distribution,
    punctuation_counts_by_class,
    top_tokens_by_class,
    url_lengths_by_class,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Przygotuj przykładowe legalne i phishingowe URL-e."""
    return pd.DataFrame(
        {
            "URL": ["safe.example/path", "bank-login.example/secure"],
            "Label": [0, 1],
        }
    )


def test_class_distribution_names_both_classes(sample_data: pd.DataFrame) -> None:
    """Rozkład pokazuje liczebność legalnych i phishingowych URL-i."""
    assert class_distribution(sample_data).to_dict() == {"Legalne": 1, "Phishing": 1}


def test_url_lengths_and_punctuation_are_calculated_by_class(
    sample_data: pd.DataFrame,
) -> None:
    """Długości i średnie liczby znaków są obliczane dla właściwej klasy."""
    lengths = url_lengths_by_class(sample_data)
    punctuation = punctuation_counts_by_class(sample_data)

    assert lengths["Długość URL"].tolist() == [len(url) for url in sample_data["URL"]]
    assert punctuation.loc["Legalne"].to_dict() == {"Kropki": 1.0, "Myślniki": 0.0}
    assert punctuation.loc["Phishing"].to_dict() == {"Kropki": 1.0, "Myślniki": 1.0}


def test_top_tokens_are_separated_by_class_and_limited(
    sample_data: pd.DataFrame,
) -> None:
    """Tokeny są liczone osobno i ograniczone do wskazanej liczby."""
    result = top_tokens_by_class(sample_data, limit=2)

    assert result.groupby("Label").size().to_dict() == {"Legalne": 2, "Phishing": 2}
    assert set(result.loc[result["Label"].eq("Phishing"), "Token"]) == {
        "bank",
        "example",
    }


def test_top_tokens_requires_positive_limit(sample_data: pd.DataFrame) -> None:
    """Niepoprawny limit tokenów zgłasza czytelny błąd."""
    with pytest.raises(ValueError, match="większy od zera"):
        top_tokens_by_class(sample_data, limit=0)
