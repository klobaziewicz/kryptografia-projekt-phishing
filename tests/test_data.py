"""Testy pipeline'u przygotowania danych."""

import pandas as pd
import pytest

from src.data import clean, split


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Przygotuj dane o znanym rozkładzie klas."""
    return pd.DataFrame(
        {
            "URL": [f"https://example-{i}.com" for i in range(320)],
            "Label": ["good"] * 240 + ["bad"] * 80,
        }
    )


def test_clean_removes_duplicate_urls_and_empty_rows() -> None:
    """Czyszczenie usuwa powtórzenia i puste wiersze oraz mapuje etykiety."""
    data = pd.DataFrame(
        {
            "URL": [
                " https://a.example ",
                "https://a.example",
                "https://b.example",
                "",
                None,
            ],
            "Label": ["bad", "bad", "good", "good", "good"],
        }
    )

    result = clean(data)

    assert result["URL"].is_unique
    assert result.to_dict("records") == [
        {"URL": "https://a.example", "Label": 1},
        {"URL": "https://b.example", "Label": 0},
    ]


def test_split_has_no_shared_urls_and_preserves_class_proportions(
    sample_data: pd.DataFrame,
) -> None:
    """Podzbiory są rozłączne, a udział klas jest zachowany."""
    train, val, test = split(sample_data)
    url_sets = [set(part["URL"]) for part in (train, val, test)]
    expected_proportions = sample_data["Label"].value_counts(normalize=True)

    assert url_sets[0].isdisjoint(url_sets[1])
    assert url_sets[0].isdisjoint(url_sets[2])
    assert url_sets[1].isdisjoint(url_sets[2])
    assert [
        len(part) / len(sample_data) for part in (train, val, test)
    ] == pytest.approx([0.70, 0.15, 0.15])

    for part in (train, val, test):
        proportions = part["Label"].value_counts(normalize=True)
        assert proportions.to_dict() == pytest.approx(expected_proportions.to_dict())
