"""Testy pipeline'u przygotowania danych."""

import pandas as pd
import pytest

from src.data import clean, normalize_url, split


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Przygotuj dane o znanym rozkładzie klas."""
    return pd.DataFrame(
        {
            "URL": [f"https://example-{i}.com" for i in range(320)],
            "Label": ["good"] * 240 + ["bad"] * 80,
        }
    )


def test_normalize_url_strips_scheme_www_and_trailing_slash() -> None:
    """Normalizacja sprowadza warianty tego samego URL-a do jednego klucza."""
    assert normalize_url(" HTTPS://WWW.Example.com/Login/ ") == "example.com/login"
    assert normalize_url("http://example.com/login") == "example.com/login"


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

    result, stats = clean(data, return_stats=True)

    assert result["URL"].is_unique
    assert result.to_dict("records") == [
        {"URL": "https://a.example", "Label": 1},
        {"URL": "https://b.example", "Label": 0},
    ]
    assert stats["empty_removed"] == 2
    assert stats["duplicates_removed"] == 1
    assert stats["conflicts_removed"] == 0
    assert stats["rows_out"] == 2


def test_clean_deduplicates_normalized_url_variants() -> None:
    """Warianty schematu/www/ukośnika są traktowane jako jeden URL."""
    data = pd.DataFrame(
        {
            "URL": [
                "http://Example.com/login",
                "https://www.example.com/login/",
                "https://other.example/path",
            ],
            "Label": ["good", "good", "bad"],
        }
    )

    result, stats = clean(data, return_stats=True)

    assert len(result) == 2
    assert stats["duplicates_removed"] == 1
    assert stats["conflicts_removed"] == 0
    assert set(result["Label"]) == {0, 1}


def test_clean_removes_label_conflicts_entirely() -> None:
    """URL z konfliktem etykiet (także po normalizacji) jest usuwany w całości."""
    data = pd.DataFrame(
        {
            "URL": [
                "http://bank-login.com/secure",
                "https://www.bank-login.com/secure/",
                "https://safe.example/",
                "https://safe.example",
            ],
            "Label": ["bad", "good", "good", "good"],
        }
    )

    result, stats = clean(data, return_stats=True)

    assert result.to_dict("records") == [{"URL": "https://safe.example/", "Label": 0}]
    assert stats["conflicts_removed"] == 2
    assert stats["duplicates_removed"] == 1
    assert stats["rows_out"] == 1


def test_split_has_no_shared_urls_and_preserves_class_proportions(
    sample_data: pd.DataFrame,
) -> None:
    """Podzbiory są rozłączne, a udział klas jest zachowany."""
    cleaned = clean(sample_data)
    train, val, test = split(cleaned)
    url_sets = [set(part["URL"]) for part in (train, val, test)]
    expected_proportions = cleaned["Label"].value_counts(normalize=True)

    assert url_sets[0].isdisjoint(url_sets[1])
    assert url_sets[0].isdisjoint(url_sets[2])
    assert url_sets[1].isdisjoint(url_sets[2])
    assert [len(part) / len(cleaned) for part in (train, val, test)] == pytest.approx(
        [0.70, 0.15, 0.15]
    )

    for part in (train, val, test):
        proportions = part["Label"].value_counts(normalize=True)
        assert proportions.to_dict() == pytest.approx(expected_proportions.to_dict())
