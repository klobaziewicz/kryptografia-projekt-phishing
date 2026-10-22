"""Funkcje przygotowujące statystyki do eksploracyjnej analizy URL-i."""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

CLASS_NAMES = {0: "Legalne", 1: "Phishing"}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Zwróć liczebność klas z polskimi nazwami."""
    return (
        df["Label"]
        .map(CLASS_NAMES)
        .value_counts()
        .reindex(CLASS_NAMES.values(), fill_value=0)
    )


def url_lengths_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Przygotuj długość każdego URL-a do porównania klas."""
    return pd.DataFrame(
        {
            "Label": df["Label"].map(CLASS_NAMES),
            "Długość URL": df["URL"].str.len(),
        }
    )


def top_tokens_by_class(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Zwróć najczęstsze tokeny URL-i osobno dla obu klas."""
    if limit < 1:
        raise ValueError("Limit tokenów musi być większy od zera.")

    rows: list[dict[str, str | int]] = []
    for label, class_name in CLASS_NAMES.items():
        token_counts: Counter[str] = Counter()
        urls = df.loc[df["Label"].eq(label), "URL"].dropna()
        for url in urls:
            token_counts.update(TOKEN_PATTERN.findall(url.lower()))

        rows.extend(
            {"Label": class_name, "Token": token, "Liczba": count}
            for token, count in sorted(
                token_counts.items(), key=lambda item: (-item[1], item[0])
            )[:limit]
        )
    return pd.DataFrame(rows, columns=["Label", "Token", "Liczba"])


def punctuation_counts_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Zwróć średnią liczbę kropek i myślników na URL w każdej klasie."""
    counts = pd.DataFrame(
        {
            "Label": df["Label"].map(CLASS_NAMES),
            "Kropki": df["URL"].str.count(r"\."),
            "Myślniki": df["URL"].str.count("-"),
        }
    )
    return (
        counts.groupby("Label")[["Kropki", "Myślniki"]]
        .mean()
        .reindex(CLASS_NAMES.values(), fill_value=0)
    )


def ensure_figures_dir(path: str | Path = "reports/figures") -> Path:
    """Utwórz katalog na wykresy, jeśli jeszcze nie istnieje."""
    figures_dir = Path(path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir
