"""Ładowanie, czyszczenie i podział danych URL-i phishingowych."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("data/raw/phishing_site_urls.csv")
PROCESSED_DATA_PATH = Path("data/processed")
REQUIRED_COLUMNS = ["URL", "Label"]


def _require_columns(df: pd.DataFrame) -> None:
    """Sprawdź, czy ramka zawiera wymagane kolumny."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Brak wymaganych kolumn: {', '.join(sorted(missing))}")


def load_raw(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Wczytaj surowy plik CSV z kolumnami URL i Label."""
    df = pd.read_csv(path)
    _require_columns(df)
    return df.loc[:, REQUIRED_COLUMNS].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Oczyść URL-e, usuń puste i powielone wiersze oraz zakoduj etykiety."""
    _require_columns(df)
    cleaned = df.loc[:, REQUIRED_COLUMNS].copy()
    cleaned["URL"] = cleaned["URL"].astype("string").str.strip()
    labels = cleaned["Label"].astype("string").str.strip().str.lower()

    valid_rows = (
        cleaned["URL"].notna() & cleaned["URL"].ne("") & labels.notna() & labels.ne("")
    )
    cleaned = cleaned.loc[valid_rows].copy()
    labels = labels.loc[valid_rows]

    mapped_labels = labels.map({"bad": 1, "good": 0})
    if mapped_labels.isna().any():
        invalid_labels = sorted(labels.loc[mapped_labels.isna()].unique().tolist())
        raise ValueError(f"Nieznane etykiety: {invalid_labels}")

    cleaned["Label"] = mapped_labels.astype(int)
    cleaned = cleaned.drop_duplicates(subset="URL", keep="first")
    return cleaned.reset_index(drop=True)


def split(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Podziel dane z zachowaniem proporcji klas na zbiory 70/15/15."""
    _require_columns(df)
    train, temporary = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["Label"],
    )
    val, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=seed,
        stratify=temporary["Label"],
    )
    return tuple(part.reset_index(drop=True) for part in (train, val, test))


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path = PROCESSED_DATA_PATH,
) -> None:
    """Zapisz podzielone zbiory jako pliki CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path / "train.csv", index=False)
    val.to_csv(output_path / "val.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)


def _print_split_summary(name: str, df: pd.DataFrame) -> None:
    """Wypisz liczebność zbioru i udziały obu klas."""
    proportions = df["Label"].value_counts(normalize=True).reindex([0, 1], fill_value=0)
    print(
        f"{name}: {len(df)} wierszy | "
        f"good (0): {proportions[0]:.2%} | phishing (1): {proportions[1]:.2%}"
    )


if __name__ == "__main__":
    data = clean(load_raw())
    train_df, val_df, test_df = split(data)
    save_splits(train_df, val_df, test_df)
    for split_name, split_df in (
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ):
        _print_split_summary(split_name, split_df)
