"""Cechy strukturalne i TF-IDF dla URL-i phishingowych."""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

PHISHING_KEYWORDS = (
    "login",
    "secure",
    "verify",
    "bank",
    "update",
    "account",
    "signin",
    "confirm",
    "paypal",
    "password",
)

_WORD_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")
_IP_HOST_RE = re.compile(
    r"^(?:\d{1,3}\.){3}\d{1,3}$|"
    r"^\[?[0-9a-fA-F:]+\]?$"
)
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

STRUCTURAL_COLUMNS = (
    "url_length",
    "n_dots",
    "n_hyphens",
    "n_digits",
    "n_special",
    "has_at",
    "is_ip_host",
    "is_https",
    "n_subdomains",
    "host_length",
    "path_length",
    "n_keywords",
)


def _ensure_scheme(url: str) -> str:
    """Dodaj schemat, jeśli URL go nie ma (wymagane przez urlparse)."""
    return url if _SCHEME_RE.match(url) else f"http://{url}"


def _parse_parts(url: str) -> tuple[str, str, str]:
    """Zwróć (host, path, scheme) dla URL-a."""
    parsed = urlparse(_ensure_scheme(url))
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    scheme = (parsed.scheme or "").lower()
    return host, path, scheme


def _is_ip_host(host: str) -> bool:
    """Sprawdź, czy host wygląda na adres IPv4 lub IPv6."""
    if not host:
        return False
    return bool(_IP_HOST_RE.match(host))


def _count_subdomains(host: str) -> int:
    """Policz subdomeny (części przed domeną drugiego poziomu)."""
    if not host or _is_ip_host(host):
        return 0
    parts = [part for part in host.split(".") if part]
    return max(0, len(parts) - 2)


def _count_keywords(url: str) -> int:
    """Policz wystąpienia słów kluczowych phishingowych w URL-u."""
    lowered = url.lower()
    return sum(lowered.count(keyword) for keyword in PHISHING_KEYWORDS)


def tokenize_url_words(url: str) -> list[str]:
    """Podziel URL na tokeny po znakach niealfanumerycznych."""
    return [token for token in _WORD_SPLIT_RE.split(url.lower()) if token]


def structural_features(urls: pd.Series | Iterable[str]) -> pd.DataFrame:
    """Wyodrębnij cechy strukturalne z serii URL-i."""
    series = pd.Series(urls, dtype="string")
    rows: list[dict[str, int]] = []

    for url in series.fillna("").astype(str):
        host, path, scheme = _parse_parts(url)
        rows.append(
            {
                "url_length": len(url),
                "n_dots": url.count("."),
                "n_hyphens": url.count("-"),
                "n_digits": sum(char.isdigit() for char in url),
                "n_special": sum(not char.isalnum() for char in url),
                "has_at": int("@" in url),
                "is_ip_host": int(_is_ip_host(host)),
                "is_https": int(scheme == "https"),
                "n_subdomains": _count_subdomains(host),
                "host_length": len(host),
                "path_length": len(path),
                "n_keywords": _count_keywords(url),
            }
        )

    return pd.DataFrame(rows, columns=list(STRUCTURAL_COLUMNS))


def build_tfidf(kind: str = "char") -> TfidfVectorizer:
    """Zbuduj wektoryzator TF-IDF (znaki lub słowa URL)."""
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=200_000,
            lowercase=True,
        )
    if kind == "word":
        return TfidfVectorizer(
            analyzer="word",
            tokenizer=tokenize_url_words,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=200_000,
            lowercase=True,
        )
    raise ValueError(f"Nieznany kind TF-IDF: {kind!r} (oczekiwano 'char' lub 'word')")


class CombinedURLFeatures(BaseEstimator, TransformerMixin):
    """Połącz przeskalowane cechy strukturalne z TF-IDF w jedną macierz rzadką."""

    def __init__(self, kind: str = "char") -> None:
        self.kind = kind
        self.scaler_: MaxAbsScaler | None = None
        self.tfidf_: TfidfVectorizer | None = None

    def fit(self, urls: pd.Series | Iterable[str], y: object | None = None):
        """Dopasuj skaler i TF-IDF wyłącznie do danych treningowych."""
        series = pd.Series(list(urls), dtype="string").fillna("").astype(str)
        structural = structural_features(series)
        self.scaler_ = MaxAbsScaler().fit(structural.to_numpy(dtype=np.float64))
        self.tfidf_ = build_tfidf(self.kind).fit(series)
        return self

    def transform(self, urls: pd.Series | Iterable[str]) -> csr_matrix:
        """Zwróć macierz rzadką: [MaxAbs(strukturalne) | TF-IDF]."""
        if self.scaler_ is None or self.tfidf_ is None:
            raise RuntimeError("CombinedURLFeatures musi być najpierw dopasowany (fit).")

        series = pd.Series(list(urls), dtype="string").fillna("").astype(str)
        structural = structural_features(series).to_numpy(dtype=np.float64)
        scaled = self.scaler_.transform(structural)
        tfidf_matrix = self.tfidf_.transform(series)
        return hstack([csr_matrix(scaled), tfidf_matrix], format="csr")

    def fit_transform(
        self, urls: pd.Series | Iterable[str], y: object | None = None, **fit_params
    ) -> csr_matrix:
        """Dopasuj i przekształć w jednym kroku (tylko train)."""
        return self.fit(urls, y).transform(urls)


def combine_structural_and_tfidf(
    structural: pd.DataFrame | np.ndarray,
    tfidf_matrix: csr_matrix,
    scaler: MaxAbsScaler | None = None,
    *,
    fit_scaler: bool = False,
) -> tuple[csr_matrix, MaxAbsScaler]:
    """Złącz cechy strukturalne (MaxAbs) z macierzą TF-IDF."""
    values = (
        structural.to_numpy(dtype=np.float64)
        if isinstance(structural, pd.DataFrame)
        else np.asarray(structural, dtype=np.float64)
    )
    if scaler is None:
        scaler = MaxAbsScaler()
        fit_scaler = True
    if fit_scaler:
        scaled = scaler.fit_transform(values)
    else:
        scaled = scaler.transform(values)
    return hstack([csr_matrix(scaled), tfidf_matrix], format="csr"), scaler
