# Model zagrożeń

## 1. Profil przeciwnika

Przeciwnik aktywny (Active Attacker) tworzy złośliwe domeny podszywające się pod banki i modyfikuje adresy URL — na przykład ich ścieżki lub parametry — aby ominąć filtr phishingowy.

## 2. Interfejs dostępny dla przeciwnika

Filtr otrzymuje wyłącznie surowy URL, a przeciwnik może przygotowywać i wysyłać URL-e do oceny. Zgodnie z zasadą Kerckhoffsa zakładamy, że zna model oraz wykorzystywane cechy. Nie zna natomiast danych treningowych.

## 3. Scenariusz

System działa jako bramka pocztowa (Email Security Gateway) w przedsiębiorstwie. Analizuje URL-e z wiadomości i ma pomagać w blokowaniu wiadomości zawierających phishing, zanim trafią do pracowników.

## 4. Ilorazowe kryterium sukcesu

Wymagane wyniki ewaluacji: **F1-Score > 0,92** oraz **FPR < 2%**. FPR oznacza odsetek prawidłowych (niephishingowych) URL-i błędnie oznaczonych jako phishing: `FP / (FP + TN)`.

## 5. Założenia i ograniczenia

- Klasyfikacja wykorzystuje wyłącznie URL-e; nie analizuje treści ani zachowania strony internetowej.
- Zbiorem danych jest Kaggle `taruntiwarihp/phishing-site-urls`.
- Wyniki zależą od reprezentatywności zbioru danych i nie gwarantują odporności na nieznane techniki omijania filtra.
- Kryteria metryk należy sprawdzać na danych testowych niewykorzystywanych do trenowania ani doboru progu decyzyjnego.