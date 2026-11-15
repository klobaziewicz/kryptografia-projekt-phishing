# Zasady projektu

- To projekt studencki do wykrywania phishingu w URL-ach z użyciem TF-IDF, Naive Bayes, SVM i scikit-learn (Python 3).
- Klasa pozytywna oznacza phishing: mapuj etykietę `bad` na `1`, a `good` na `0`.
- Umieszczaj modularny kod w `src/`; dodawaj type hints oraz krótkie docstringi po polsku. Notebooki służą do eksploracji i prezentacji — nie umieszczaj w nich logiki projektu.
- TF-IDF i każdy inny transformer dopasowuj (`fit`) wyłącznie na zbiorze treningowym, aby uniknąć data leakage. Potem używaj dopasowanego transformera do walidacji i testu.
- Ustawiaj `random_state=42` wszędzie tam, gdzie biblioteka udostępnia tę opcję.
- Raportuj metryki: F1, precision, recall, FPR (`FP / (FP + TN)`) i ROC-AUC.
- Progi decyzyjne dobieraj na zbiorze walidacyjnym; nigdy nie dobieraj ich na zbiorze testowym.
- Korzystaj wyłącznie z bibliotek wymienionych w `requirements.txt`.
