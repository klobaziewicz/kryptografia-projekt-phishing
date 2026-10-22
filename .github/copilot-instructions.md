Projekt: wykrywanie phishingu w URL-ach (TF-IDF + Naive Bayes + SVM), scikit-learn.
- Klasa pozytywna = phishing (label "bad" -> 1).
- Kod modularny w src/, type hints, docstringi, bez logiki w notebookach.
- TF-IDF fitować TYLKO na zbiorze treningowym (zakaz data leakage).
- random_state=42 wszędzie.
- Metryki: F1, precision, recall, FPR, ROC-AUC.