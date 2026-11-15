# Wykrywanie phishingu w URL-ach (TF-IDF + Naive Bayes + SVM)

Projekt studencki: klasyfikacja złośliwych URL-i na podstawie cech strukturalnych i TF-IDF (scikit-learn).  
Kryterium sukcesu: **F1 > 0.92** przy **FPR < 2%**.

Threat model: [`docs/threat_model.md`](docs/threat_model.md).

## Wymagania

- Python 3.12+ (testowane na 3.12 / 3.13)
- konto Kaggle (pobranie zbioru)
- Windows: PowerShell

## Instalacja

W katalogu projektu:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jeśli PowerShell zablokuje aktywację venv:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

W VS Code / Cursor: **Python: Select Interpreter** → `.venv\Scripts\python.exe`.

## Dane

1. Pobierz zbiór: [Phishing Site URLs (Kaggle)](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)
2. Rozpakuj ZIP i skopiuj CSV jako:

```text
data/raw/phishing_site_urls.csv
```

Kolumny: `URL`, `Label` (`good` / `bad`).  
`data/raw/` i `data/processed/` są w `.gitignore` — każdy członek zespołu pobiera dane lokalnie.

Sprawdzenie:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/raw/phishing_site_urls.csv'); print(df.shape); print(df.columns.tolist()); print(df['Label'].value_counts())"
```

## Uruchomienie (kolejność)

Upewnij się, że venv jest aktywny (w promptcie widać `(.venv)`).

### 1. Czyszczenie i podział danych (70 / 15 / 15)

```powershell
python -m src.data
```

Zapisuje: `data/processed/train.csv`, `val.csv`, `test.csv`.  
Wypisuje m.in. usunięte duplikaty/konflikty etykiet oraz proporcje klas.

### 2. Trening modeli (+ strojenie na val)

Szybki przebieg (domyślnie 50 tys. wierszy train):

```powershell
python -m src.models --sample 50000
```

Cały zbiór treningowy:

```powershell
python -m src.models --sample 0
```

Wybrane modele:

```powershell
python -m src.models --sample 50000 --models nb_char svm_combined
```

Artefakty:

- `models/*.joblib` (gitignore)
- `reports/best_params.json`

### 3. EDA

Notebook: `notebooks/01_eda.ipynb`  
Kernel: interpreter z `.venv`. Katalog roboczy notebooków: korzeń projektu (ustawione w `.vscode/settings.json`).

Wykresy: `reports/figures/`.

### 4. Testy

```powershell
python -m pytest
```

### 5. Ewaluacja i atak (w przygotowaniu)

Moduły `src/evaluate.py` i `src/attack.py` są zaplanowane w kolejnych krokach. Po ich dokończeniu:

```powershell
python -m src.evaluate
python -m src.attack
```

## Przydatne komendy

```powershell
# weryfikacja bibliotek
python -c "import pandas, sklearn; print(pandas.__version__, sklearn.__version__)"

# podgląd proporcji w train
python -c "import pandas as pd; t=pd.read_csv('data/processed/train.csv'); print(t.shape); print(t['Label'].value_counts(normalize=True))"

# szybki F1 na próbce val (po treningu)
python -c "import joblib, pandas as pd; from sklearn.metrics import f1_score; v=pd.read_csv('data/processed/val.csv').sample(5000, random_state=42); m=joblib.load('models/svm_combined.joblib'); print('F1', f1_score(v['Label'], m.predict(v['URL'].astype(str))))"
```

## Struktura

```text
├── data/raw/              # CSV z Kaggle (nie w git)
├── data/processed/        # train/val/test (nie w git)
├── docs/threat_model.md
├── models/                # *.joblib (nie w git)
├── notebooks/01_eda.ipynb
├── reports/
│   ├── best_params.json
│   └── figures/
├── src/
│   ├── data.py
│   ├── eda.py
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py        # TODO
│   └── attack.py          # TODO
├── tests/
├── requirements.txt
└── README.md
```

## Modele

| Nazwa                    | Opis                                           |
| ------------------------ | ---------------------------------------------- |
| `baseline_dummy`         | DummyClassifier (grupa kontrolna)              |
| `baseline_structural_lr` | regresja logistyczna na cechach strukturalnych |
| `nb_char` / `nb_word`    | TF-IDF + MultinomialNB                         |
| `svm_char` / `svm_word`  | TF-IDF + LinearSVC                             |
| `svm_combined`           | cechy strukturalne + TF-IDF char + LinearSVC   |

Klasa pozytywna: phishing (`bad` → `1`). `random_state=42`. TF-IDF fit tylko na train.
