# Data Mining Studio

Interactive data-mining pipeline built with Streamlit — Mini-projet *Fouille de Données 1*, Faculté d'Informatique 2025-2026.

The app covers the full pipeline in three coordinated tabs:

| Volet | Contenu |
|------|--------|
| **Prétraitement** | Import (CSV / Excel / JSON / Parquet) · exploration (stats, types, missing) · nettoyage (drop / impute) · normalisation (Min-Max, Z-score) · visualisation (boxplot, scatter, histogram, heatmap) |
| **Clustering** | K-Means & K-Medoids **implémentés from scratch** · courbe d'Elbow · score de Silhouette · projection 2D (PCA) |
| **Classification** | Train/test split · 7 modèles (KNN, Decision Tree, Random Forest, Gradient Boosting, Logistic Regression, SVM, Naïve Bayes) · matrice de confusion · Accuracy / Precision / Recall / F1 · importance des variables |

## Quick start

```bash
make install   # create venv & install dependencies
make run       # launch Streamlit on http://localhost:8501
```

Other targets:

```bash
make help      # list every target
make dev       # run with auto-reload
make lint      # syntax check
make clean     # remove caches
```

## Project layout

```
mini-projet/
├── app.py                # Streamlit entry point
├── Makefile              # install / run / dev / lint / clean
├── requirements.txt
├── .streamlit/config.toml
└── src/
    ├── styles.py         # custom CSS + UI primitives
    ├── utils.py          # data loading, plot theming, session state
    ├── preprocessing.py  # Volet 1
    ├── clustering.py     # Volet 2 (K-Means & K-Medoids from scratch)
    └── classification.py # Volet 3
```

## Stack

Streamlit · scikit-learn · Plotly · pandas · numpy
