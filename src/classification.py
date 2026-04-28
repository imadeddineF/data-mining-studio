"""Volet 3 — Supervised learning: train/test split, models, metrics, confusion matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .styles import section, empty_state
from .utils import PALETTE, numeric_columns, style_fig


MODELS = {
    "K-Nearest Neighbors": lambda p: KNeighborsClassifier(n_neighbors=p["k"]),
    "Decision Tree":       lambda p: DecisionTreeClassifier(max_depth=p["max_depth"], random_state=42),
    "Random Forest":       lambda p: RandomForestClassifier(n_estimators=p["n_estimators"],
                                                            max_depth=p["max_depth"], random_state=42),
    "Gradient Boosting":   lambda p: GradientBoostingClassifier(n_estimators=p["n_estimators"], random_state=42),
    "Logistic Regression": lambda p: LogisticRegression(max_iter=2000, C=p["C"]),
    "SVM":                 lambda p: SVC(C=p["C"], kernel=p["kernel"], probability=False, random_state=42),
    "Gaussian Naïve Bayes": lambda p: GaussianNB(),
}


# ---------------------------------------------------------------------------
def _hyperparams(name: str) -> dict:
    p: dict = {}
    if name == "K-Nearest Neighbors":
        p["k"] = st.slider("Nombre de voisins (k)", 1, 30, 5, key="cls_knn_k")
    elif name in ("Decision Tree", "Random Forest"):
        depth = st.slider("Profondeur max (0 = illimité)", 0, 30, 6, key=f"cls_{name}_depth")
        p["max_depth"] = None if depth == 0 else depth
        if name == "Random Forest":
            p["n_estimators"] = st.slider("Nombre d'arbres", 10, 500, 150, step=10, key="cls_rf_trees")
    elif name == "Gradient Boosting":
        p["n_estimators"] = st.slider("Nombre d'estimateurs", 50, 500, 150, step=10, key="cls_gb_n")
    elif name == "Logistic Regression":
        p["C"] = st.slider("C (régularisation inverse)", 0.01, 10.0, 1.0, key="cls_lr_C")
    elif name == "SVM":
        p["C"] = st.slider("C", 0.01, 10.0, 1.0, key="cls_svm_C")
        p["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="cls_svm_kernel")
    return p


def _confusion_fig(cm: np.ndarray, classes: list[str]) -> go.Figure:
    fig = px.imshow(
        cm, x=classes, y=classes, text_auto=True,
        color_continuous_scale=["#0E1117", "#312E81", "#7C7CFF", "#A78BFA"],
        labels=dict(x="Prédit", y="Réel", color="Effectif"),
    )
    fig.update_xaxes(side="bottom")
    return style_fig(fig, 460)


def _metric_card(label: str, value: float) -> None:
    st.metric(label, f"{value:.4f}")


# ---------------------------------------------------------------------------
def render() -> None:
    if st.session_state.df is None:
        empty_state("🤖", "Aucune donnée à entraîner",
                    "Importez d'abord un dataset depuis l'onglet Prétraitement.")
        return

    df: pd.DataFrame = st.session_state.df
    if df.shape[1] < 2:
        st.warning("Au moins 2 colonnes (features + target) sont nécessaires.")
        return

    # ------------------------------------------------------------------ config
    section("Configuration")

    cfg = st.columns([1, 1, 1, 1])
    with cfg[0]:
        target = st.selectbox("Variable cible", df.columns.tolist(),
                              index=len(df.columns) - 1, key="cls_target")
    with cfg[1]:
        test_size = st.slider("Taille de test", 0.1, 0.5, 0.25, 0.05, key="cls_test_size")
    with cfg[2]:
        seed = st.number_input("Random state", min_value=0, value=42, step=1, key="cls_seed")
    with cfg[3]:
        scale = st.checkbox("Standardiser les features", value=True, key="cls_scale")

    feature_pool = [c for c in numeric_columns(df) if c != target]
    if not feature_pool:
        st.warning("Aucune feature numérique disponible (la cible exclue).")
        return

    features = st.multiselect("Variables explicatives", feature_pool,
                              default=feature_pool, key="cls_features")
    if not features:
        st.info("Sélectionnez au moins une feature.")
        return

    # ------------------------------------------------------------------ model
    section("Modèle")
    mc = st.columns([1, 2])
    with mc[0]:
        model_name = st.selectbox("Algorithme", list(MODELS.keys()), key="cls_model")
    with mc[1]:
        params = _hyperparams(model_name)

    if not st.button("🚀 Entraîner & évaluer", key="cls_train_btn"):
        return

    # ------------------------------------------------------------------ train
    data = df[features + [target]].dropna()
    X = data[features].to_numpy(dtype=float)
    y_raw = data[target]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = [str(c) for c in le.classes_]

    if len(np.unique(y)) < 2:
        st.error("La cible doit contenir au moins 2 classes.")
        return

    stratify = y if min(np.bincount(y)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=int(seed), stratify=stratify,
    )

    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    model = MODELS[model_name](params)
    with st.spinner(f"Entraînement {model_name}…"):
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

    # ------------------------------------------------------------------ metrics
    avg = "binary" if len(classes) == 2 else "weighted"
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_te, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_te, y_pred, average=avg, zero_division=0)

    section("Résultats")
    m = st.columns(4)
    with m[0]: _metric_card("Accuracy", acc)
    with m[1]: _metric_card("Precision", prec)
    with m[2]: _metric_card("Recall", rec)
    with m[3]: _metric_card("F1-score", f1)

    section("Matrice de confusion")
    cm = confusion_matrix(y_te, y_pred)
    st.plotly_chart(_confusion_fig(cm, classes), use_container_width=True)

    section("Rapport détaillé par classe")
    report = classification_report(y_te, y_pred, target_names=classes, zero_division=0, output_dict=True)
    rep_df = pd.DataFrame(report).T.round(4)
    st.dataframe(rep_df, use_container_width=True)

    # Feature importance / coefficients
    if hasattr(model, "feature_importances_"):
        section("Importance des variables")
        imp = pd.DataFrame({"feature": features, "importance": model.feature_importances_}) \
                .sort_values("importance", ascending=True)
        fig = px.bar(imp, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale=["#312E81", "#7C7CFF"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, max(280, 28 * len(features))), use_container_width=True)
    elif hasattr(model, "coef_"):
        section("Coefficients du modèle")
        coef = np.array(model.coef_).reshape(len(np.unique(y)) if model.coef_.ndim > 1 else 1, -1)[0]
        df_coef = pd.DataFrame({"feature": features, "coefficient": coef}) \
                    .sort_values("coefficient", key=lambda s: s.abs(), ascending=True)
        fig = px.bar(df_coef, x="coefficient", y="feature", orientation="h",
                     color="coefficient", color_continuous_scale=["#F472B6", "#0E1117", "#7C7CFF"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, max(280, 28 * len(features))), use_container_width=True)
