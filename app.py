"""Data Mining Studio — Streamlit entry point.

Mini-projet : Fouille de Données 1 — Faculté d'Informatique 2025-2026.
Three coordinated tabs cover the full pipeline:
    1. Prétraitement   — import, explore, clean, normalise, visualise
    2. Clustering      — K-Means / K-Medoids + Elbow + Silhouette + 2D projection
    3. Classification  — train/test split, models, confusion matrix, metrics
"""

from __future__ import annotations

import streamlit as st

from src import classification, clustering, preprocessing
from src.styles import inject_css, hero
from src.utils import ensure_state


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Mining Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Data Mining Studio**  \n"
            "Mini-projet Fouille de Données 1 — Faculté d'Informatique 2025-2026."
        ),
    },
)

inject_css()
ensure_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="logo">DM</div>
                <div class="text">Data Mining Studio
                    <small>Fouille de Données · 2025-26</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dataset status
        st.markdown("**État du dataset**")
        if st.session_state.df is None:
            st.markdown(
                '<span class="badge badge-warn">Aucune donnée</span>',
                unsafe_allow_html=True,
            )
            st.caption("Importez un dataset depuis l'onglet Prétraitement.")
        else:
            df = st.session_state.df
            st.markdown(
                f'<span class="badge badge-ok">{st.session_state.source_name or "data"}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"{len(df):,} lignes × {df.shape[1]} colonnes")

            missing = int(df.isna().sum().sum())
            num_cols = df.select_dtypes(include="number").shape[1]
            st.markdown("---")
            st.markdown("**Aperçu rapide**")
            c1, c2 = st.columns(2)
            c1.metric("Manquantes", missing)
            c2.metric("Numériques", num_cols)

        st.markdown("---")
        with st.expander("À propos", expanded=False):
            st.markdown(
                """
                Pipeline complet de fouille de données :
                - **Prétraitement** : import, exploration, nettoyage, normalisation, visualisation.
                - **Clustering** : K-Means & K-Medoids (implémentations from scratch), Elbow, Silhouette, projection PCA.
                - **Classification** : 7 modèles, matrice de confusion, Accuracy / Precision / Recall / F1.
                """
            )
            st.caption("Stack : Streamlit · scikit-learn · Plotly · pandas")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _sidebar()

    hero(
        title="Data Mining Studio",
        subtitle="Pipeline interactif de fouille de données — du chargement à l'évaluation des modèles.",
        pill="Mini-projet · Fouille de Données 1",
    )

    tab_pre, tab_clu, tab_cla = st.tabs([
        "🧹  Prétraitement",
        "🎯  Clustering",
        "🤖  Classification",
    ])

    with tab_pre:
        preprocessing.render()
    with tab_clu:
        clustering.render()
    with tab_cla:
        classification.render()


if __name__ == "__main__":
    main()
