"""Volet 1 — Preprocessing: import, explore, clean, normalize, visualise."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .styles import section, empty_state
from .utils import (
    PALETTE,
    SAMPLES,
    categorical_columns,
    df_to_csv_bytes,
    human_size,
    load_sample,
    load_uploaded_file,
    log_step,
    numeric_columns,
    reset_working_df,
    style_fig,
)


# ---------------------------------------------------------------------------
# Importation
# ---------------------------------------------------------------------------
def _render_import() -> None:
    section("Importation")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("**Charger un fichier**  \n*CSV · TSV · Excel · JSON · Parquet*")
        uploaded = st.file_uploader(
            "Glisser-déposer un fichier",
            type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"],
            label_visibility="collapsed",
            key="pre_uploader",
        )
        if uploaded is not None:
            try:
                df = load_uploaded_file(uploaded)
                st.session_state.raw_df = df.copy()
                st.session_state.df = df.copy()
                st.session_state.source_name = uploaded.name
                st.session_state.history = [f"Chargé `{uploaded.name}` ({len(df)} lignes × {df.shape[1]} colonnes)"]
                st.success(f"Importé : **{uploaded.name}** — {len(df)} × {df.shape[1]}")
            except Exception as e:
                st.error(f"Erreur de lecture : {e}")

    with col2:
        st.markdown("**Ou un dataset de démonstration**")
        sample = st.selectbox("Sample", list(SAMPLES.keys()),
                              label_visibility="collapsed", key="pre_sample")
        if st.button("Charger le dataset", use_container_width=True, key="pre_load_sample"):
            df = load_sample(sample)
            st.session_state.raw_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.source_name = sample
            st.session_state.history = [f"Chargé sample `{sample}` ({len(df)} × {df.shape[1]})"]
            st.success(f"Sample **{sample}** chargé.")


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------
def _render_overview() -> None:
    df: pd.DataFrame = st.session_state.df
    section("Aperçu général")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", f"{len(df):,}")
    c2.metric("Colonnes", df.shape[1])
    c3.metric("Manquantes", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Mémoire", human_size(int(df.memory_usage(deep=True).sum())))

    with st.expander("Aperçu des données (head)", expanded=True):
        st.dataframe(df.head(15), use_container_width=True, height=320)


def _render_explore() -> None:
    df: pd.DataFrame = st.session_state.df
    section("Exploration")

    tab1, tab2, tab3 = st.tabs(["Statistiques descriptives", "Types de données", "Valeurs manquantes"])

    with tab1:
        num = numeric_columns(df)
        if num:
            st.dataframe(df[num].describe().T.round(4), use_container_width=True)
        else:
            st.info("Aucune colonne numérique.")
        cat = categorical_columns(df)
        if cat:
            st.markdown("**Colonnes catégorielles**")
            stats = pd.DataFrame({
                "unique": [df[c].nunique() for c in cat],
                "top":    [df[c].mode().iloc[0] if df[c].notna().any() else None for c in cat],
                "freq":   [df[c].value_counts().iloc[0] if df[c].notna().any() else 0 for c in cat],
            }, index=cat)
            st.dataframe(stats, use_container_width=True)

    with tab2:
        types = pd.DataFrame({
            "Colonne": df.columns,
            "Type":    [str(t) for t in df.dtypes],
            "Non-null": df.notna().sum().values,
            "Null":     df.isna().sum().values,
            "Unique":   [df[c].nunique() for c in df.columns],
        })
        st.dataframe(types, use_container_width=True, hide_index=True)

    with tab3:
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=True)
        if miss.empty:
            st.success("Aucune valeur manquante détectée.")
        else:
            fig = px.bar(
                x=miss.values, y=miss.index, orientation="h",
                labels={"x": "Manquantes", "y": "Colonne"},
                color=miss.values, color_continuous_scale=["#7C7CFF", "#F472B6"],
            )
            st.plotly_chart(style_fig(fig, height=max(260, 26 * len(miss))), use_container_width=True)


# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------
def _render_cleaning() -> None:
    df: pd.DataFrame = st.session_state.df
    section("Nettoyage — valeurs manquantes")

    total_missing = int(df.isna().sum().sum())
    if total_missing == 0:
        st.success("Le dataset ne contient aucune valeur manquante.")
        return

    st.caption(f"{total_missing} valeurs manquantes au total.")

    cols = st.columns([1, 1, 1])
    with cols[0]:
        strategy = st.selectbox(
            "Stratégie",
            ["Supprimer lignes", "Supprimer colonnes", "Imputer — moyenne",
             "Imputer — médiane", "Imputer — mode", "Imputer — valeur fixe"],
            key="pre_clean_strategy",
        )
    with cols[1]:
        target_cols = st.multiselect(
            "Colonnes ciblées (vide = toutes)",
            df.columns.tolist(),
            key="pre_clean_cols",
        )
    with cols[2]:
        fill_value = (
            st.text_input("Valeur (si fixe)", value="0", key="pre_clean_fill")
            if "fixe" in strategy else None
        )

    if st.button("Appliquer le nettoyage", use_container_width=True, key="pre_clean_apply"):
        target = target_cols or df.columns.tolist()
        before = df.isna().sum().sum()
        new_df = df.copy()

        if strategy == "Supprimer lignes":
            new_df = new_df.dropna(subset=target)
        elif strategy == "Supprimer colonnes":
            to_drop = [c for c in target if new_df[c].isna().any()]
            new_df = new_df.drop(columns=to_drop)
        elif strategy == "Imputer — moyenne":
            for c in target:
                if pd.api.types.is_numeric_dtype(new_df[c]):
                    new_df[c] = new_df[c].fillna(new_df[c].mean())
        elif strategy == "Imputer — médiane":
            for c in target:
                if pd.api.types.is_numeric_dtype(new_df[c]):
                    new_df[c] = new_df[c].fillna(new_df[c].median())
        elif strategy == "Imputer — mode":
            for c in target:
                if new_df[c].notna().any():
                    new_df[c] = new_df[c].fillna(new_df[c].mode().iloc[0])
        elif strategy == "Imputer — valeur fixe":
            try:
                v = float(fill_value)
            except ValueError:
                v = fill_value
            for c in target:
                new_df[c] = new_df[c].fillna(v)

        st.session_state.df = new_df
        after = new_df.isna().sum().sum()
        log_step(f"Nettoyage `{strategy}` → {before - after} valeurs traitées (reste {after}).")
        st.success(f"Stratégie appliquée : **{before - after}** valeurs traitées.")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _render_normalisation() -> None:
    df: pd.DataFrame = st.session_state.df
    section("Normalisation")

    num_cols = numeric_columns(df)
    if not num_cols:
        st.info("Aucune colonne numérique à normaliser.")
        return

    cols = st.columns([1, 1.4, 1])
    with cols[0]:
        method = st.radio("Méthode", ["Min-Max [0,1]", "Standardisation (Z-score)"],
                          horizontal=False, key="pre_norm_method")
    with cols[1]:
        target = st.multiselect("Colonnes", num_cols, default=num_cols, key="pre_norm_cols")
    with cols[2]:
        st.write("")
        st.write("")
        apply = st.button("Normaliser", use_container_width=True, key="pre_norm_apply")

    if apply and target:
        scaler = MinMaxScaler() if method.startswith("Min") else StandardScaler()
        new_df = df.copy()
        new_df[target] = scaler.fit_transform(new_df[target])
        st.session_state.df = new_df
        log_step(f"Normalisation `{method}` appliquée sur {len(target)} colonnes.")
        st.success(f"Normalisation **{method}** appliquée sur {len(target)} colonnes.")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _render_visualisation() -> None:
    df: pd.DataFrame = st.session_state.df
    section("Visualisation")

    num_cols = numeric_columns(df)
    if not num_cols:
        st.info("Aucune colonne numérique à visualiser.")
        return

    tabs = st.tabs(["Boxplot", "Scatter", "Histogramme", "Heatmap (corrélation)"])

    with tabs[0]:
        cols = st.multiselect("Colonnes", num_cols, default=num_cols[: min(4, len(num_cols))], key="box_cols")
        group = st.selectbox("Grouper par (optionnel)", ["—"] + categorical_columns(df), key="box_group")
        if cols:
            data = df[cols].melt(var_name="variable", value_name="value")
            if group != "—":
                data["group"] = np.tile(df[group].astype(str).values, len(cols))
                fig = px.box(data, x="variable", y="value", color="group", color_discrete_sequence=PALETTE)
            else:
                fig = px.box(data, x="variable", y="value", color="variable", color_discrete_sequence=PALETTE)
            st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    with tabs[1]:
        c = st.columns(3)
        x = c[0].selectbox("X", num_cols, key="sc_x")
        y = c[1].selectbox("Y", [c for c in num_cols if c != x], key="sc_y")
        color = c[2].selectbox("Couleur (optionnel)", ["—"] + df.columns.tolist(), key="sc_color")
        kwargs = {}
        if color != "—":
            kwargs["color"] = color
        fig = px.scatter(df, x=x, y=y, color_discrete_sequence=PALETTE, opacity=0.75, **kwargs)
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="rgba(255,255,255,0.4)")))
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    with tabs[2]:
        col = st.selectbox("Colonne", num_cols, key="hist_col")
        bins = st.slider("Bins", 5, 100, 30)
        fig = px.histogram(df, x=col, nbins=bins, color_discrete_sequence=["#7C7CFF"])
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    with tabs[3]:
        if len(num_cols) < 2:
            st.info("Au moins 2 colonnes numériques requises.")
        else:
            corr = df[num_cols].corr()
            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale=["#1E3A8A", "#0E1117", "#7C7CFF"],
                aspect="auto", zmin=-1, zmax=1,
            )
            st.plotly_chart(style_fig(fig, 480), use_container_width=True)


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------
def render() -> None:
    """Render the full preprocessing tab."""

    if st.session_state.df is None:
        _render_import()
        empty_state(
            "📂",
            "Aucune donnée chargée",
            "Importez un fichier ci-dessus ou choisissez un dataset de démonstration pour commencer.",
        )
        return

    _render_import()

    # Quick action bar
    bar = st.columns([1, 1, 1, 4])
    if bar[0].button("Réinitialiser", use_container_width=True, key="pre_reset"):
        reset_working_df()
        st.success("Données réinitialisées à la version d'origine.")
        st.rerun()
    bar[1].download_button(
        "Exporter CSV",
        data=df_to_csv_bytes(st.session_state.df),
        file_name=f"{st.session_state.source_name or 'dataset'}_clean.csv",
        mime="text/csv",
        use_container_width=True,
        key="pre_export",
    )
    with bar[3]:
        if st.session_state.history:
            with st.expander(f"Historique des transformations ({len(st.session_state.history)})", expanded=False):
                for i, h in enumerate(st.session_state.history, 1):
                    st.markdown(f"`{i:02d}.` {h}")

    _render_overview()
    _render_explore()
    _render_cleaning()
    _render_normalisation()
    _render_visualisation()
