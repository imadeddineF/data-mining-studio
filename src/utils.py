"""Shared utilities: data loading, session state, plot theming."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Plotly default theme (dark, restrained palette)
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_dark"
PALETTE = [
    "#7C7CFF", "#22D3EE", "#F472B6", "#FBBF24", "#34D399",
    "#FB7185", "#A78BFA", "#60A5FA", "#FCA5A5", "#4ADE80",
]


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#161B26",
        plot_bgcolor="#161B26",
        font=dict(family="Inter, sans-serif", size=12, color="#E2E8F0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def ensure_state() -> None:
    """Initialise expected session-state keys on first render."""
    defaults: dict[str, Any] = {
        "raw_df": None,        # Original loaded DataFrame
        "df": None,            # Working DataFrame (after preprocessing)
        "source_name": None,   # File or sample name
        "history": [],         # Log of preprocessing steps applied
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def log_step(message: str) -> None:
    st.session_state.history.append(message)


def reset_working_df() -> None:
    if st.session_state.raw_df is not None:
        st.session_state.df = st.session_state.raw_df.copy()
        st.session_state.history = []


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_uploaded_file(file) -> pd.DataFrame:
    """Read an uploaded file; supports csv, tsv, xls(x), json, parquet."""
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(file, sep="\t")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    if name.endswith(".json"):
        return pd.read_json(file)
    if name.endswith(".parquet"):
        return pd.read_parquet(file)
    raise ValueError(f"Unsupported file type: {file.name}")


SAMPLES = {
    "Iris": "iris",
    "Wine": "wine",
    "Breast Cancer": "breast_cancer",
    "Diabetes": "diabetes",
}


def load_sample(name: str) -> pd.DataFrame:
    """Load a built-in sklearn dataset and return a labelled DataFrame."""
    from sklearn import datasets

    key = SAMPLES[name]
    if key == "iris":
        d = datasets.load_iris(as_frame=True)
    elif key == "wine":
        d = datasets.load_wine(as_frame=True)
    elif key == "breast_cancer":
        d = datasets.load_breast_cancer(as_frame=True)
    elif key == "diabetes":
        d = datasets.load_diabetes(as_frame=True)
    else:
        raise ValueError(name)

    df = d.frame.copy()
    if "target" in df.columns and hasattr(d, "target_names") and getattr(d, "target_names", None) is not None:
        try:
            df["target"] = pd.Categorical.from_codes(df["target"].astype(int), categories=list(d.target_names))
        except (ValueError, TypeError):
            pass
    return df


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude=[np.number]).columns.tolist()


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
