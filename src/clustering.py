"""Volet 2 — Clustering.

Implements K-Means and K-Medoids from scratch (per the project specification),
plus the elbow method, silhouette scoring, and a 2D PCA projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .styles import section, empty_state
from .utils import PALETTE, numeric_columns, style_fig


# ---------------------------------------------------------------------------
# Algorithm implementations (from scratch)
# ---------------------------------------------------------------------------
@dataclass
class ClusterResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int


def _seed_rng(random_state: int | None) -> np.random.Generator:
    return np.random.default_rng(random_state)


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int | None = 42,
) -> ClusterResult:
    """K-Means with k-means++ initialisation."""
    rng = _seed_rng(random_state)
    n = X.shape[0]

    # k-means++ init
    idx0 = int(rng.integers(n))
    centers = [X[idx0]]
    for _ in range(1, k):
        d2 = np.min(np.linalg.norm(X[:, None, :] - np.array(centers)[None, :, :], axis=2) ** 2, axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        centers.append(X[rng.choice(n, p=probs)])
    centers = np.array(centers, dtype=float)

    labels = np.zeros(n, dtype=int)
    for it in range(max_iter):
        # Assignment
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)

        # Update
        new_centers = np.array([
            X[new_labels == j].mean(axis=0) if np.any(new_labels == j) else centers[j]
            for j in range(k)
        ])

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        labels = new_labels
        if shift < tol:
            break

    inertia = float(np.sum((X - centers[labels]) ** 2))
    return ClusterResult(labels=labels, centers=centers, inertia=inertia, n_iter=it + 1)


def kmedoids(
    X: np.ndarray,
    k: int,
    *,
    max_iter: int = 300,
    random_state: int | None = 42,
) -> ClusterResult:
    """K-Medoids (PAM-style) with greedy swap.

    Uses Euclidean distance. Medoids are actual data points.
    """
    rng = _seed_rng(random_state)
    n = X.shape[0]

    # Pre-compute pairwise distances (n×n)
    diff = X[:, None, :] - X[None, :, :]
    D = np.linalg.norm(diff, axis=2)

    # Initialise medoids: spread-out (k-medoids++ style)
    medoid_idx = [int(rng.integers(n))]
    for _ in range(1, k):
        d_to_nearest = np.min(D[:, medoid_idx], axis=1)
        probs = d_to_nearest / d_to_nearest.sum() if d_to_nearest.sum() > 0 else np.full(n, 1 / n)
        medoid_idx.append(int(rng.choice(n, p=probs)))
    medoid_idx = np.array(medoid_idx, dtype=int)

    def _cost(idx):
        return float(np.sum(np.min(D[:, idx], axis=1)))

    cost = _cost(medoid_idx)
    labels = np.argmin(D[:, medoid_idx], axis=1)

    for it in range(max_iter):
        improved = False
        # For each cluster, try to swap medoid with the point that minimises within-cluster cost
        for j in range(k):
            cluster_pts = np.where(labels == j)[0]
            if cluster_pts.size == 0:
                continue
            # Best candidate inside the cluster
            within = D[np.ix_(cluster_pts, cluster_pts)].sum(axis=1)
            best = cluster_pts[int(np.argmin(within))]
            if best != medoid_idx[j]:
                trial = medoid_idx.copy()
                trial[j] = best
                trial_cost = _cost(trial)
                if trial_cost + 1e-9 < cost:
                    medoid_idx = trial
                    cost = trial_cost
                    labels = np.argmin(D[:, medoid_idx], axis=1)
                    improved = True
        if not improved:
            break

    centers = X[medoid_idx]
    inertia = float(np.sum(np.min(D[:, medoid_idx], axis=1) ** 2))
    return ClusterResult(labels=labels, centers=centers, inertia=inertia, n_iter=it + 1)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _elbow_chart(X: np.ndarray, k_max: int, algorithm: str) -> go.Figure:
    ks = list(range(2, k_max + 1))
    inertias, sils = [], []
    for k in ks:
        res = kmeans(X, k) if algorithm == "K-Means" else kmedoids(X, k)
        inertias.append(res.inertia)
        try:
            sils.append(silhouette_score(X, res.labels))
        except Exception:
            sils.append(np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=inertias, mode="lines+markers", name="Inertie",
        line=dict(color="#7C7CFF", width=3), marker=dict(size=9),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=ks, y=sils, mode="lines+markers", name="Silhouette",
        line=dict(color="#22D3EE", width=2, dash="dot"), marker=dict(size=8),
        yaxis="y2",
    ))
    fig.update_layout(
        xaxis=dict(title="k (nombre de clusters)", dtick=1),
        yaxis=dict(title="Inertie (within-cluster sum of squares)"),
        yaxis2=dict(title="Silhouette", overlaying="y", side="right", range=[-0.1, 1]),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    return style_fig(fig, 400)


def _projection_chart(X: np.ndarray, labels: np.ndarray, centers: np.ndarray | None) -> go.Figure:
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        if centers is not None and len(centers):
            C2 = pca.transform(centers)
        else:
            C2 = None
        ev = pca.explained_variance_ratio_ * 100
        x_title, y_title = f"PC1 ({ev[0]:.1f}%)", f"PC2 ({ev[1]:.1f}%)"
    else:
        X2 = X
        C2 = centers
        x_title, y_title = "x₁", "x₂"

    fig = go.Figure()
    for k in np.unique(labels):
        mask = labels == k
        fig.add_trace(go.Scatter(
            x=X2[mask, 0], y=X2[mask, 1],
            mode="markers",
            marker=dict(size=9, color=PALETTE[int(k) % len(PALETTE)],
                        line=dict(width=0.6, color="rgba(255,255,255,0.4)"), opacity=0.85),
            name=f"Cluster {int(k)}",
        ))
    if C2 is not None and len(C2):
        fig.add_trace(go.Scatter(
            x=C2[:, 0], y=C2[:, 1], mode="markers",
            marker=dict(size=18, color="#FBBF24", symbol="x",
                        line=dict(width=2, color="white")),
            name="Centroïdes",
        ))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title,
                      legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
    return style_fig(fig, 520)


def render() -> None:
    if st.session_state.df is None:
        empty_state("🔍", "Aucune donnée à clusteriser",
                    "Importez d'abord un dataset depuis l'onglet Prétraitement.")
        return

    df: pd.DataFrame = st.session_state.df
    num_cols = numeric_columns(df)
    if len(num_cols) < 2:
        st.warning("Le clustering nécessite au moins 2 colonnes numériques.")
        return

    section("Configuration")
    cols = st.columns([1.2, 1, 1, 1])
    with cols[0]:
        features = st.multiselect("Variables", num_cols, default=num_cols, key="clu_features")
    with cols[1]:
        algo = st.selectbox("Algorithme", ["K-Means", "K-Medoids"], key="clu_algo")
    with cols[2]:
        k = st.number_input("k", min_value=2, max_value=15, value=3, step=1, key="clu_k")
    with cols[3]:
        seed = st.number_input("Random state", min_value=0, value=42, step=1, key="clu_seed")

    if not features or len(features) < 2:
        st.info("Sélectionnez au moins 2 variables pour exécuter le clustering.")
        return

    X = df[features].dropna().to_numpy(dtype=float)
    if len(X) < k:
        st.error(f"Pas assez d'échantillons ({len(X)}) pour k={k}.")
        return

    # Elbow
    section("Méthode du coude (Elbow)")
    k_max = st.slider("k maximum testé", 3, 12, 8, key="clu_elbow_k")
    with st.spinner("Calcul de la courbe d'Elbow…"):
        st.plotly_chart(_elbow_chart(X, k_max, algo), use_container_width=True)
    st.caption(
        "L'inertie (gauche) doit décroître ; le « coude » est le k où la décroissance "
        "ralentit. La silhouette (droite) doit être maximisée."
    )

    # Run
    section(f"Exécution — {algo} (k = {k})")
    with st.spinner(f"{algo} en cours…"):
        res = (kmeans(X, k, random_state=int(seed))
               if algo == "K-Means" else kmedoids(X, k, random_state=int(seed)))

    try:
        sil = silhouette_score(X, res.labels)
    except Exception:
        sil = float("nan")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette", f"{sil:.4f}")
    m2.metric("Inertie", f"{res.inertia:,.2f}")
    m3.metric("Itérations", res.n_iter)
    m4.metric("Échantillons", f"{len(X):,}")

    section("Projection 2D")
    st.plotly_chart(_projection_chart(X, res.labels, res.centers), use_container_width=True)

    section("Distribution des clusters")
    counts = pd.Series(res.labels).value_counts().sort_index()
    counts.index = [f"Cluster {i}" for i in counts.index]
    fig = px.bar(x=counts.index, y=counts.values,
                 color=counts.index, color_discrete_sequence=PALETTE,
                 labels={"x": "Cluster", "y": "Effectif"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    # Per-cluster summary
    with st.expander("Statistiques par cluster", expanded=False):
        df_clust = df[features].dropna().copy()
        df_clust["cluster"] = res.labels
        st.dataframe(df_clust.groupby("cluster").mean().round(3), use_container_width=True)
