"""Centralised styling: a single CSS injection plus a few small helpers.

Selectors are kept conservative so they work across Streamlit versions and
never hide the layout (no display/visibility tricks on Streamlit chrome).
"""

from __future__ import annotations

import streamlit as st


CUSTOM_CSS = """
<style>
/* ---------- Global ---------- */
html, body {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, #1E1B4B 0%, #2D1B69 50%, #0F172A 100%);
    border: 1px solid rgba(124, 124, 255, 0.25);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 22px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}
.hero h1 {
    color: #F8FAFC;
    margin: 0;
    font-size: 1.95rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.hero p {
    color: #CBD5E1;
    margin: 6px 0 0 0;
    font-size: 0.95rem;
    opacity: 0.9;
}
.hero .pill {
    display: inline-block;
    background: rgba(124, 124, 255, 0.18);
    color: #C7D2FE;
    border: 1px solid rgba(124, 124, 255, 0.4);
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* ---------- Section title ---------- */
.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 18px 0 10px 0;
    color: #E2E8F0;
    font-size: 1.15rem;
    font-weight: 600;
}
.section-title .dot {
    width: 8px;
    height: 8px;
    background: #7C7CFF;
    border-radius: 999px;
    box-shadow: 0 0 12px #7C7CFF;
    display: inline-block;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #7C7CFF 100%);
    color: white;
    border: 0;
    border-radius: 10px;
    padding: 9px 18px;
    font-weight: 600;
    transition: transform 0.08s ease, box-shadow 0.15s ease;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.3);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
}
.stDownloadButton > button {
    background: #1F2937;
    color: #E5E7EB;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    font-weight: 500;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 18px;
    border-radius: 8px 8px 0 0;
    color: #94A3B8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #C7D2FE !important;
}

/* ---------- Sidebar ---------- */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.sidebar-brand .logo {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #6366F1 0%, #7C7CFF 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: white;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    font-size: 0.85rem;
}
.sidebar-brand .text {
    color: #E2E8F0;
    font-weight: 700;
    font-size: 0.98rem;
}
.sidebar-brand .text small {
    display: block;
    color: #64748B;
    font-weight: 400;
    font-size: 0.72rem;
    margin-top: 1px;
}

/* ---------- Empty state ---------- */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: #64748B;
    background: #0F1420;
    border: 1px dashed rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    margin: 12px 0;
}
.empty-state .icon { font-size: 2.4rem; margin-bottom: 10px; opacity: 0.5; }
.empty-state .title { color: #CBD5E1; font-weight: 600; margin-bottom: 4px; }
.empty-state .desc  { font-size: 0.86rem; }

/* ---------- Status badges ---------- */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-ok    { background: rgba(34, 197, 94, 0.15);  color: #86EFAC; border: 1px solid rgba(34, 197, 94, 0.3); }
.badge-warn  { background: rgba(234, 179, 8, 0.15);  color: #FDE68A; border: 1px solid rgba(234, 179, 8, 0.3); }
.badge-info  { background: rgba(124, 124, 255, 0.15); color: #C7D2FE; border: 1px solid rgba(124, 124, 255, 0.3); }
</style>
"""


def inject_css() -> None:
    """Render the global stylesheet — call once at the top of the app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, pill: str | None = None) -> None:
    pill_html = f'<div class="pill">{pill}</div>' if pill else ""
    st.markdown(
        f"""
        <div class="hero">
            {pill_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(
        f'<div class="section-title"><span class="dot"></span>{title}</div>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, desc: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="icon">{icon}</div>
            <div class="title">{title}</div>
            <div class="desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "info") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'
