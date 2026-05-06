"""
components/ui.py
=================
Reusable Streamlit UI building blocks used across model pages.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np


# ── Styled info / theory boxes ────────────────────────────────────────────────
def theory_box(title: str, content: str):
    """Render a collapsible theory / explanation card."""
    with st.expander(f"📚 Theory: {title}"):
        st.markdown(content)


def formula_box(latex_str: str, caption: str = ""):
    """Render a centred LaTeX formula with optional caption."""
    st.latex(latex_str)
    if caption:
        st.caption(caption)


# ── Metric row helper ─────────────────────────────────────────────────────────
def metric_row(items: list):
    """
    items: list of (label, value) tuples
    Renders them as equal-width st.metric columns.
    """
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


# ── Plotly dark defaults ──────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,1)",
    font=dict(family="Space Mono, monospace", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)


def apply_dark(fig: go.Figure, height: int = 360, title: str = "") -> go.Figure:
    fig.update_layout(**DARK_LAYOUT, height=height, title=title)
    return fig


# ── Step badge ────────────────────────────────────────────────────────────────
def step_badge(number: int, label: str):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:12px 0 6px;">'
        f'<span style="background:#7c3aed;color:white;border-radius:50%;'
        f'width:28px;height:28px;display:flex;align-items:center;justify-content:center;'
        f'font-weight:700;font-size:.85rem;flex-shrink:0;">{number}</span>'
        f'<span style="font-weight:600;font-size:.95rem;">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Colour palette for multi-line charts ────────────────────────────────────
PALETTE = ["#00d4ff", "#7c3aed", "#10b981", "#f59e0b",
           "#ec4899", "#06b6d4", "#8b5cf6", "#14b8a6"]


# ── Generic loss-curve chart ─────────────────────────────────────────────────
def loss_curve(loss_history: list, label: str = "Loss",
               color: str = "#10b981", height: int = 300) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=loss_history, mode="lines",
        line=dict(color=color, width=2),
        name=label,
    ))
    apply_dark(fig, height=height, title=label)
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text=label)
    return fig


# ── Weight matrix heatmap ────────────────────────────────────────────────────
def weight_heatmap(W: np.ndarray, title: str = "Weight Matrix") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=W, colorscale="RdBu", zmid=0, showscale=True
    ))
    apply_dark(fig, height=260, title=title)
    return fig


# ── Section divider ───────────────────────────────────────────────────────────
def section(title: str):
    st.markdown(
        f'<div style="border-left:3px solid #00d4ff;padding-left:12px;'
        f'margin:24px 0 12px;font-weight:700;font-size:1rem;">{title}</div>',
        unsafe_allow_html=True,
    )
