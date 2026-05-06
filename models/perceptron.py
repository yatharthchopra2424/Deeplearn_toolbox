"""
models/perceptron.py
=====================
Manual single-layer Perceptron for binary classification.

Learning Rule:  w ← w + η · (y − ŷ) · x
Activation:     Heaviside step function (threshold at 0)

Features:
  • AND, OR, NAND, custom 2-D dataset
  • Step-by-step epoch table with all intermediate values
  • Decision boundary across epochs (slider)
  • Weight evolution chart
  • Theory expandable
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from components.ui import theory_box, formula_box, step_badge, metric_row, PALETTE


# ──────────────────────────────────────────────────────────────────────────────
#  Core logic
# ──────────────────────────────────────────────────────────────────────────────

def step_fn(z: float) -> int:
    """Heaviside step activation: 1 if z ≥ 0 else 0."""
    return 1 if z >= 0 else 0


def perceptron_predict(x, w, b):
    z = float(np.dot(w, x) + b)
    return step_fn(z), z


def perceptron_train(X, y, lr: float, max_epochs: int):
    """
    Full perceptron training loop.
    Returns (final_w, final_b, history)
    history: list of dicts, one per epoch.
    """
    w = np.zeros(X.shape[1])
    b = 0.0
    history = []

    for epoch in range(1, max_epochs + 1):
        total_error = 0
        rows = []
        for i, (xi, yi) in enumerate(zip(X, y)):
            y_hat, z = perceptron_predict(xi, w, b)
            err = int(yi) - y_hat
            w_new = w + lr * err * xi
            b_new = b + lr * err
            rows.append({
                "Sample":    i + 1,
                "x₁":       round(float(xi[0]), 4),
                "x₂":       round(float(xi[1]), 4),
                "Target":   int(yi),
                "z (net)":  round(z, 4),
                "ŷ":        y_hat,
                "Error":    err,
                "Δw₁":      round(float(lr * err * xi[0]), 4),
                "Δw₂":      round(float(lr * err * xi[1]), 4),
                "w₁→":      round(float(w_new[0]), 4),
                "w₂→":      round(float(w_new[1]), 4),
                "b→":       round(float(b_new), 4),
            })
            w, b = w_new, b_new
            total_error += abs(err)

        history.append(dict(epoch=epoch, total_error=total_error,
                            w=w.copy(), b=float(b), rows=rows))
        if total_error == 0:
            break
    return w, b, history


# ──────────────────────────────────────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────────────────────────────────────

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,1)",
    font_family="Space Mono",
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def plot_boundary(X, y, w, b, epoch_label="Final"):
    colors  = ["#ef4444" if yi == 0 else "#10b981" for yi in y]
    symbols = ["circle" if yi == 0 else "diamond" for yi in y]
    fig = go.Figure()

    # Class regions
    xs_g = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 80)
    ys_g = np.linspace(X[:,1].min()-1, X[:,1].max()+1, 80)
    XG, YG = np.meshgrid(xs_g, ys_g)
    ZG = np.vectorize(lambda xi, yi: step_fn(w[0]*xi + w[1]*yi + b))(XG, YG)
    fig.add_trace(go.Contour(
        x=xs_g, y=ys_g, z=ZG, showscale=False, hoverinfo="skip", name="",
        colorscale=[[0,"rgba(239,68,68,0.10)"],[1,"rgba(16,185,129,0.10)"]],
        contours=dict(showlines=False),
    ))

    # Decision line
    if abs(w[1]) > 1e-9:
        xs = np.linspace(X[:,0].min()-.8, X[:,0].max()+.8, 200)
        ys = -(w[0]*xs + b) / w[1]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                  line=dict(color="#00d4ff", width=2.5, dash="dash"),
                                  name="Decision boundary"))

    # Points
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1], mode="markers",
        marker=dict(color=colors, size=14, symbol=symbols,
                    line=dict(width=1.5, color="#1e2d45")),
        name="Samples",
    ))

    fig.update_layout(**_DARK, height=380,
                      title=f"Decision Boundary — {epoch_label}",
                      xaxis_title="x₁", yaxis_title="x₂")
    return fig


def plot_error_curve(history):
    epochs = [h["epoch"] for h in history]
    errors = [h["total_error"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=epochs, y=errors,
        marker_color=["#ef4444" if e > 0 else "#10b981" for e in errors],
        name="Errors",
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=errors, mode="lines+markers",
        line=dict(color="#00d4ff", width=2), marker=dict(size=6),
        name="Trend",
    ))
    fig.update_layout(**_DARK, height=300, title="Errors per Epoch",
                      xaxis_title="Epoch", yaxis_title="Misclassifications")
    return fig


def plot_weight_evolution(history):
    epochs = [h["epoch"] for h in history]
    fig = go.Figure()
    for vals, name, color in zip(
        [[h["w"][0] for h in history],
         [h["w"][1] for h in history],
         [h["b"]    for h in history]],
        ["w₁", "w₂", "bias"],
        PALETTE,
    ):
        fig.add_trace(go.Scatter(
            x=epochs, y=vals, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=5),
            name=name,
        ))
    fig.update_layout(**_DARK, height=280, title="Weight & Bias Evolution",
                      xaxis_title="Epoch", yaxis_title="Value")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Streamlit UI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run():
    theory_box("Perceptron: Mathematical Foundation", """
**Definition:** Linear binary classifier with threshold activation

Input: $x \\in \\mathbb{R}^d$, Weights: $w \\in \\mathbb{R}^d$, Bias: $b \\in \\mathbb{R}$

**Forward Pass (Prediction):**

Weighted sum:
$$z = w \\cdot x + b = \\sum_{i=1}^{d} w_i x_i + b$$

Step activation (Heaviside function):
$$\\hat{y} = \\begin{cases} 1 & \\text{if } z \\geq 0 \\\\ 0 & \\text{if } z < 0 \\end{cases}$$

**Learning Rule (Perceptron Update):**

When prediction is wrong ($\\hat{y} \\neq y$):
$$w \\leftarrow w + \\eta (y - \\hat{y}) x$$
$$b \\leftarrow b + \\eta (y - \\hat{y})$$

where $\\eta$ = learning rate, $(y - \\hat{y}) \\in \\{-1, 1\\}$

**Decision Boundary:**

The hyperplane separating classes:
$$w \\cdot x + b = 0$$

**Convergence:**

For linearly separable data, guaranteed convergence:
- Maximum iterations: $O(R^2/\\gamma^2)$
- $R$ = data radius, $\\gamma$ = margin

**Limitations:**
- Cannot learn non-linearly separable functions (e.g., XOR)
- Binary classification only
- No probabilistic outputs
    """)

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        lr = st.slider("Learning Rate (η)", 0.01, 1.0, 0.1, 0.01)
    with c2:
        epochs = st.slider("Max Epochs", 1, 100, 25)
    with c3:
        dataset = st.selectbox("Dataset",
            ["AND Gate", "OR Gate", "NAND Gate", "XOR (non-separable)"])

    ds_map = {
        "AND Gate":           (np.array([[0,0],[0,1],[1,0],[1,1]],float), np.array([0,0,0,1])),
        "OR Gate":            (np.array([[0,0],[0,1],[1,0],[1,1]],float), np.array([0,1,1,1])),
        "NAND Gate":          (np.array([[0,0],[0,1],[1,0],[1,1]],float), np.array([1,1,1,0])),
        "XOR (non-separable)":(np.array([[0,0],[0,1],[1,0],[1,1]],float), np.array([0,1,1,0])),
    }
    X, y = ds_map[dataset]

    step_badge(1, "Training Data")
    df = pd.DataFrame(X, columns=["x₁","x₂"])
    df["y (target)"] = y.astype(int)
    st.dataframe(df, use_container_width=True)

    if st.button("▶  Train Perceptron", use_container_width=True):
        w_f, b_f, history = perceptron_train(X, y, lr, epochs)
        converged = history[-1]["total_error"] == 0

        step_badge(2, "Training Results")
        metric_row([
            ("Epochs Run",   len(history)),
            ("Final Errors", history[-1]["total_error"]),
            ("Converged?",   "✅ Yes" if converged else "❌ No"),
            ("Final w₁",     round(float(w_f[0]), 4)),
            ("Final w₂",     round(float(w_f[1]), 4)),
            ("Final bias",   round(float(b_f), 4)),
        ])

        if not converged and "XOR" in dataset:
            st.warning("⚠️ XOR is not linearly separable — "
                       "a single perceptron cannot solve it. Try the MLP module!")

        step_badge(3, "Visualisations")
        t1, t2, t3 = st.tabs(
            ["🗺️ Decision Boundary", "📉 Error Curve", "⚖️ Weight Evolution"])

        with t1:
            ep = st.slider("Show boundary at epoch:", 1, len(history),
                           len(history), key="p_bd")
            h = history[ep - 1]
            st.plotly_chart(plot_boundary(X, y, h["w"], h["b"], f"Epoch {ep}"),
                            use_container_width=True)
        with t2:
            st.plotly_chart(plot_error_curve(history), use_container_width=True)
        with t3:
            st.plotly_chart(plot_weight_evolution(history), use_container_width=True)

        step_badge(4, "Per-Epoch Calculation Table")
        ep_sel = st.slider("Inspect epoch:", 1, len(history), 1, key="p_tbl")
        ep_d = history[ep_sel - 1]
        st.caption(f"Epoch {ep_d['epoch']}  ·  Errors: {ep_d['total_error']}  "
                   f"·  w={np.round(ep_d['w'],4)}  ·  b={ep_d['b']:.4f}")
        st.dataframe(pd.DataFrame(ep_d["rows"]), use_container_width=True)

        formula_box(r"w \leftarrow w + \eta \cdot (y - \hat{y}) \cdot x",
                    "Perceptron update rule — only fires on misclassified samples")
