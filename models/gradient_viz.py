"""
models/gradient_viz.py
=======================
Interactive gradient visualizer.
- Trains a simple network and tracks per-epoch gradients
- Shows gradient norm over time (vanishing / exploding)
- Shows gradient landscape (loss surface) for a 2-param model
- Explains gradient flow with colour-coded layer histograms
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd


# ── Simple 2-D logistic model for loss surface ────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_loss(w1: float, w2: float,
                  X: np.ndarray, y: np.ndarray) -> float:
    """Binary cross-entropy for 2-feature logistic regression."""
    z     = X[:, 0] * w1 + X[:, 1] * w2
    y_hat = sigmoid(z)
    eps   = 1e-12
    return float(-np.mean(y * np.log(y_hat + eps) +
                           (1 - y) * np.log(1 - y_hat + eps)))


# ── MLP training with gradient tracking ──────────────────────────────────────
def train_with_grad_tracking(X, y, layer_sizes, lr, epochs, act_name="sigmoid"):
    np.random.seed(42)

    # Build weights
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.5
        b = np.zeros(layer_sizes[i+1])
        weights.append(W)
        biases.append(b)

    grad_norms_per_layer = [[] for _ in range(len(weights))]
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        # Accumulate grads over all samples
        dW_acc = [np.zeros_like(w) for w in weights]

        for xi, yi in zip(X, y):
            # Forward
            a = xi.copy()
            activations = [a]
            zs = []
            for W, b in zip(weights, biases):
                z = W @ a + b
                zs.append(z)
                a = sigmoid(z)
                activations.append(a)

            # Loss
            y_hat  = activations[-1]
            loss   = float(0.5 * np.mean((y_hat - yi)**2))
            epoch_loss += loss

            # Backward
            delta = (y_hat - yi) * (y_hat * (1 - y_hat))
            deltas = [delta]
            for l in range(len(weights) - 2, -1, -1):
                d = (weights[l+1].T @ deltas[0]) * \
                    (activations[l+1] * (1 - activations[l+1]))
                deltas.insert(0, d)

            for l in range(len(weights)):
                dW = np.outer(deltas[l], activations[l])
                dW_acc[l] += dW
                weights[l] -= lr * dW
                biases[l]  -= lr * deltas[l]

        epoch_loss /= len(X)
        loss_history.append(epoch_loss)

        for l in range(len(weights)):
            grad_norms_per_layer[l].append(
                float(np.linalg.norm(dW_acc[l])) / len(X))

    return weights, loss_history, grad_norms_per_layer


# ── Loss surface ─────────────────────────────────────────────────────────────
def compute_loss_surface(X, y, w_range=3.0, n=50):
    ws = np.linspace(-w_range, w_range, n)
    Z  = np.zeros((n, n))
    for i, w1 in enumerate(ws):
        for j, w2 in enumerate(ws):
            Z[i, j] = logistic_loss(w1, w2, X, y)
    return ws, Z


# ── UI ────────────────────────────────────────────────────────────────────────
def run():
    st.info("📖 **Gradient Visualization** — Gradients tell us which direction "
            "and how steeply the loss changes w.r.t. each weight. "
            "*Vanishing* gradients (→ 0) halt learning; *exploding* gradients (→ ∞) "
            "destabilise training.")

    tab1, tab2, tab3 = st.tabs(
        ["📉 Gradient Norms", "🗺️ Loss Surface", "📊 Layer-wise Histogram"])

    # ── Config (shared) ───────────────────────────────────────────────────
    with st.sidebar:
        pass  # no sidebar needed

    c1, c2, c3 = st.columns(3)
    with c1:
        lr     = st.slider("Learning Rate", 0.001, 1.0, 0.05, step=0.001,
                           format="%.3f", key="gv_lr")
    with c2:
        epochs = st.slider("Epochs", 10, 300, 80, key="gv_ep")
    with c3:
        n_hidden = st.slider("Hidden Layers", 1, 4, 2, key="gv_hl")

    # Dataset: XOR
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)
    layer_sizes = [2] + [4]*n_hidden + [1]

    COLORS = ["#00d4ff", "#7c3aed", "#10b981", "#f59e0b", "#ec4899"]

    with tab1:
        if st.button("▶ Run & Track Gradients", use_container_width=True, key="gv_run"):
            with st.spinner("Training…"):
                _, loss_hist, grad_norms = train_with_grad_tracking(
                    X, y, layer_sizes, lr, epochs)

            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=["MSE Loss", "Gradient Norms per Layer"])

            fig.add_trace(go.Scatter(
                y=loss_hist, mode="lines",
                line=dict(color="#f97316", width=2), name="Loss"
            ), row=1, col=1)

            for l, norms in enumerate(grad_norms):
                fig.add_trace(go.Scatter(
                    y=norms, mode="lines",
                    line=dict(color=COLORS[l % len(COLORS)], width=2),
                    name=f"Layer {l+1}"
                ), row=2, col=1)

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,1)",
                font_family="Space Mono",
                height=520,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Diagnose
            last_norms = [n[-1] for n in grad_norms]
            if max(last_norms) < 0.001:
                st.warning("⚠️ Gradients are very small — possible **vanishing gradient** problem.")
            elif max(last_norms) > 10:
                st.warning("⚠️ Gradients are large — possible **exploding gradient** problem.")
            else:
                st.success("✅ Gradients look healthy.")

    with tab2:
        st.markdown("Loss surface for a 2-parameter logistic model on the AND gate.")
        X2 = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y2 = np.array([0, 0, 0, 1], dtype=float)
        ws, Z = compute_loss_surface(X2, y2, w_range=4.0, n=60)

        fig2 = go.Figure(data=[
            go.Surface(z=Z, x=ws, y=ws, colorscale="Plasma",
                       opacity=0.85, showscale=True)
        ])
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis_title="w₁",
                yaxis_title="w₂",
                zaxis_title="Loss",
                bgcolor="rgba(17,24,39,1)",
            ),
            font_family="Space Mono",
            height=480,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 2-D contour
        fig3 = go.Figure(go.Contour(
            z=Z, x=ws, y=ws,
            colorscale="Plasma", contours_coloring="heatmap",
            showscale=True,
        ))
        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,1)",
            xaxis_title="w₁", yaxis_title="w₂",
            font_family="Space Mono",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        if st.button("▶ Compute Layer Gradients", key="gv_hist", use_container_width=True):
            _, _, grad_norms = train_with_grad_tracking(
                X, y, layer_sizes, lr, epochs)

            fig4 = go.Figure()
            for l, norms in enumerate(grad_norms):
                fig4.add_trace(go.Histogram(
                    x=norms,
                    name=f"Layer {l+1}",
                    marker_color=COLORS[l % len(COLORS)],
                    opacity=0.7,
                    nbinsx=30,
                ))
            fig4.update_layout(
                barmode="overlay",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,1)",
                xaxis_title="Gradient Norm",
                yaxis_title="Frequency",
                font_family="Space Mono",
                height=360,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("Ideally all layers should show similar gradient magnitude ranges.")
