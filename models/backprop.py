"""
models/backprop.py
==================
Step-by-step Backpropagation Visualizer.

Network: 2 → 2 (hidden, sigmoid) → 1 (output, sigmoid)
Loss:    MSE = 0.5 · (ŷ − y)²

Shows every quantity in the chain:
  Forward:  z1, a1, z2, a2, loss
  Backward: δ2, δ1, ∂L/∂W2, ∂L/∂W1
  Update:   W_new = W_old − η · ∂L/∂W
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
from components.ui import theory_box, formula_box, step_badge, metric_row, PALETTE


# ──────────────────────────────────────────────────────────────────────────────
# Core math
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sig_d(a):
    """Sigmoid derivative given output a."""
    return a * (1.0 - a)


def one_pass(x, y_true, W1, b1, W2, b2, lr):
    """
    Full forward + backward + weight update for a single sample.
    Returns detailed dict of every intermediate value.
    """
    # ── Forward ──────────────────────────────────────────────────────────
    z1 = W1 @ x + b1          # (2,)
    a1 = sigmoid(z1)           # (2,)
    z2 = W2 @ a1 + b2          # (1,)
    a2 = sigmoid(z2)           # (1,)
    loss = 0.5 * float((a2[0] - y_true) ** 2)

    # ── Backward ─────────────────────────────────────────────────────────
    # Output layer
    dL_da2  = float(a2[0] - y_true)          # ∂L/∂a2
    da2_dz2 = float(sig_d(a2)[0])            # ∂a2/∂z2
    delta2  = dL_da2 * da2_dz2               # δ2  (scalar)

    dL_dW2 = np.outer([delta2], a1)          # (1,2)  ∂L/∂W2
    dL_db2 = np.array([delta2])              # (1,)

    # Hidden layer
    delta1 = (W2[0] * delta2) * sig_d(a1)   # (2,)
    dL_dW1 = np.outer(delta1, x)            # (2,2)  ∂L/∂W1
    dL_db1 = delta1                          # (2,)

    # ── Update ───────────────────────────────────────────────────────────
    W1_new = W1 - lr * dL_dW1
    b1_new = b1 - lr * dL_db1
    W2_new = W2 - lr * dL_dW2
    b2_new = b2 - lr * dL_db2

    return dict(
        # inputs
        x=x, y=y_true,
        # forward
        z1=z1, a1=a1, z2=z2, a2=a2, loss=loss,
        # backward
        dL_da2=dL_da2, da2_dz2=da2_dz2, delta2=delta2,
        delta1=delta1,
        dL_dW2=dL_dW2, dL_db2=dL_db2,
        dL_dW1=dL_dW1, dL_db1=dL_db1,
        # updates
        W1_old=W1.copy(), W1_new=W1_new,
        W2_old=W2.copy(), W2_new=W2_new,
        b1_old=b1.copy(), b1_new=b1_new,
        b2_old=b2.copy(), b2_new=b2_new,
    )


def full_train(X, y_vec, lr, epochs):
    np.random.seed(0)
    W1 = np.random.randn(2, 2) * 0.5
    b1 = np.zeros(2)
    W2 = np.random.randn(1, 2) * 0.5
    b2 = np.zeros(1)

    loss_hist, w1_norms, w2_norms = [], [], []

    for _ in range(epochs):
        ep_loss = 0.0
        for xi, yi in zip(X, y_vec):
            d = one_pass(xi, yi, W1, b1, W2, b2, lr)
            W1, b1, W2, b2 = d["W1_new"], d["b1_new"], d["W2_new"], d["b2_new"]
            ep_loss += d["loss"]
        loss_hist.append(ep_loss / len(X))
        w1_norms.append(float(np.linalg.norm(W1)))
        w2_norms.append(float(np.linalg.norm(W2)))

    return W1, b1, W2, b2, loss_hist, w1_norms, w2_norms


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,1)",
    font_family="Space Mono",
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def run():
    theory_box("Backpropagation: The Chain Rule Applied", """
**Core Concept:** Compute gradients of loss with respect to all parameters using the
chain rule from calculus, applying it systematically from output → input.

**Forward Pass:** (compute activations top to bottom)

Layer 1: $z^{(1)} = W^{(1)} x + b^{(1)}$, $a^{(1)} = \\sigma(z^{(1)})$

Layer 2: $z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}$, $a^{(2)} = \\sigma(z^{(2)})$

**Loss Function (MSE):**
$$\\mathcal{L} = \\frac{1}{2}(a^{(2)} - y)^2$$

**Backward Pass:** (compute gradients bottom to top)

Output layer error:
$$\\delta^{(2)} = \\frac{\\partial \\mathcal{L}}{\\partial z^{(2)}} = (a^{(2)} - y) \\cdot \\sigma'(z^{(2)})$$

where $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$ for sigmoid

Gradient w.r.t. Output weights:
$$\\frac{\\partial \\mathcal{L}}{\\partial W^{(2)}} = \\delta^{(2)} (a^{(1)})^T$$

Propagate to hidden layer:
$$\\delta^{(1)} = (W^{(2)})^T \\delta^{(2)} \\odot \\sigma'(z^{(1)})$$

Gradient w.r.t. Hidden weights:
$$\\frac{\\partial \\mathcal{L}}{\\partial W^{(1)}} = \\delta^{(1)} x^T$$

**Parameter Updates:**
$$W^{(l)} \\leftarrow W^{(l)} - \\eta \\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}}$$
$$b^{(l)} \\leftarrow b^{(l)} - \\eta \\frac{\\partial \\mathcal{L}}{\\partial b^{(l)}}$$

**Computational Efficiency:**
- Reuses activations from forward pass
- Complexity: $O(n)$ for forward and backward combined
- Alternative (numerical gradient): $O(n^2)$

**Vanishing Gradients Problem:**
When $|\\sigma'(z)| < 0.3$ for many layers:
$$\\delta^{(1)} = W^{(2)^T} \\cdot W^{(1)^T} \\cdot ... \\cdot \\delta^{(L)}$$
$$\\approx (\\sigma'_{\\text{small}})^L \\delta^{(L)} \\approx 0$$

Solutions: ReLU activation, residual connections, layer normalization
    """)

    # Controls
    c1, c2 = st.columns(2)
    with c1:
        lr     = st.slider("Learning Rate (η)", 0.01, 2.0, 0.5, 0.01)
    with c2:
        epochs = st.slider("Training Epochs", 1, 500, 150)

    X     = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_vec = np.array([0, 1, 1, 0], dtype=float)   # XOR

    st.markdown("**Network: `2 → 2 (sigmoid) → 1 (sigmoid)` on XOR**")

    tab1, tab2 = st.tabs(["🔬 Single Forward/Backward Step", "📈 Full Training"])

    # ── Single step inspector ─────────────────────────────────────────────
    with tab1:
        # Fixed initial weights for reproducibility
        W1 = np.array([[0.15, 0.20],[0.25, 0.30]])
        b1 = np.array([0.35, 0.35])
        W2 = np.array([[0.40, 0.45]])
        b2 = np.array([0.60])

        sample = st.slider("Sample index", 0, 3, 1, key="bp_samp")
        d = one_pass(X[sample], y_vec[sample], W1, b1, W2, b2, lr)

        # ① Forward
        step_badge(1, "Forward Pass")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Hidden layer**")
            st.dataframe(pd.DataFrame({
                "neuron": [1, 2],
                "z (net)":    np.round(d["z1"], 6),
                "a = σ(z)":   np.round(d["a1"], 6),
                "σ'(a)":      np.round(sig_d(d["a1"]), 6),
            }), use_container_width=True)
        with col_b:
            st.markdown("**Output layer**")
            st.dataframe(pd.DataFrame({
                "neuron":  [1],
                "z (net)": np.round(d["z2"], 6),
                "ŷ = σ(z)":np.round(d["a2"], 6),
            }), use_container_width=True)

        st.metric("MSE Loss  0.5·(ŷ − y)²", f"{d['loss']:.8f}")
        formula_box(r"\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2")

        # ② Backward
        step_badge(2, "Backward Pass — Chain Rule")

        st.markdown("**Output layer gradient (δ₂)**")
        st.code(
            f"∂L/∂a2  = ŷ − y      = {d['dL_da2']:.6f}\n"
            f"∂a2/∂z2 = σ'(a2)     = {d['da2_dz2']:.6f}\n"
            f"δ₂      = ∂L/∂a2 · ∂a2/∂z2 = {d['delta2']:.6f}",
            language=None
        )

        st.markdown("**∂L/∂W2  (output weight gradients)**")
        df_dW2 = pd.DataFrame(np.round(d["dL_dW2"], 6),
                               columns=["a1₁","a1₂"],
                               index=["∂L/∂W2[0]"])
        st.dataframe(df_dW2, use_container_width=True)

        st.markdown("**Hidden layer gradient (δ₁)**")
        st.code(
            f"δ₁ = (W2^T · δ₂) ⊙ σ'(a1)\n"
            f"   = {np.round(d['delta1'], 6)}",
            language=None
        )

        st.markdown("**∂L/∂W1  (hidden weight gradients)**")
        df_dW1 = pd.DataFrame(np.round(d["dL_dW1"], 6),
                               columns=["x₁","x₂"],
                               index=["∂L/∂W1[0]","∂L/∂W1[1]"])
        st.dataframe(df_dW1, use_container_width=True)

        # ③ Update
        step_badge(3, "Weight Update  w_new = w_old − η · ∂L/∂w")
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            st.markdown("**W1**")
            st.dataframe(pd.DataFrame({
                "idx":    [f"W1{i}" for i in range(4)],
                "old":    d["W1_old"].flatten().round(6),
                "grad":   d["dL_dW1"].flatten().round(6),
                "new":    d["W1_new"].flatten().round(6),
            }), use_container_width=True)
        with col_u2:
            st.markdown("**W2**")
            st.dataframe(pd.DataFrame({
                "idx":  ["W2₁","W2₂"],
                "old":  d["W2_old"].flatten().round(6),
                "grad": d["dL_dW2"].flatten().round(6),
                "new":  d["W2_new"].flatten().round(6),
            }), use_container_width=True)

    # ── Full training ─────────────────────────────────────────────────────
    with tab2:
        if st.button("▶  Run Full Training", use_container_width=True, key="bp_run"):
            with st.spinner("Training…"):
                _, _, _, _, loss_hist, w1_n, w2_n = full_train(
                    X, y_vec, lr, epochs)

            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=["MSE Loss", "Weight Magnitudes"],
                                vertical_spacing=0.14)
            fig.add_trace(go.Scatter(
                y=loss_hist, mode="lines",
                line=dict(color="#10b981", width=2), name="Loss"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                y=w1_n, mode="lines",
                line=dict(color="#7c3aed", width=2), name="‖W1‖"
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                y=w2_n, mode="lines",
                line=dict(color="#f59e0b", width=2), name="‖W2‖"
            ), row=2, col=1)

            fig.update_layout(
                **_DARK, height=520,
                showlegend=True,
            )
            fig.update_xaxes(title_text="Epoch")
            st.plotly_chart(fig, use_container_width=True)
            metric_row([
                ("Final Loss",      f"{loss_hist[-1]:.6f}"),
                ("Min Loss",        f"{min(loss_hist):.6f}"),
                ("Epochs to min",   loss_hist.index(min(loss_hist))+1),
            ])
