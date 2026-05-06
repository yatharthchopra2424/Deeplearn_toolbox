"""
models/mlp.py
==============
Manual Multilayer Perceptron (MLP).

Architecture:  input → [hidden × N] → output
Forward pass:  z = W·a_prev + b;  a = σ(z)
Backward pass: manual chain-rule (BPTT-style)
Init:          Xavier uniform

Features:
  • Configurable depth / width / activation
  • SVG architecture diagram
  • Per-layer z & a values for any sample
  • Live training on XOR / make_moons
  • Prediction grid (decision surface)
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
from components.ui import (theory_box, formula_box, step_badge,
                            metric_row, PALETTE)

# ──────────────────────────────────────────────────────────────────────────────
# Activations
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0.0, z)

def tanh_act(z):
    return np.tanh(z)

def leaky_relu(z):
    return np.where(z > 0, z, 0.01 * z)

ACT_FNS = {"sigmoid": sigmoid, "relu": relu,
           "tanh": tanh_act, "leaky_relu": leaky_relu}

def act_deriv(name, a):
    """Derivative given activation output a."""
    if name == "sigmoid":     return a * (1 - a)
    if name == "relu":        return (a > 0).astype(float)
    if name == "tanh":        return 1 - a**2
    if name == "leaky_relu":  return np.where(a > 0, 1.0, 0.01)
    return np.ones_like(a)


# ──────────────────────────────────────────────────────────────────────────────
# Weight init
# ──────────────────────────────────────────────────────────────────────────────

def xavier(fan_in, fan_out):
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-lim, lim, (fan_out, fan_in))


# ──────────────────────────────────────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────────────────────────────────────

def build_net(sizes):
    Ws = [xavier(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
    bs = [np.zeros(sizes[i+1])         for i in range(len(sizes)-1)]
    return Ws, bs


def forward(x, Ws, bs, act_name, return_all=False):
    """
    Forward pass through all layers.
    If return_all: return (a_final, activations_list, z_list)
    """
    a = x.copy()
    acts = [a]
    zs   = []
    for idx, (W, b) in enumerate(zip(Ws, bs)):
        z = W @ a + b
        zs.append(z)
        is_last = (idx == len(Ws) - 1)
        a = sigmoid(z) if is_last else ACT_FNS[act_name](z)
        acts.append(a)
    if return_all:
        return a, acts, zs
    return a


def backward(acts, zs, Ws, y, act_name, lr):
    """Manual backprop; returns updated Ws, bs and loss."""
    y_hat = acts[-1]
    loss  = float(0.5 * np.mean((y_hat - y)**2))

    # Output delta
    delta = (y_hat - y) * (y_hat * (1 - y_hat))
    deltas = [delta]

    for l in range(len(Ws) - 2, -1, -1):
        d = (Ws[l+1].T @ deltas[0]) * act_deriv(act_name, acts[l+1])
        deltas.insert(0, d)

    new_Ws, new_bs = [], []
    for l in range(len(Ws)):
        new_Ws.append(Ws[l] - lr * np.outer(deltas[l], acts[l]))
        new_bs.append(np.zeros(Ws[l].shape[0]) - lr * deltas[l])

    return new_Ws, new_bs, loss


def train(X, y, sizes, act_name, lr, epochs):
    np.random.seed(42)
    Ws, bs = build_net(sizes)
    loss_hist = []

    for _ in range(epochs):
        ep_loss = 0.0
        for xi, yi in zip(X, y):
            a_out, acts, zs = forward(xi, Ws, bs, act_name, return_all=True)
            Ws, bs, loss = backward(acts, zs, Ws, yi, act_name, lr)
            ep_loss += loss
        loss_hist.append(ep_loss / len(X))

    return Ws, bs, loss_hist


# ──────────────────────────────────────────────────────────────────────────────
# Architecture SVG diagram
# ──────────────────────────────────────────────────────────────────────────────

def architecture_svg(sizes):
    """Return an inline SVG of the network topology."""
    n_layers = len(sizes)
    max_n    = max(sizes)
    W, H     = 700, max(240, max_n * 45 + 60)
    x_step   = W / (n_layers + 1)

    circles  = []
    lines    = []
    labels   = []

    node_pos = {}  # (layer, neuron) → (cx, cy)

    for li, n in enumerate(sizes):
        cx = x_step * (li + 1)
        for ni in range(n):
            cy = H/2 + (ni - (n-1)/2) * 44
            node_pos[(li, ni)] = (cx, cy)

            # Color
            if li == 0:
                fill = "#00d4ff"
            elif li == n_layers - 1:
                fill = "#10b981"
            else:
                fill = "#7c3aed"

            circles.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="16" '
                f'fill="{fill}" fill-opacity="0.85" '
                f'stroke="#1e2d45" stroke-width="1.5"/>'
            )

        # Layer label
        ltype = "Input" if li==0 else ("Output" if li==n_layers-1 else f"H{li}")
        labels.append(
            f'<text x="{cx:.1f}" y="{H-8}" text-anchor="middle" '
            f'font-size="11" fill="#64748b" font-family="Space Mono">'
            f'{ltype} ({n})</text>'
        )

    # Connections (only up to 4 neurons per layer to avoid clutter)
    for li in range(n_layers - 1):
        n_src = min(sizes[li], 4)
        n_dst = min(sizes[li+1], 4)
        for si in range(n_src):
            for di in range(n_dst):
                x1, y1 = node_pos[(li, si)]
                x2, y2 = node_pos[(li+1, di)]
                lines.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
                    f'x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="#1e2d45" stroke-width="1" stroke-opacity="0.6"/>'
                )

    svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
           f'style="background:rgba(17,24,39,0.6);border-radius:12px;'
           f'border:1px solid #1e2d45;width:100%;">'
           + "\n".join(lines)
           + "\n".join(circles)
           + "\n".join(labels)
           + "</svg>")
    return svg


# ──────────────────────────────────────────────────────────────────────────────
# Decision surface
# ──────────────────────────────────────────────────────────────────────────────

def decision_surface(X, y, Ws, bs, act_name):
    xs = np.linspace(-0.5, 1.5, 60)
    ys = np.linspace(-0.5, 1.5, 60)
    XG, YG = np.meshgrid(xs, ys)
    ZG = np.array([[float(forward(np.array([xi, yi]), Ws, bs, act_name)[0])
                    for xi in xs] for yi in ys])

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=ZG, showscale=True,
        colorscale="RdYlGn", opacity=0.65,
        contours=dict(start=0, end=1, size=0.1),
        name="Output",
    ))

    colors = ["#ef4444" if yi[0]==0 else "#10b981" for yi in y]
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1], mode="markers",
        marker=dict(color=colors, size=14, line=dict(width=1.5, color="white")),
        name="Samples",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,1)",
        font_family="Space Mono",
        xaxis_title="x₁", yaxis_title="x₂",
        height=380, margin=dict(l=20,r=20,t=36,b=20),
        title="Decision Surface",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

def run():
    theory_box("Multilayer Perceptron: Complete Mathematics", """
**Network Architecture:**

For $L$ layers with dimensions $(n_0, n_1, ..., n_L)$:

**Layer $l$ Forward Pass:**

Affine transformation:
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \\quad W^{(l)} \\in \\mathbb{R}^{n_l \\times n_{l-1}}$$

Non-linear activation:
$$a^{(l)} = \\sigma(z^{(l)})$$

Common activations:
- Sigmoid: $\\sigma(z) = \\frac{1}{1 + e^{-z}}$
- ReLU: $\\sigma(z) = \\max(0, z)$
- Tanh: $\\sigma(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$

**Output Layer:**
$$\\hat{y} = a^{(L)} \\in \\mathbb{R}^{n_L}$$

**Loss Function (Mean Squared Error):**
$$\\mathcal{L} = \\frac{1}{2N} \\sum_{i=1}^{N} ||\\hat{y}_i - y_i||^2$$

**Backpropagation (Chain Rule):**

Output layer gradient:
$$\\delta^{(L)} = (\\hat{y} - y) \\odot \\sigma'(z^{(L)})$$

Hidden layer gradient:
$$\\delta^{(l)} = (W^{(l+1)})^T \\delta^{(l+1)} \\odot \\sigma'(z^{(l)})$$

Weight gradient:
$$\\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}} = \\delta^{(l)} (a^{(l-1)})^T$$

**Parameter Update (Gradient Descent):**
$$W^{(l)} \\leftarrow W^{(l)} - \\eta \\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}}$$
$$b^{(l)} \\leftarrow b^{(l)} - \\eta \\frac{\\partial \\mathcal{L}}{\\partial b^{(l)}}$$

**Weight Initialization (Xavier/Glorot):**
$$W^{(l)} \\sim \\mathcal{N}(0, \\sqrt{\\frac{2}{n_l + n_{l-1}}})$$

**Universal Approximation Theorem:**
A feedforward network with one hidden layer with sigmoid activation can approximate any continuous function on compact domain.
    """)

    # ── Config ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_hidden  = st.slider("Hidden Layers", 1, 4, 2)
    with c2:
        neurons   = st.slider("Neurons per Hidden Layer", 2, 16, 4)
    with c3:
        act_name  = st.selectbox("Activation", ["sigmoid","relu","tanh","leaky_relu"])
    with c4:
        dataset   = st.selectbox("Dataset", ["XOR", "AND", "OR"])

    c5, c6 = st.columns(2)
    with c5:
        lr     = st.slider("Learning Rate", 0.001, 1.0, 0.05, 0.001, format="%.3f")
    with c6:
        epochs = st.slider("Epochs", 10, 1000, 200)

    ds_map = {
        "XOR": (np.array([[0,0],[0,1],[1,0],[1,1]],float), np.array([[0],[1],[1],[0]],float)),
        "AND": (np.array([[0,0],[0,1],[1,0],[1,1]],float), np.array([[0],[0],[0],[1]],float)),
        "OR":  (np.array([[0,0],[0,1],[1,0],[1,1]],float), np.array([[0],[1],[1],[1]],float)),
    }
    X, y = ds_map[dataset]
    sizes = [2] + [neurons]*n_hidden + [1]

    step_badge(1, "Network Architecture")
    st.markdown(
        f'**`{" → ".join(str(s) for s in sizes)}`**  '
        f'(input → {n_hidden} hidden → output)'
    )
    st.markdown(architecture_svg(sizes), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["🔬 Forward Pass Inspector", "📈 Train & Loss Curve", "🗺️ Decision Surface"])

    # ── Forward Inspector ─────────────────────────────────────────────────
    with tab1:
        step_badge(2, "Inspect a Single Sample")
        np.random.seed(42)
        Ws0, bs0 = build_net(sizes)
        sample   = st.slider("Sample (0-3)", 0, 3, 0, key="mlp_s")
        xi       = X[sample]
        _, acts, zs = forward(xi, Ws0, bs0, act_name, return_all=True)

        st.markdown(f"**Input:** `x = {xi}`  |  **True label:** `{int(y[sample][0])}`")

        for li, (z, a) in enumerate(zip(zs, acts[1:])):
            ltype = "Output" if li == len(zs)-1 else f"Hidden {li+1}"
            with st.expander(f"Layer {li+1} — {ltype}  ({len(z)} neurons)",
                             expanded=(li < 2)):
                cc1, cc2 = st.columns(2)
                cc1.markdown("**z (pre-activation)**")
                cc1.dataframe(
                    pd.DataFrame({"neuron": range(1,len(z)+1),
                                  "z": np.round(z, 6)}),
                    use_container_width=True)
                cc2.markdown(f"**a = {act_name}(z)**")
                cc2.dataframe(
                    pd.DataFrame({"neuron": range(1,len(a)+1),
                                  "a": np.round(a, 6)}),
                    use_container_width=True)

        formula_box(r"z^{(l)} = W^{(l)} \cdot a^{(l-1)} + b^{(l)} \qquad "
                    r"a^{(l)} = \sigma(z^{(l)})",
                    "Forward propagation through layer l")

    # ── Training ──────────────────────────────────────────────────────────
    with tab2:
        if st.button("▶  Train MLP", use_container_width=True, key="mlp_train"):
            with st.spinner("Training…"):
                Ws_t, bs_t, loss_hist = train(X, y, sizes, act_name, lr, epochs)

            st.session_state["mlp_Ws"]   = Ws_t
            st.session_state["mlp_bs"]   = bs_t
            st.session_state["mlp_loss"] = loss_hist
            st.session_state["mlp_sizes"] = sizes
            st.session_state["mlp_act"]  = act_name

        if "mlp_loss" in st.session_state:
            lh = st.session_state["mlp_loss"]
            fig = go.Figure(go.Scatter(
                y=lh, mode="lines",
                line=dict(color="#10b981", width=2)))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,1)",
                font_family="Space Mono",
                xaxis_title="Epoch", yaxis_title="MSE Loss",
                height=320, margin=dict(l=20,r=20,t=36,b=20),
                title="Training Loss",
            )
            st.plotly_chart(fig, use_container_width=True)

            metric_row([
                ("Final Loss", f"{lh[-1]:.6f}"),
                ("Min Loss",   f"{min(lh):.6f}"),
                ("Epochs",     len(lh)),
            ])

            # Predictions
            step_badge(3, "Final Predictions")
            rows = []
            for xi, yi in zip(X, y):
                out = forward(xi,
                              st.session_state["mlp_Ws"],
                              st.session_state["mlp_bs"],
                              st.session_state["mlp_act"])
                rows.append({"x": xi.tolist(), "target": int(yi[0]),
                             "raw output": round(float(out[0]), 5),
                             "predicted class": int(float(out[0]) >= 0.5)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Decision Surface ──────────────────────────────────────────────────
    with tab3:
        if "mlp_Ws" in st.session_state:
            st.plotly_chart(
                decision_surface(X, y,
                                 st.session_state["mlp_Ws"],
                                 st.session_state["mlp_bs"],
                                 st.session_state["mlp_act"]),
                use_container_width=True)
        else:
            st.info("Train the model first (Tab 2) to see the decision surface.")
