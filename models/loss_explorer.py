"""
models/loss_explorer.py
========================
Interactive Loss Function Explorer.
- MSE: mean((y - ŷ)²)
- Binary Cross-Entropy: -[y·log(ŷ) + (1-y)·log(1-ŷ)]
- Categorical Cross-Entropy
- Hinge Loss
- Shows loss curves, gradient of loss, and effect of outliers
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd


# ── Loss functions ────────────────────────────────────────────────────────────
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_true - y_pred) ** 2

def mse_mean(y_true, y_pred) -> float:
    return float(np.mean(mse(y_true, y_pred)))

def bce(y_true: np.ndarray, y_pred: np.ndarray,
        eps: float = 1e-12) -> np.ndarray:
    yp = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(yp) + (1 - y_true) * np.log(1 - yp))

def bce_mean(y_true, y_pred) -> float:
    return float(np.mean(bce(y_true, y_pred)))

def hinge(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """y_true ∈ {-1, +1}, y_pred ∈ ℝ."""
    return np.maximum(0, 1 - y_true * y_pred)

def hinge_mean(y_true, y_pred) -> float:
    return float(np.mean(hinge(y_true, y_pred)))


# ── Simulate training curves ──────────────────────────────────────────────────
def simulate_curve(loss_fn_name: str, lr: float, epochs: int, noise: float):
    """
    Simulate a loss curve for a logistic regression on a separable 2-class dataset.
    Returns: loss_values (list)
    """
    np.random.seed(7)
    N = 30
    X = np.vstack([np.random.randn(N//2, 2) + [2, 2],
                   np.random.randn(N//2, 2) + [-2, -2]])
    y = np.array([1]*(N//2) + [0]*(N//2), dtype=float)

    w = np.zeros(2)
    b = 0.0
    history = []

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    for _ in range(epochs):
        z     = X @ w + b
        y_hat = sigmoid(z)

        if loss_fn_name == "MSE":
            loss = mse_mean(y, y_hat)
            grad = (y_hat - y) * y_hat * (1 - y_hat)
        elif loss_fn_name == "Binary Cross-Entropy":
            loss = bce_mean(y, y_hat)
            grad = y_hat - y
        else:  # Hinge — use raw scores
            y_hinge = 2*y - 1   # convert {0,1} → {-1,+1}
            loss = hinge_mean(y_hinge, z)
            margin = y_hinge * z
            grad = np.where(margin < 1, -y_hinge, 0)

        history.append(loss + np.random.randn() * noise)
        dw = X.T @ grad / N
        db = grad.mean()
        w -= lr * dw
        b -= lr * db

    return history


# ── UI ────────────────────────────────────────────────────────────────────────
def run():
    st.info("📖 **Loss Function Explorer** — The loss quantifies how wrong our model is. "
            "Different tasks require different losses: "
            "**MSE** for regression, **cross-entropy** for classification, "
            "**hinge** for SVMs.")

    tab1, tab2, tab3 = st.tabs(
        ["📈 Training Curves", "🔍 Loss Shape", "⚖️ Loss Comparison Table"])

    # ── Shared controls ───────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        lr      = st.slider("Learning Rate", 0.001, 1.0, 0.1, step=0.001,
                            format="%.3f", key="lx_lr")
    with c2:
        epochs  = st.slider("Epochs", 10, 500, 150, key="lx_ep")
    with c3:
        noise   = st.slider("Noise Level", 0.0, 0.2, 0.02, step=0.01,
                            key="lx_noise")

    LOSS_NAMES  = ["MSE", "Binary Cross-Entropy", "Hinge"]
    COLORS      = {"MSE": "#00d4ff",
                   "Binary Cross-Entropy": "#7c3aed",
                   "Hinge": "#f59e0b"}

    with tab1:
        selected = st.multiselect("Loss functions to compare",
                                  LOSS_NAMES, default=LOSS_NAMES)
        if st.button("▶ Simulate Training", use_container_width=True, key="lx_run"):
            fig = go.Figure()
            for name in selected:
                hist = simulate_curve(name, lr, epochs, noise)
                fig.add_trace(go.Scatter(
                    y=hist, mode="lines",
                    name=name,
                    line=dict(color=COLORS[name], width=2)
                ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,1)",
                font_family="Space Mono",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Loss vs Prediction (for a single sample, y_true = 1)")
        y_pred_range = np.linspace(0.001, 0.999, 200)
        y_true_val   = 1.0

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=y_pred_range,
            y=mse(np.array([y_true_val]), y_pred_range.reshape(-1,1)).flatten(),
            mode="lines", name="MSE",
            line=dict(color=COLORS["MSE"], width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=y_pred_range,
            y=bce(np.array([y_true_val]), y_pred_range.reshape(-1,1)).flatten(),
            mode="lines", name="BCE",
            line=dict(color=COLORS["Binary Cross-Entropy"], width=2)
        ))

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,1)",
            font_family="Space Mono",
            xaxis_title="ŷ (prediction)",
            yaxis_title="Loss value",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Gradient of BCE
        st.markdown("#### Gradient of BCE w.r.t. ŷ  (∂L/∂ŷ)")
        eps = 1e-12
        grad_bce = -(y_true_val / (y_pred_range + eps) -
                     (1 - y_true_val) / (1 - y_pred_range + eps))
        fig3 = go.Figure(go.Scatter(
            x=y_pred_range, y=grad_bce, mode="lines",
            line=dict(color="#10b981", width=2),
        ))
        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,1)",
            font_family="Space Mono",
            xaxis_title="ŷ",
            yaxis_title="∂L/∂ŷ",
            height=260,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        y_preds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_true_arr = np.ones(len(y_preds))

        rows = []
        for yp in y_preds:
            rows.append({
                "ŷ (prediction)": yp,
                "y (true)": 1,
                "MSE Loss":  round(mse_mean(np.array([1.0]), np.array([yp])), 5),
                "BCE Loss":  round(bce_mean(np.array([1.0]), np.array([yp])), 5),
                "Hinge Loss (y=+1)": round(float(np.maximum(0, 1 - yp)), 5),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        st.markdown("""
        **Key Observations:**
        - BCE penalises confident wrong predictions much harder than MSE.
        - MSE gradients saturate near 0 or 1 for sigmoid outputs (slow learning).
        - Hinge loss is zero when the margin `y·ŷ ≥ 1` is satisfied.
        """)
