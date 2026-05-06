"""
models/cnn.py
==============
Manual 2-D Convolutional Neural Network demonstrator.

Pure NumPy convolution — no PyTorch / TensorFlow.
Shows:
  • Input → Convolution → ReLU → Feature Map → Max Pooling
  • Step-by-step spotlight for one output position
  • Multi-kernel feature map comparison
  • Effect of stride and padding
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
from PIL import Image
from components.ui import theory_box, formula_box, step_badge, metric_row


# ──────────────────────────────────────────────────────────────────────────────
# Predefined kernels
# ──────────────────────────────────────────────────────────────────────────────

KERNELS = {
    "Edge — Sobel X":   np.array([[-1,0,1],[-2,0,2],[-1,0,1]],   float),
    "Edge — Sobel Y":   np.array([[-1,-2,-1],[0,0,0],[1,2,1]],    float),
    "Sharpen":          np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],   float),
    "Gaussian Blur":    np.array([[1,2,1],[2,4,2],[1,2,1]],       float) / 16,
    "Emboss":           np.array([[-2,-1,0],[-1,1,1],[0,1,2]],    float),
    "Identity":         np.array([[0,0,0],[0,1,0],[0,0,0]],       float),
    "Laplacian":        np.array([[0,1,0],[1,-4,1],[0,1,0]],      float),
    "Box Blur":         np.ones((3,3), float) / 9,
}


# ──────────────────────────────────────────────────────────────────────────────
# Core ops (pure NumPy)
# ──────────────────────────────────────────────────────────────────────────────

def convolve2d(img: np.ndarray, kernel: np.ndarray,
               stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    2-D convolution (valid or same with explicit padding).
    Steps per output position:
      z[i,j] = Σ Σ kernel[m,n] · img_padded[i*s+m, j*s+n]
    """
    if pad > 0:
        img = np.pad(img, pad, mode="constant", constant_values=0)
    kh, kw = kernel.shape
    oh = (img.shape[0] - kh) // stride + 1
    ow = (img.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            r, c = i * stride, j * stride
            out[i, j] = np.sum(img[r:r+kh, c:c+kw] * kernel)
    return out


def relu2d(fm: np.ndarray) -> np.ndarray:
    return np.maximum(0, fm)


def max_pool(fm: np.ndarray, size: int = 2) -> np.ndarray:
    ph, pw = fm.shape[0]//size, fm.shape[1]//size
    out = np.zeros((ph, pw))
    for i in range(ph):
        for j in range(pw):
            out[i,j] = np.max(fm[i*size:(i+1)*size, j*size:(j+1)*size])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic images
# ──────────────────────────────────────────────────────────────────────────────

def make_pattern(kind: str, size: int = 32) -> np.ndarray:
    img = np.zeros((size, size))
    if kind == "Diagonal lines":
        for i in range(size):
            img[i, i % size] = 220
            img[i, (i+6) % size] = 150
    elif kind == "Circle":
        cx = cy = size // 2
        r = size // 3
        Y, X = np.ogrid[:size, :size]
        mask = np.abs(np.sqrt((X-cx)**2 + (Y-cy)**2) - r) < 1.5
        img[mask] = 220
    elif kind == "Checkerboard":
        b = size // 8
        for i in range(size):
            for j in range(size):
                if (i//b + j//b) % 2 == 0:
                    img[i,j] = 220
    elif kind == "Gradient":
        img = np.tile(np.linspace(0, 255, size), (size, 1))
    elif kind == "Cross":
        mid = size // 2
        img[mid-2:mid+2, :] = 220
        img[:, mid-2:mid+2] = 220
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap helper
# ──────────────────────────────────────────────────────────────────────────────

def hm(z, cs="Viridis"):
    return go.Heatmap(z=z, colorscale=cs, showscale=True)


_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    font_family="Space Mono",
    margin=dict(l=10, r=10, t=50, b=10),
)


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

def run():
    theory_box("Convolutional Neural Networks: Mathematical Foundation", """
**Convolution Operation:**

Given input $X \\in \\mathbb{R}^{H \\times W}$ and kernel $K \\in \\mathbb{R}^{K_h \\times K_w}$:

Output element at position $(i,j)$:
$$Y[i,j] = \\sum_{m=0}^{K_h-1} \\sum_{n=0}^{K_w-1} K[m,n] \\cdot X[i+m, j+n] + b$$

This is a **linear operation** — same weights applied everywhere (weight sharing).

**Output Dimensions:**

With stride $s$ and padding $p$:
$$H_{out} = \\frac{H - K_h + 2p}{s} + 1$$
$$W_{out} = \\frac{W - K_w + 2p}{s} + 1$$

**Multiple Filters:**

With $F$ filters, output is $H_{out} \\times W_{out} \\times F$ (feature maps)

**Max Pooling:**

Divides feature maps into non-overlapping regions of size $P \\times P$:
$$Y[i,j] = \\max_{0 \\le m,n < P} X[iP+m, jP+n]$$

Reduces spatial dimensions by factor of $P$, adds translation robustness.

**Full CNN Pipeline:**

1. Input: $X \\in \\mathbb{R}^{H \\times W \\times C}$ (C channels)
2. Conv filters: $F$ kernels of size $K_h \\times K_w \\times C$
3. Activation: $a = \\text{ReLU}(z)$
4. Pooling: Reduce spatial size
5. Flatten: $\\text{vec}(a) \\in \\mathbb{R}^{\\text{size}}$
6. Fully Connected: Classification head

**Weight Sharing Benefits:**

- Vanilla network: $H \\times W \\times K_h \\times K_w$ parameters per layer
- CNN: Only $K_h \\times K_w$ parameters (shared across all spatial positions)
- Reduction: $O(HW)$ factor fewer parameters!

**Translation Equivariance:**

If object shifts, feature activation shifts correspondingly:
$$Y_\\text{shift}[i,j] = Y[i-\\Delta i, j-\\Delta j]$$

**Common Kernels:**

Edge detection (Sobel):
$$K_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{bmatrix}$$

Blur (Gaussian):
$$K = \\frac{1}{16}\\begin{bmatrix} 1 & 2 & 1 \\\\ 2 & 4 & 2 \\\\ 1 & 2 & 1 \\end{bmatrix}$$

**Backpropagation:**

Gradient computation uses convolution with rotated kernel − equally efficient!

$$\\frac{\\partial \\mathcal{L}}{\\partial K} = \\text{conv}(X, \\nabla Y)$$
$$\\frac{\\partial \\mathcal{L}}{\\partial X} = \\text{conv}(\\nabla Y, K_{rotated})$$
    """)

    tab1, tab2, tab3 = st.tabs(
        ["🖼️ Image Pipeline", "🔢 Step-by-Step Spotlight", "🔬 Multi-Kernel Comparison"])

    # ── Tab 1: Image pipeline ─────────────────────────────────────────────
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kernel_name = st.selectbox("Kernel", list(KERNELS.keys()))
        with c2:
            img_src = st.selectbox("Image", ["Synthetic", "Upload"])
        with c3:
            stride = st.slider("Stride", 1, 3, 1, key="cnn_stride")
        with c4:
            pad = st.slider("Padding", 0, 3, 0, key="cnn_pad")

        apply_relu = st.checkbox("Apply ReLU after convolution", value=True)

        if img_src == "Synthetic":
            pattern = st.selectbox("Pattern",
                ["Diagonal lines","Circle","Checkerboard","Gradient","Cross"])
            img = make_pattern(pattern, 32).astype(float)
        else:
            up = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
            if up is None:
                st.info("Upload an image to continue.")
                return
            img = np.array(Image.open(up).convert("L").resize((64,64)), float)

        kernel = KERNELS[kernel_name]
        fm = convolve2d(img, kernel, stride, pad)
        if apply_relu:
            fm = relu2d(fm)
        pooled = max_pool(fm)

        step_badge(1, "Pipeline: Input → Conv+ReLU → MaxPool")

        fig = make_subplots(1, 3,
            subplot_titles=["Input", "Feature Map (conv+ReLU)", "After MaxPool (2×2)"])
        fig.add_trace(hm(img, "gray"),     1, 1)
        fig.add_trace(hm(fm, "Viridis"),   1, 2)
        fig.add_trace(hm(pooled, "Plasma"),1, 3)
        fig.update_layout(**_DARK, height=380)
        st.plotly_chart(fig, use_container_width=True)

        metric_row([
            ("Input size",     f"{img.shape[0]}×{img.shape[1]}"),
            ("Feature map",    f"{fm.shape[0]}×{fm.shape[1]}"),
            ("After pooling",  f"{pooled.shape[0]}×{pooled.shape[1]}"),
            ("Params (kernel)",f"{kernel.size}"),
        ])

        st.markdown("**Kernel used:**")
        st.dataframe(pd.DataFrame(np.round(kernel,4)), use_container_width=False)

    # ── Tab 2: Spotlight ──────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Trace a single output position on a 6×6 patch")
        default_patch = np.array([
            [10, 50, 80, 120, 90, 30],
            [20, 70,100, 140,110, 40],
            [30, 60, 90, 110,100, 50],
            [40, 80,120,  90, 70, 60],
            [20, 50, 70,  80, 60, 30],
            [10, 30, 50,  60, 40, 20],
        ], float)

        ker_c = st.selectbox("Kernel", list(KERNELS.keys()), key="cnn_sp_ker")
        ker   = KERNELS[ker_c]

        c_r, c_c = st.columns(2)
        with c_r:
            ri = st.slider("Output row",    0, 3, 0, key="cnn_sp_r")
        with c_c:
            ci = st.slider("Output column", 0, 3, 0, key="cnn_sp_c")

        patch     = default_patch[ri:ri+3, ci:ci+3]
        product   = patch * ker
        result    = product.sum()

        step_badge(2, f"Computing output[{ri},{ci}]")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Patch slice**")
            st.dataframe(pd.DataFrame(patch.astype(int)), use_container_width=True)
        with col2:
            st.markdown("**Kernel**")
            st.dataframe(pd.DataFrame(np.round(ker,4)), use_container_width=True)
        with col3:
            st.markdown("**Element-wise ×**")
            st.dataframe(pd.DataFrame(np.round(product,2)), use_container_width=True)

        formula_box(r"F[" + str(ri) + "," + str(ci) + r"] = \sum_{m}\sum_{n} K[m,n]\cdot P[m,n]",
                    f"= {round(result,4)}")
        st.success(f"Output value at [{ri},{ci}] = **{round(result,4)}**  "
                   f"→ after ReLU = **{round(max(0,result),4)}**")

    # ── Tab 3: Multi-kernel ───────────────────────────────────────────────
    with tab3:
        st.markdown("#### See how different kernels respond to the same image")
        pattern_m = st.selectbox("Pattern", ["Circle","Checkerboard","Cross"], key="cnn_mk")
        img_m = make_pattern(pattern_m, 32).astype(float)

        selected_kernels = st.multiselect(
            "Kernels to compare", list(KERNELS.keys()),
            default=["Edge — Sobel X","Edge — Sobel Y","Sharpen","Gaussian Blur"])

        if len(selected_kernels) == 0:
            st.info("Select at least one kernel.")
            return

        n = len(selected_kernels)
        fig2 = make_subplots(1, n+1,
            subplot_titles=["Input"] + selected_kernels)
        fig2.add_trace(hm(img_m, "gray"), 1, 1)
        for idx, kname in enumerate(selected_kernels):
            fm2 = relu2d(convolve2d(img_m, KERNELS[kname]))
            fig2.add_trace(hm(fm2, "Viridis"), 1, idx+2)

        fig2.update_layout(**_DARK, height=340)
        st.plotly_chart(fig2, use_container_width=True)
