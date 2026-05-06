"""
models/rnn.py
==============
Vanilla RNN for next-word prediction — fully manual NumPy.

Architecture:
  h_t = tanh(Wxh · x_t + Whh · h_{t-1} + b_h)
  y_t = softmax(Why · h_t + b_y)

Training: BPTT (truncated, seq_len steps)
Sampling: temperature-scaled softmax
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from components.ui import theory_box, formula_box, step_badge, metric_row, PALETTE


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def softmax(z, T=1.0):
    z = z.flatten() / T
    e = np.exp(z - np.max(z))
    return e / (e.sum() + 1e-12)


def build_vocab(text: str):
    words = text.lower().split()
    vocab = sorted(set(words))
    w2i   = {w: i for i, w in enumerate(vocab)}
    i2w   = {i: w for w, i in w2i.items()}
    return words, vocab, w2i, i2w


def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1.0
    return v


# ──────────────────────────────────────────────────────────────────────────────
# Vanilla RNN
# ──────────────────────────────────────────────────────────────────────────────

class VanillaRNN:
    def __init__(self, V, H, lr):
        self.V, self.H, self.lr = V, H, lr
        # Xavier-ish init
        self.Wxh = np.random.randn(H, V) * np.sqrt(2/(H+V))
        self.Whh = np.random.randn(H, H) * np.sqrt(2/(H+H))
        self.bh  = np.zeros((H, 1))
        self.Why = np.random.randn(V, H) * np.sqrt(2/(V+H))
        self.by  = np.zeros((V, 1))
        # AdaGrad memory
        self.mWxh = np.ones_like(self.Wxh)
        self.mWhh = np.ones_like(self.Whh)
        self.mWhy = np.ones_like(self.Why)
        self.mbh  = np.ones_like(self.bh)
        self.mby  = np.ones_like(self.by)

    def forward(self, inputs_oh, h_prev):
        """
        inputs_oh: list of one-hot (V,) arrays
        Returns xs, hs, ys, ps dicts keyed by time step.
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = h_prev.copy()
        for t, xv in enumerate(inputs_oh):
            xs[t] = xv.reshape(-1, 1)
            hs[t] = np.tanh(
                self.Wxh @ xs[t] +
                self.Whh @ hs[t-1] +
                self.bh
            )
            ys[t] = self.Why @ hs[t] + self.by
            ps[t] = softmax(ys[t]).reshape(-1, 1)
        return xs, hs, ys, ps

    def loss_grads(self, input_idx, target_idx, h_prev):
        """BPTT over one sequence chunk."""
        T      = len(input_idx)
        inp_oh = [one_hot(i, self.V) for i in input_idx]
        xs, hs, ys, ps = self.forward(inp_oh, h_prev)

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh  = np.zeros_like(self.bh)
        dby  = np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        loss = 0.0

        for t in reversed(range(T)):
            loss -= np.log(ps[t][target_idx[t], 0] + 1e-12)
            dy = ps[t].copy()
            dy[target_idx[t]] -= 1
            dWhy += dy @ hs[t].T
            dby  += dy
            dh   = self.Why.T @ dy + dhnext
            dhr  = (1 - hs[t]**2) * dh
            dbh  += dhr
            dWxh += dhr @ xs[t].T
            dWhh += dhr @ hs[t-1].T
            dhnext = self.Whh.T @ dhr

        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)

        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[T-1]

    def adagrad_update(self, dWxh, dWhh, dWhy, dbh, dby):
        eps = 1e-8
        for param, dparam, mem in [
            (self.Wxh, dWxh, self.mWxh),
            (self.Whh, dWhh, self.mWhh),
            (self.Why, dWhy, self.mWhy),
            (self.bh,  dbh,  self.mbh),
            (self.by,  dby,  self.mby),
        ]:
            mem += dparam * dparam
            param -= self.lr * dparam / np.sqrt(mem + eps)

    def predict_next(self, seed_words, w2i, i2w, top_k=10, temperature=1.0):
        h = np.zeros((self.H, 1))
        for word in seed_words:
            idx = w2i.get(word.lower(), 0)
            xv  = one_hot(idx, self.V).reshape(-1, 1)
            h   = np.tanh(self.Wxh @ xv + self.Whh @ h + self.bh)

        logits = self.Why @ h + self.by
        probs  = softmax(logits, T=temperature).flatten()
        top_idx = np.argsort(probs)[::-1][:top_k]
        return [(i2w[i], round(float(probs[i]), 6)) for i in top_idx], h

    def generate(self, seed_word, w2i, i2w, n_words=10, temperature=1.0):
        h = np.zeros((self.H, 1))
        idx = w2i.get(seed_word.lower(), 0)
        out = [seed_word]
        for _ in range(n_words):
            xv  = one_hot(idx, self.V).reshape(-1, 1)
            h   = np.tanh(self.Wxh @ xv + self.Whh @ h + self.bh)
            logits = self.Why @ h + self.by
            probs  = softmax(logits, T=temperature).flatten()
            idx    = np.random.choice(len(probs), p=probs)
            out.append(i2w.get(idx, "<unk>"))
        return " ".join(out)


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,1)",
    font_family="Space Mono",
    margin=dict(l=20, r=20, t=40, b=20),
)


def run():
    theory_box("Recurrent Neural Network: Sequence Modeling", """
**Key Innovation:** Hidden state maintains memory of past inputs

**RNN Forward Pass:**

At time step $t$, given input $x_t$ and previous state $h_{t-1}$:

Hidden state update:
$$h_t = \\tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

Output prediction:
$$y_t = \\text{softmax}(W_y h_t + b_y)$$

Loss at step $t$:
$$\\mathcal{L}_t = -\\log(y_t[w_t])$$

Total sequence loss:
$$\\mathcal{L} = \\frac{1}{T} \\sum_{t=1}^{T} \\mathcal{L}_t$$

**Backpropagation Through Time (BPTT):**

Error at output:
$$\\delta_t^{(y)} = y_t - \\text{onehot}(w_t)$$

Hidden state gradient:
$$\\delta_t^{(h)} = (W_y^T \\delta_t^{(y)} + W_{hh}^T \\delta_{t+1}^{(h)}) \\odot \\text{tanh}'(h_t)$$

Weight gradients (accumulated over all time steps):
$$\\frac{\\partial \\mathcal{L}}{\\partial W_{xh}} = \\sum_{t=1}^{T} \\delta_t^{(h)} x_t^T$$
$$\\frac{\\partial \\mathcal{L}}{\\partial W_{hh}} = \\sum_{t=1}^{T} \\delta_t^{(h)} h_{t-1}^T$$
$$\\frac{\\partial \\mathcal{L}}{\\partial W_y} = \\sum_{t=1}^{T} \\delta_t^{(y)} h_t^T$$

**Vanishing Gradient Problem:**

Gradient at early step $\\tau$:
$$\\frac{\\partial \\mathcal{L}_T}{\\partial h_\\tau} = \\prod_{t=\\tau+1}^{T} (W_{hh}^T \\odot \\tanh'(h_t))$$

When $|W_{hh} \\tanh'| < 1$:
$$\\left|\\prod_{t=\\tau}^{T} W_{hh}^T \\tanh'\\right| \\approx \\alpha^{T-\\tau} \\to 0$$

**Temperature Sampling:**

Modified softmax for generation:
$$P(w_i | T) = \\frac{e^{z_i/T}}{\\sum_j e^{z_j/T}}$$

- $T \\to 0$: greedy selection
- $T = 1$: standard softmax
- $T \\to \\infty$: uniform random

**Solutions to Vanishing Gradients:**
- LSTM/GRU with gating mechanisms
- Gradient clipping: $||\\nabla|| > \\theta \\Rightarrow \\nabla := \\theta \\frac{\\nabla}{||\\nabla||}$
- RMSprop/Adam optimization
    """)

    # ── Corpus ────────────────────────────────────────────────────────────
    default_text = (
        "the cat sat on the mat the cat ate the rat "
        "the dog sat on the log the dog chased the cat "
        "the rat ran from the cat the mat was on the floor "
        "the floor was cold the cat was warm the dog was happy "
        "the happy dog chased the cold rat on the mat"
    )
    text = st.text_area("Training Corpus", value=default_text, height=110)

    c1, c2, c3 = st.columns(3)
    with c1:
        H  = st.slider("Hidden Units", 8, 64, 24)
    with c2:
        lr = st.slider("Learning Rate", 0.001, 0.5, 0.05, 0.001, format="%.3f")
    with c3:
        epochs = st.slider("Epochs", 10, 600, 150)

    words, vocab, w2i, i2w = build_vocab(text)
    V = len(vocab)

    step_badge(1, f"Corpus stats — {V} unique words, {len(words)} tokens")
    word_freq = pd.Series(words).value_counts().head(10).reset_index()
    word_freq.columns = ["word","count"]
    st.dataframe(word_freq, use_container_width=False)

    if V < 3:
        st.warning("Add more text.")
        return

    if st.button("▶  Train RNN", use_container_width=True):
        rnn  = VanillaRNN(V, H, lr)
        indices  = [w2i[w] for w in words]
        seq_len  = min(8, len(indices) - 1)
        loss_hist = []

        prog = st.progress(0)
        for ep in range(epochs):
            ep_loss = 0.0
            h = np.zeros((H, 1))
            for start in range(0, len(indices) - seq_len - 1, seq_len):
                inp = indices[start:start+seq_len]
                tgt = indices[start+1:start+seq_len+1]
                loss, *grads, h = rnn.loss_grads(inp, tgt, h)
                rnn.adagrad_update(*grads)
                ep_loss += loss
            loss_hist.append(ep_loss)
            prog.progress((ep+1)/epochs)

        st.session_state.update(
            rnn_model=rnn, rnn_w2i=w2i,
            rnn_i2w=i2w, rnn_loss=loss_hist, rnn_V=V, rnn_H=H)
        st.success("Training complete!")

    # ── Post-train panels ─────────────────────────────────────────────────
    if "rnn_model" not in st.session_state:
        return

    rnn  = st.session_state.rnn_model
    w2i_ = st.session_state.rnn_w2i
    i2w_ = st.session_state.rnn_i2w
    loss_hist = st.session_state.rnn_loss

    tab1, tab2, tab3 = st.tabs(
        ["📊 Loss Curve", "🔮 Next-Word Prediction", "✍️ Text Generation"])

    with tab1:
        fig = go.Figure(go.Scatter(
            y=loss_hist, mode="lines",
            line=dict(color="#ec4899", width=2)))
        fig.update_layout(**_DARK, height=300, title="Cross-Entropy Loss",
                          xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)
        metric_row([("Final Loss", f"{loss_hist[-1]:.4f}"),
                    ("Min Loss",   f"{min(loss_hist):.4f}"),
                    ("Vocab Size", st.session_state.rnn_V)])

    with tab2:
        seed = st.text_input("Seed word(s)", value="the cat")
        temp = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1, key="rnn_temp")
        seed_words = seed.lower().split()
        unk = [w for w in seed_words if w not in w2i_]
        if unk:
            st.warning(f"Unknown words: {unk}")
        else:
            preds, h_state = rnn.predict_next(seed_words, w2i_, i2w_,
                                              top_k=min(10, len(w2i_)),
                                              temperature=temp)
            top5 = preds[:5]
            fig2 = go.Figure(go.Bar(
                x=[p[0] for p in top5],
                y=[p[1] for p in top5],
                marker_color=PALETTE[:5],
            ))
            fig2.update_layout(**_DARK, height=280,
                               xaxis_title="Next word",
                               yaxis_title="Probability",
                               title="Top-5 Predictions")
            st.plotly_chart(fig2, use_container_width=True)
            st.success(f"🔮 Predicted next word: **{top5[0][0]}**  "
                       f"(p ≈ {top5[0][1]:.4f})")

            with st.expander("🔬 Hidden State Heatmap"):
                fig3 = go.Figure(go.Heatmap(
                    z=h_state.T, colorscale="RdBu", showscale=True))
                fig3.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=120, margin=dict(l=10,r=10,t=20,b=10),
                    title="h_t hidden state vector",
                    font_family="Space Mono",
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.caption(f"Hidden state dim: {h_state.shape[0]}  |  "
                           f"min={h_state.min():.3f}  max={h_state.max():.3f}")

    with tab3:
        seed_gen = st.text_input("Start word", value="the", key="rnn_gen")
        n_gen    = st.slider("Words to generate", 5, 30, 12)
        temp_gen = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1, key="rnn_temp_gen")

        if st.button("✍️ Generate text", key="rnn_generate"):
            if seed_gen.lower() not in w2i_:
                st.warning("Seed word not in vocabulary.")
            else:
                generated = rnn.generate(seed_gen, w2i_, i2w_,
                                          n_gen, temperature=temp_gen)
                st.markdown(
                    f'<div style="background:rgba(124,58,237,0.12);border:1px solid '
                    f'rgba(124,58,237,0.3);border-radius:10px;padding:16px 20px;'
                    f'font-family:Space Mono,monospace;font-size:.95rem;'
                    f'color:#e2e8f0;line-height:1.8;">{generated}</div>',
                    unsafe_allow_html=True)
        formula_box(
            r"h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)",
            "RNN hidden state update — the core recurrence relation")
