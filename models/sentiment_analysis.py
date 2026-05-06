"""
models/sentiment_analysis.py
=============================
LSTM-based Sentiment Analysis Model for binary classification (Positive/Negative).

Architecture:
  - LSTM cells with gating mechanisms (forget, input, output gates)
  - Embedding layer for word representations
  - FC layer for binary classification

Training: BPTT with truncated sequences
Inference: Forward pass with sigmoid output for binary classification
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components.ui import (
    theory_box, formula_box, step_badge, metric_row, PALETTE,
    apply_dark, loss_curve, section
)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities & Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid(x):
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(y):
    """Derivative of sigmoid."""
    return y * (1 - y)


def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU."""
    return (x > 0).astype(float)


def softmax(x, axis=1):
    """Softmax normalization."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-12)


def preprocess_text(text: str):
    """Simple text preprocessing."""
    text = text.lower()
    text = text.replace('<br>', ' ')
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    words = text.split()
    return [w for w in words if len(w) > 0]


def build_vocab(texts: list, max_vocab: int = 5000):
    """Build vocabulary from list of texts."""
    word_freq = {}
    for text in texts:
        words = preprocess_text(text)
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and keep top words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab-2]
    vocab = ['<PAD>', '<UNK>'] + [w for w, _ in sorted_words]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return vocab, w2i, i2w


def encode_text(text: str, w2i: dict, max_len: int = 100):
    """Encode text to indices with padding."""
    words = preprocess_text(text)
    indices = [w2i.get(w, 1) for w in words]  # 1 is <UNK>
    
    # Pad or truncate
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))  # 0 is <PAD>
    else:
        indices = indices[:max_len]
    
    return np.array(indices)


def create_embeddings(vocab_size: int, embedding_dim: int):
    """Initialize random word embeddings."""
    return np.random.randn(vocab_size, embedding_dim) * np.sqrt(2.0 / vocab_size)


# ──────────────────────────────────────────────────────────────────────────────
# LSTM Cell
# ──────────────────────────────────────────────────────────────────────────────

class LSTMCell:
    """Single LSTM Cell with gating mechanisms."""
    
    def __init__(self, input_dim: int, hidden_dim: int, lr: float = 0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # Gates: forget, input, output, cell
        # Each gate: weight matrix (hidden_dim, input_dim + hidden_dim)
        gate_size = hidden_dim
        self.Wf = np.random.randn(gate_size, input_dim + hidden_dim) * 0.01
        self.Wi = np.random.randn(gate_size, input_dim + hidden_dim) * 0.01
        self.Wo = np.random.randn(gate_size, input_dim + hidden_dim) * 0.01
        self.Wc = np.random.randn(gate_size, input_dim + hidden_dim) * 0.01
        
        self.bf = np.zeros((gate_size, 1))
        self.bi = np.zeros((gate_size, 1))
        self.bo = np.zeros((gate_size, 1))
        self.bc = np.zeros((gate_size, 1))
        
        # Clip gradients
        self.clip_value = 5.0
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass through LSTM cell.
        x: (input_dim,) input vector
        h_prev: (hidden_dim,) previous hidden state
        c_prev: (hidden_dim,) previous cell state
        
        Returns: (h, c) and cache for backward
        """
        # Concatenate input and hidden state
        x = x.reshape(-1, 1)
        h_prev = h_prev.reshape(-1, 1)
        c_prev = c_prev.reshape(-1, 1)
        z = np.concatenate([x, h_prev], axis=0)
        
        # Forget gate
        f = sigmoid(self.Wf @ z + self.bf)
        
        # Input gate
        i = sigmoid(self.Wi @ z + self.bi)
        
        # Output gate
        o = sigmoid(self.Wo @ z + self.bo)
        
        # Cell candidate
        c_tilde = np.tanh(self.Wc @ z + self.bc)
        
        # New cell state
        c = f * c_prev + i * c_tilde
        
        # New hidden state
        h = o * np.tanh(c)
        
        # Cache for backward
        cache = (x, h_prev, c_prev, f, i, o, c_tilde, c, h, z)
        
        return h.flatten(), c.flatten(), cache
    
    def backward(self, dh, dc, cache):
        """Backward pass through LSTM cell."""
        x, h_prev, c_prev, f, i, o, c_tilde, c, h, z = cache
        
        # Output gate gradient
        do = dh * np.tanh(c)
        do = sigmoid_derivative(o) * do
        
        # Cell state from hidden
        dc_from_h = dh * o * (1 - np.tanh(c)**2)
        dc_total = dc + dc_from_h
        
        # Cell candidate gradient
        dc_tilde = i * (1 - c_tilde**2) * dc_total
        
        # Input gate gradient
        di = c_tilde * (1 - i) * i * dc_total
        
        # Forget gate gradient
        df = c_prev * (1 - f) * f * dc_total
        
        # Cell state previous gradient
        dc_prev = f * dc_total
        
        # Weight gradients (concatenate gates)
        d = np.concatenate([df, di, do, dc_tilde], axis=0)
        
        dWf = df @ z.T
        dWi = di @ z.T
        dWo = do @ z.T
        dWc = dc_tilde @ z.T
        
        dbf = df
        dbi = di
        dbo = do
        dbc = dc_tilde
        
        # Gradient w.r.t. input and previous hidden
        dz = self.Wf.T @ df + self.Wi.T @ di + self.Wo.T @ do + self.Wc.T @ dc_tilde
        dx = dz[:self.input_dim]
        dh_prev = dz[self.input_dim:]
        
        return dx.flatten(), dh_prev.flatten(), dc_prev.flatten(), {
            'dWf': dWf, 'dWi': dWi, 'dWo': dWo, 'dWc': dWc,
            'dbf': dbf, 'dbi': dbi, 'dbo': dbo, 'dbc': dbc
        }


# ──────────────────────────────────────────────────────────────────────────────
# Full LSTM Model
# ──────────────────────────────────────────────────────────────────────────────

class SentimentLSTM:
    """LSTM-based sentiment analysis model."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64,
                 hidden_dim: int = 128, lr: float = 0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # Embedding layer
        self.embeddings = create_embeddings(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = LSTMCell(embedding_dim, hidden_dim, lr)
        
        # Output layer (hidden_dim -> 1)
        self.Wo = np.random.randn(1, hidden_dim) * 0.01
        self.bo = np.zeros((1, 1))
        
        # Optimizer memory (AdaGrad)
        self.mem_Wo = np.ones_like(self.Wo)
        self.mem_bo = np.ones_like(self.bo)
        
        self.loss_history = []
    
    def forward(self, indices):
        """
        Forward pass through entire sequence.
        indices: (seq_len,) array of word indices
        Returns: prediction (0-1), and cache
        """
        seq_len = len(indices)
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        
        caches = []
        embeddings_out = []
        
        for t in range(seq_len):
            # Get embedding
            word_idx = indices[t]
            if word_idx == 0:  # PAD token - no update
                embeddings_out.append(np.zeros(self.embedding_dim))
            else:
                embeddings_out.append(self.embeddings[word_idx].copy())
                # Forward through LSTM
                h, c, cache = self.lstm.forward(embeddings_out[-1], h, c)
                caches.append((t, cache))
        
        # Output from final hidden state
        logit = self.Wo @ h.reshape(-1, 1) + self.bo
        pred = sigmoid(logit[0, 0])
        
        return pred, (h, c, caches, indices, embeddings_out)
    
    def backward(self, target, pred, cache_data):
        """Backward pass."""
        h, c, caches, indices, embeddings_out = cache_data
        
        # Output layer gradient
        dlogit = pred - target
        
        # Gradient of output weights
        dWo = dlogit * h.reshape(1, -1)
        dbo = dlogit
        
        dh = dlogit * self.Wo
        dc = np.zeros(self.hidden_dim)
        
        # Process caches in reverse
        for t, cache in reversed(caches):
            dx, dh, dc, gate_grads = self.lstm.backward(dh.flatten(), dc, cache)
            
            # Update LSTM weights (simplified Adam-like update)
            lr_scaled = self.lr / np.sqrt(1.0 + len(self.loss_history))
            for param, grad in gate_grads.items():
                if 'W' in param:
                    setattr(self.lstm, param, getattr(self.lstm, param) - 
                           lr_scaled * np.clip(grad, -5, 5) / (1.0 + np.abs(grad)))
                else:
                    setattr(self.lstm, param, getattr(self.lstm, param) - 
                           lr_scaled * np.clip(grad, -5, 5) / (1.0 + np.abs(grad)))
        
        # Update output layer
        self.mem_Wo += dWo ** 2
        self.mem_bo += dbo ** 2
        self.Wo -= self.lr * dWo / (np.sqrt(self.mem_Wo) + 1e-8)
        self.bo -= self.lr * dbo / (np.sqrt(self.mem_bo) + 1e-8)
    
    def compute_loss(self, pred, target):
        """Binary cross-entropy loss."""
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        if target == 1:
            return -np.log(pred)
        else:
            return -np.log(1 - pred)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

def run():
    """Main Streamlit app for sentiment analysis."""
    
    st.markdown("""
    <div style="border-left:4px solid #00d4ff; padding-left:16px; margin-bottom:24px;">
    <h2 style="margin-top:0;">🎯 Sentiment Analysis with LSTM</h2>
    <p style="color:#64748b; margin-bottom:0;">Binary classification of text sentiment using LSTM neural network</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Theory ────────────────────────────────────────────────────────────────
    theory_box("LSTM Architecture & Mathematics", """
    **LSTM (Long Short-Term Memory)** solves the vanishing/exploding gradient problem in RNNs using gating mechanisms.
    
    **Forget Gate:** Decides what to discard from cell state
    $$f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)$$
    
    **Input Gate:** Controls what new information to store
    $$i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)$$
    $$\\tilde{C}_t = \\tanh(W_c \\cdot [h_{t-1}, x_t] + b_c)$$
    
    **Cell State Update:** Combines old memory with new information
    $$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$$
    
    **Output Gate:** Controls what to output
    $$o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)$$
    
    **Hidden State:**
    $$h_t = o_t \\odot \\tanh(C_t)$$
    
    **Backpropagation Through Time (BPTT):**
    $$\\frac{\\partial L}{\\partial W} = \\sum_{t=1}^{T} \\frac{\\partial L_t}{\\partial W}$$
    
    **Binary Cross-Entropy Loss:**
    $$\\mathcal{L} = -[y \\log(\\hat{y}) + (1-y) \\log(1-\\hat{y})]$$
    
    where $\\sigma$ = sigmoid, $\\tanh$ = hyperbolic tangent, $\\odot$ = element-wise multiplication
    """)
    
    # ── Sample data ────────────────────────────────────────────────────────────
    st.divider()
    section("📊 Sample Data & Training")
    
    sample_data = {
        'text': [
            'this movie is absolutely wonderful and entertaining',
            'i loved this film incredibly well made',
            'terrible movie waste of time horrible acting',
            'amazing cinematography best film ever',
            'worst movie i have ever seen so bad',
            'brilliant performance outstanding work',
            'i enjoyed watching this great entertainment',
            'awful plot not recommended at all',
        ],
        'label': [1, 1, 0, 1, 0, 1, 1, 0]
    }
    
    df_samples = pd.DataFrame(sample_data)
    st.write("**Training Dataset (Mini):**")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df_samples, use_container_width=True, height=250)
    with col2:
        st.metric("Total Samples", len(df_samples))
        st.metric("Positive", sum(df_samples['label']))
        st.metric("Negative", len(df_samples) - sum(df_samples['label']))
    
    # ── Build vocab ────────────────────────────────────────────────────────────
    texts = sample_data['text']
    vocab, w2i, i2w = build_vocab(texts)
    
    st.info(f"📚 Built vocabulary with **{len(vocab)}** unique words "
            f"(including <PAD> and <UNK>)")
    
    # ── Initialize model ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        embedding_dim = st.slider("Embedding Dimension", 32, 128, 64, step=32)
    with col2:
        hidden_dim = st.slider("Hidden Dimension", 64, 256, 128, step=32)
    with col3:
        learning_rate = st.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.05, 0.1], 0.01)
    
    # ── Train model ────────────────────────────────────────────────────────────
    epochs = st.slider("Training Epochs", 5, 50, 20, step=5)
    
    if st.button("🚀 Train Model", key="train_btn", use_container_width=True):
        model = SentimentLSTM(len(vocab), embedding_dim, hidden_dim, learning_rate)
        
        progress_bar = st.progress(0)
        loss_placeholder = st.empty()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for text, label in zip(texts, sample_data['label']):
                # Encode text
                indices = encode_text(text, w2i, max_len=20)
                
                # Forward
                pred, cache_data = model.forward(indices)
                
                # Compute loss
                loss = model.compute_loss(pred, label)
                epoch_loss += loss
                
                # Backward
                model.backward(label, pred, cache_data)
            
            model.loss_history.append(epoch_loss / len(texts))
            progress_bar.progress((epoch + 1) / epochs)
            
            with loss_placeholder.container():
                st.metric("Epoch Loss", f"{model.loss_history[-1]:.4f}")
        
        progress_bar.empty()
        loss_placeholder.empty()
        st.success(f"✅ Training completed! Final loss: **{model.loss_history[-1]:.4f}**")
        st.session_state.trained_model = model
        st.session_state.vocab_data = (vocab, w2i, i2w)
    
    # ── Training history visualization ─────────────────────────────────────────
    if 'trained_model' in st.session_state and len(st.session_state.trained_model.loss_history) > 0:
        st.divider()
        section("📈 Training Progress")
        
        loss_fig = loss_curve(
            st.session_state.trained_model.loss_history,
            label="Cross-Entropy Loss",
            color="#7c3aed",
            height=300
        )
        st.plotly_chart(loss_fig, use_container_width=True)
    
    # ── Inference ──────────────────────────────────────────────────────────────
    if 'trained_model' in st.session_state:
        st.divider()
        section("🔮 Sentiment Prediction")
        
        user_text = st.text_area(
            "Enter text for sentiment analysis:",
            placeholder="Type something like: 'This movie is absolutely amazing!'",
            height=100
        )
        
        if user_text and st.button("🎯 Predict Sentiment", use_container_width=True):
            model = st.session_state.trained_model
            vocab, w2i, i2w = st.session_state.vocab_data
            
            indices = encode_text(user_text, w2i, max_len=20)
            pred, _ = model.forward(indices)
            
            sentiment = "Positive 😊" if pred > 0.5 else "Negative 😔"
            confidence = max(pred, 1 - pred) * 100
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric("Sentiment", sentiment)
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            with col3:
                st.metric("Score", f"{pred:.4f}")
            
            # Visualize confidence
            fig = go.Figure(go.Bar(
                x=['Negative', 'Positive'],
                y=[1-pred, pred],
                marker_color=['#ef4444', '#10b981'],
                text=[f'{(1-pred)*100:.1f}%', f'{pred*100:.1f}%'],
                textposition='outside'
            ))
            apply_dark(fig, height=300, title="Prediction Confidence")
            st.plotly_chart(fig, use_container_width=True)
    
    # ── Architecture diagram ────────────────────────────────────────────────────
    st.divider()
    theory_box("Model Architecture & Forward Pass", """
    **Overall Architecture:**
    ```
    Input Text → Tokenization → Word Indices → Embedding Layer 
              → LSTM Sequence → Final Hidden State → Dense+Sigmoid → Prediction
    ```
    
    **Embedding Layer:** Maps discrete word indices to continuous vectors
    $$e_i = E[idx_i], \\quad e_i \\in \\mathbb{R}^{d_{embed}}$$
    
    **LSTM Forward Pass:** Processes sequence sequentially
    $$h_t, C_t = \\text{LSTM}(e_t, h_{t-1}, C_{t-1})$$
    
    **Classification Head:** Convert final hidden state to probability
    $$z = W_{out} \\cdot h_T + b_{out}, \\quad z \\in \\mathbb{R}$$
    $$\\hat{y} = \\sigma(z) = \\frac{1}{1 + e^{-z}}$$
    
    **Loss Function (Binary Cross-Entropy):**
    $$\\mathcal{L} = -\\frac{1}{N}\\sum_{i=1}^{N}[y_i \\log(\\hat{y}_i) + (1-y_i) \\log(1-\\hat{y}_i)]$$
    
    **Optimization:** AdaGrad with gradient clipping
    $$W \\leftarrow W - \\frac{\\eta}{\\sqrt{G + \\epsilon}} \\odot \\nabla W$$
    where $G = \\sum_{t=1}^{T} (\\nabla W)^2$
    
    **Hyperparameters:**
    - Embedding dimension: $d_{embed}$ (e.g., 64)
    - Hidden dimension: $d_h$ (e.g., 128)
    - Learning rate: $\\eta$ (e.g., 0.01)
    - Sequence length: $T$ = 20
    """)

