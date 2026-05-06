"""
models/next_word_predictor.py
==============================
Next Word Predictor using multiple approaches:
  1. N-gram Language Model (Markov chains)
  2. Neural Network-based predictor (from embeddings)

Demonstrates sequence modeling, probability distributions, and text generation.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from collections import defaultdict
from components.ui import (
    theory_box, formula_box, step_badge, metric_row, PALETTE,
    apply_dark, loss_curve, section
)


# ──────────────────────────────────────────────────────────────────────────────
# Text Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str):
    """Tokenize text into words."""
    text = text.lower()
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    words = text.split()
    return [w for w in words if len(w) > 0]


def build_word_vocab(texts: list):
    """Build vocabulary from text list."""
    word_freq = defaultdict(int)
    for text in texts:
        words = preprocess_text(text)
        for word in words:
            word_freq[word] += 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = ['<UNK>', '<START>', '<END>'] + [w for w, _ in sorted_words]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return vocab, w2i, i2w


# ──────────────────────────────────────────────────────────────────────────────
# N-gram Language Model
# ──────────────────────────────────────────────────────────────────────────────

class NgramLanguageModel:
    """N-gram based language model using Markov chains."""
    
    def __init__(self, n: int = 2):
        self.n = n  # Order of n-gram
        self.ngrams = defaultdict(lambda: defaultdict(int))
        self.unigrams = defaultdict(int)  # For fallback
        self.total_count = 0
        self.training_loss = []
    
    def train(self, texts: list):
        """Train n-gram model on texts."""
        for text in texts:
            words = preprocess_text(text)
            words = ['<START>'] * (self.n - 1) + words + ['<END>']
            
            # Build unigrams
            for word in words:
                self.unigrams[word] += 1
            
            # Build n-grams
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i:i+self.n-1])
                next_word = words[i+self.n-1]
                self.ngrams[context][next_word] += 1
                self.total_count += 1
    
    def predict(self, context: list, top_k: int = 5):
        """
        Predict next word probabilities given context.
        Returns: list of (word, probability) tuples
        """
        if len(context) < self.n - 1:
            context = ['<START>'] * (self.n - 1 - len(context)) + context
        else:
            context = context[-(self.n-1):]
        
        context_tuple = tuple(context)
        
        # Try full context first
        if context_tuple in self.ngrams and len(self.ngrams[context_tuple]) > 0:
            counts = self.ngrams[context_tuple]
            total = sum(counts.values())
            
            probs = [(word, count / total) for word, count in counts.items()]
            probs = sorted(probs, key=lambda x: x[1], reverse=True)[:top_k]
            
            # Normalize
            total_prob = sum(p[1] for p in probs)
            if total_prob > 0:
                probs = [(w, p / total_prob) for w, p in probs]
                return probs
        
        # Fallback to unigrams
        if len(self.unigrams) > 0:
            total = sum(self.unigrams.values())
            probs = [(word, count / total) for word, count in self.unigrams.items()]
            probs = sorted(probs, key=lambda x: x[1], reverse=True)[:top_k]
            
            # Filter out START/END tokens
            probs = [(w, p) for w, p in probs if w not in ['<START>', '<END>']][:top_k]
            
            # Normalize
            total_prob = sum(p[1] for p in probs)
            if total_prob > 0:
                probs = [(w, p / total_prob) for w, p in probs]
                return probs
        
        return [('word', 1.0)]  # Fallback
    
    def generate(self, prompt: list, max_length: int = 15):
        """Generate text from prompt."""
        generated = prompt.copy()
        
        for _ in range(max_length):
            preds = self.predict(generated[-(self.n-1):])
            
            # Filter out START and END tokens for generation
            preds = [(w, p) for w, p in preds if w not in ['<START>', '<END>']]
            
            if not preds:
                break
            
            # Normalize probabilities after filtering
            total_prob = sum(p[1] for p in preds)
            if total_prob == 0:
                break
            preds = [(w, p / total_prob) for w, p in preds]
            
            # Sample based on probabilities
            words, probs = zip(*preds)
            next_word = np.random.choice(words, p=probs)
            generated.append(next_word)
        
        return generated


# ──────────────────────────────────────────────────────────────────────────────
# Neural Network Next Word Predictor
# ──────────────────────────────────────────────────────────────────────────────

class NeuralNextWordPredictor:
    """Neural network based next word predictor."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64,
                 hidden_dim: int = 128, context_size: int = 3, lr: float = 0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        self.lr = lr
        
        # Embeddings for each word
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Layer 1: context embeddings -> hidden
        input_size = context_size * embedding_dim
        self.W1 = np.random.randn(hidden_dim, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_dim, 1))
        
        # Layer 2: hidden -> output (vocab)
        self.W2 = np.random.randn(vocab_size, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((vocab_size, 1))
        
        # Optimizer memory (AdaGrad)
        self.m_W1 = np.ones_like(self.W1)
        self.m_b1 = np.ones_like(self.b1)
        self.m_W2 = np.ones_like(self.W2)
        self.m_b2 = np.ones_like(self.b2)
        
        self.loss_history = []
    
    def forward(self, context_indices):
        """
        Forward pass.
        context_indices: (context_size,) array of word indices
        Returns: logits (vocab_size,) and cache for backward
        """
        # Get embeddings
        context_emb = self.embeddings[context_indices].flatten().reshape(-1, 1)
        
        # Hidden layer with ReLU
        h = np.maximum(0, self.W1 @ context_emb + self.b1)
        
        # Output layer (logits)
        logits = self.W2 @ h + self.b2
        
        # Softmax
        logits = logits.flatten()
        logits = logits - np.max(logits)
        probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-12)
        
        cache = (context_indices, context_emb, h, logits, probs)
        return probs, cache
    
    def backward(self, target_idx, probs, cache):
        """Backward pass."""
        context_indices, context_emb, h, logits, _ = cache
        
        # Cross-entropy gradient
        dlogits = probs.copy()
        dlogits[target_idx] -= 1.0
        
        # Output layer gradients
        dW2 = dlogits.reshape(-1, 1) @ h.T
        db2 = dlogits.reshape(-1, 1)
        
        # Hidden layer gradient
        dh = self.W2.T @ dlogits.reshape(-1, 1)
        
        # ReLU gradient
        dh[h <= 0] = 0
        
        # Layer 1 gradients
        dW1 = dh @ context_emb.T
        db1 = dh
        
        # Update with AdaGrad
        lr = self.lr / np.sqrt(1.0 + len(self.loss_history) / 10)
        
        self.m_W2 += dW2 ** 2
        self.m_b2 += db2 ** 2
        self.W2 -= lr * dW2 / (np.sqrt(self.m_W2) + 1e-8)
        self.b2 -= lr * db2 / (np.sqrt(self.m_b2) + 1e-8)
        
        self.m_W1 += dW1 ** 2
        self.m_b1 += db1 ** 2
        self.W1 -= lr * dW1 / (np.sqrt(self.m_W1) + 1e-8)
        self.b1 -= lr * db1 / (np.sqrt(self.m_b1) + 1e-8)
    
    def compute_loss(self, target_idx, probs):
        """Cross-entropy loss."""
        return -np.log(probs[target_idx] + 1e-12)
    
    def predict(self, context_indices, top_k: int = 5):
        """Predict next word."""
        probs, _ = self.forward(context_indices)
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(i, probs[i]) for i in top_indices]
    
    def generate(self, prompt_indices: list, w2i: dict, i2w: dict,
                 max_length: int = 15):
        """Generate text."""
        context = list(prompt_indices[-(self.context_size-1):])
        generated = prompt_indices.copy()
        
        for _ in range(max_length):
            # Pad context if needed
            while len(context) < self.context_size:
                context = [w2i.get('<START>', 1)] + context
            
            context_arr = np.array(context[-self.context_size:])
            preds = self.predict(context_arr, top_k=10)
            
            if not preds:
                break
            
            # Sample top predictions
            words, probs = zip(*preds)
            probs = np.array(probs)
            probs = probs / probs.sum()  # Normalize
            
            next_idx = np.random.choice(len(words), p=probs)
            next_word_idx = words[next_idx]
            
            if i2w.get(next_word_idx) == '<END>':
                break
            
            generated.append(next_word_idx)
            context.append(next_word_idx)
        
        return generated


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

def run():
    """Main Streamlit app for next word prediction."""
    
    st.markdown("""
    <div style="border-left:4px solid #f59e0b; padding-left:16px; margin-bottom:24px;">
    <h2 style="margin-top:0;">✍️ Next Word Predictor</h2>
    <p style="color:#64748b; margin-bottom:0;">Language modeling with N-grams and Neural Networks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Theory ────────────────────────────────────────────────────────────────
    theory_box("Language Modeling: Theory & Mathematics", """
    **N-gram Model (Markov Chains):**
    
    Markov Assumption for bigrams:
    $$P(w_t | w_1, w_2, ..., w_{t-1}) \\approx P(w_t | w_{t-1})$$
    
    Probability estimation using Maximum Likelihood:
    $$P(w_t | w_{t-1}) = \\frac{\\text{count}(w_{t-1}, w_t)}{\\text{count}(w_{t-1})}$$
    
    For n-grams:
    $$P(w_t | w_{t-n+1}, ..., w_{t-1}) = \\frac{c(w_{t-n+1}, ..., w_t)}{c(w_{t-n+1}, ..., w_{t-1})}$$
    
    **Language Probability (Chain Rule):**
    $$P(w_1, w_2, ..., w_T) = \\prod_{t=1}^{T} P(w_t | w_1, ..., w_{t-1}) \\approx \\prod_{t=1}^{T} P(w_t | w_{t-1})$$
    
    **Neural Network Approach (Feed-Forward):**
    
    Context Embedding:
    $$c = \\text{concat}(e_{w_{t-n+1}}, ..., e_{w_{t-1}}) \\in \\mathbb{R}^{n \\cdot d}$$
    
    Hidden Layer with ReLU:
    $$h = \\text{ReLU}(W_1 \\cdot c + b_1) = \\max(0, W_1 \\cdot c + b_1)$$
    
    Output Probabilities (Softmax):
    $$\\hat{P}(w_t | c) = \\text{softmax}(W_2 \\cdot h + b_2)$$
    $$P(w_i) = \\frac{e^{z_i}}{\\sum_{j=1}^{V} e^{z_j}}$$
    
    **Cross-Entropy Loss:**
    $$\\mathcal{L} = -\\sum_{i=1}^{V} y_i \\log(\\hat{P}_i)$$
    where $y$ is one-hot encoded target word
    """)
    
    # ── Sample data ────────────────────────────────────────────────────────────
    st.divider()
    section("📚 Training Data")
    
    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "to be or not to be that is the question",
        "all that glitters is not gold",
        "the early bird catches the worm",
        "a journey of a thousand miles begins with a single step",
        "what we do today determines who we become tomorrow",
        "success is not final failure is not fatal",
        "the only way to do great work is to love what you do",
    ]
    
    st.write("**Sample Training Corpus:**")
    for i, text in enumerate(sample_texts, 1):
        st.caption(f"{i}. {text}")
    
    # ── Model selection ────────────────────────────────────────────────────────
    st.divider()
    section("🧠 Model Selection & Training")
    
    model_type = st.radio(
        "Select Model Type:",
        ["N-gram Model", "Neural Network Model"],
        horizontal=True
    )
    
    if model_type == "N-gram Model":
        ngram_order = st.slider("N-gram Order", 2, 4, 2)
        
        if st.button("🚀 Train N-gram Model", use_container_width=True, key="train_ngram"):
            model = NgramLanguageModel(n=ngram_order)
            model.train(sample_texts)
            st.session_state.ngram_model = model
            st.session_state.sample_texts = sample_texts
            st.success(f"✅ N-gram model trained! Learned {len(model.ngrams)} n-grams")
    
    else:  # Neural Network Model
        col1, col2, col3 = st.columns(3)
        with col1:
            embedding_dim = st.slider("Embedding Dim", 32, 128, 64, key="emb_nn")
        with col2:
            hidden_dim = st.slider("Hidden Dim", 64, 256, 128, key="hid_nn")
        with col3:
            context_size = st.slider("Context Size", 2, 5, 3, key="ctx_nn")
        
        lr = st.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.05, 0.1], 0.01, key="lr_nn")
        epochs = st.slider("Epochs", 5, 50, 20, key="epochs_nn")
        
        if st.button("🚀 Train Neural Network", use_container_width=True, key="train_nn"):
            # Build vocabulary
            vocab, w2i, i2w = build_word_vocab(sample_texts)
            
            # Initialize model
            model = NeuralNextWordPredictor(len(vocab), embedding_dim, hidden_dim, context_size, lr)
            
            # Training loop
            progress_bar = st.progress(0)
            loss_text = st.empty()
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_samples = 0
                
                for text in sample_texts:
                    words = preprocess_text(text)
                    words = ['<START>'] * (context_size - 1) + words + ['<END>']
                    indices = [w2i.get(w, 0) for w in words]
                    
                    for i in range(len(indices) - context_size):
                        context = np.array(indices[i:i+context_size-1])
                        target_idx = indices[i+context_size-1]
                        
                        probs, cache = model.forward(context)
                        loss = model.compute_loss(target_idx, probs)
                        epoch_loss += loss
                        num_samples += 1
                        
                        model.backward(target_idx, probs, cache)
                
                avg_loss = epoch_loss / max(num_samples, 1)
                model.loss_history.append(avg_loss)
                progress_bar.progress((epoch + 1) / epochs)
                
                with loss_text.container():
                    st.metric("Epoch Loss", f"{avg_loss:.4f}")
            
            progress_bar.empty()
            loss_text.empty()
            
            st.session_state.nn_model = model
            st.session_state.vocab_data = (vocab, w2i, i2w)
            st.session_state.sample_texts = sample_texts
            st.success(f"✅ Neural network trained! Final loss: **{model.loss_history[-1]:.4f}**")
    
    # ── Training loss visualization ────────────────────────────────────────────
    if 'nn_model' in st.session_state and len(st.session_state.nn_model.loss_history) > 0:
        st.divider()
        section("📈 Training Loss")
        
        loss_fig = loss_curve(
            st.session_state.nn_model.loss_history,
            label="Cross-Entropy Loss",
            color="#f59e0b",
            height=300
        )
        st.plotly_chart(loss_fig, use_container_width=True)
    
    # ── Prediction ─────────────────────────────────────────────────────────────
    if 'ngram_model' in st.session_state or 'nn_model' in st.session_state:
        st.divider()
        section("🔮 Next Word Prediction")
        
        prompt = st.text_input(
            "Enter prompt:",
            placeholder="Type something like: 'the quick brown'",
            key="prompt_input"
        )
        
        if prompt:
            try:
                if 'ngram_model' in st.session_state:
                    model = st.session_state.ngram_model
                    words = preprocess_text(prompt)
                    preds = model.predict(words)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write("**Top 5 Next Words:**")
                        pred_df = pd.DataFrame(
                            [(w, f"{p*100:.1f}%") for w, p in preds],
                            columns=["Word", "Probability"]
                        )
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        # Visualization
                        words_list, probs_list = zip(*preds)
                        fig = go.Figure(go.Bar(
                            y=list(words_list),
                            x=probs_list,
                            orientation='h',
                            marker_color='#f59e0b',
                            text=[f'{p*100:.1f}%' for p in probs_list],
                            textposition='outside'
                        ))
                        apply_dark(fig, height=300, title="Next Word Probs")
                        st.plotly_chart(fig, use_container_width=True)
                
                else:  # Neural network
                    model = st.session_state.nn_model
                    vocab, w2i, i2w = st.session_state.vocab_data
                    
                    words = preprocess_text(prompt)
                    
                    # Encode words, use index 0 (UNK) for unknown words
                    encoded = [w2i.get(w, 0) for w in words]
                    
                    # Get the last (context_size-1) words for context
                    if len(encoded) >= model.context_size - 1:
                        context_indices = np.array(encoded[-(model.context_size-1):])
                    else:
                        # Pad with UNK if not enough words
                        context_indices = np.array([0] * (model.context_size - 1 - len(encoded)) + encoded)
                    
                    # Make sure it's exactly the right size
                    context_indices = context_indices[-(model.context_size-1):] if len(context_indices) >= model.context_size-1 else context_indices
                    
                    preds = model.predict(context_indices, top_k=5)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write("**Top 5 Next Words:**")
                        pred_df = pd.DataFrame(
                            [(i2w.get(idx, 'unknown'), f"{p*100:.1f}%") for idx, p in preds],
                            columns=["Word", "Probability"]
                        )
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        # Visualization
                        words_list = [i2w.get(idx, 'unknown') for idx, _ in preds]
                        probs_list = [p for _, p in preds]
                        fig = go.Figure(go.Bar(
                            y=words_list,
                            x=probs_list,
                            orientation='h',
                            marker_color='#f59e0b',
                            text=[f'{p*100:.1f}%' for p in probs_list],
                            textposition='outside'
                        ))
                        apply_dark(fig, height=300, title="Next Word Probs")
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
        
        # ── Text generation ────────────────────────────────────────────────────
        st.divider()
        section("✨ Text Generation")
        
        seed_text = st.text_input(
            "Seed text for generation:",
            placeholder="Start with: 'the quick'",
            key="seed_input"
        )
        
        max_length = st.slider("Max generated words", 5, 30, 10, key="max_len")
        
        if seed_text and st.button("🎨 Generate Text", use_container_width=True):
            try:
                if 'ngram_model' in st.session_state:
                    model = st.session_state.ngram_model
                    words = preprocess_text(seed_text)
                    generated = model.generate(words, max_length=max_length)
                    
                    st.info(f"**Generated:** {' '.join(generated)}")
                
                else:  # Neural network
                    model = st.session_state.nn_model
                    vocab, w2i, i2w = st.session_state.vocab_data
                    
                    words = preprocess_text(seed_text)
                    seed_indices = [w2i.get(w, 0) for w in words]
                    generated = seed_indices.copy()
                    
                    for _ in range(max_length):
                        # Get last (context_size-1) indices
                        if len(generated) >= model.context_size - 1:
                            context = np.array(generated[-(model.context_size-1):])
                        else:
                            context = np.array([0] * (model.context_size - 1 - len(generated)) + generated)
                        
                        # Predict next word
                        preds = model.predict(context, top_k=10)
                        if not preds:
                            break
                        
                        # Sample from top predictions
                        indices, probs = zip(*preds)
                        probs = np.array(probs)
                        probs = probs / probs.sum()
                        
                        next_idx = np.random.choice(list(indices), p=probs)
                        
                        # Stop on END token
                        if i2w.get(next_idx) == '<END>':
                            break
                        
                        generated.append(next_idx)
                    
                    generated_text = ' '.join([i2w.get(idx, 'unknown') for idx in generated])
                    st.info(f"**Generated:** {generated_text}")
            
            except Exception as e:
                st.error(f"❌ Generation error: {str(e)}")
    
    # ── Architecture explanation ───────────────────────────────────────────────
    st.divider()
    theory_box("Architecture & Training Details", """
    **Vocabulary & Tokenization:**
    $$\\text{Vocab} = \\{w_1, w_2, ..., w_V\\}, \\quad |\\text{Vocab}| = V$$
    
    Word Index Mapping:
    $$w \\in \\text{Vocab} \\rightarrow \\text{idx} \\in \\{0, 1, ..., V-1\\}$$
    
    **N-gram Data Preparation:**
    
    For a sequence: $w_1, w_2, w_3, ..., w_T$
    
    Bigram pairs: $(w_1, w_2), (w_2, w_3), ..., (w_{T-1}, w_T)$
    
    Training objective:
    $$\\max_{\\theta} \\sum_{i=1}^{N} \\log P(w_i | w_{i-1}; \\theta)$$
    
    **Neural Network Training:**
    
    Forward Pass:
    $$\\text{input}_{emb} = \\text{Embedding}[w_{t-n+1}, ..., w_{t-1}]$$
    $$h = \\text{ReLU}(W_1 \\cdot \\text{input}_{emb} + b_1)$$
    $$\\text{logits} = W_2 \\cdot h + b_2$$
    
    Loss Computation:
    $$\\mathcal{L} = -\\log(\\text{softmax}(\\text{logits})[w_t])$$
    
    Gradient Updates (AdaGrad):
    $$g_t = g_{t-1} + (\\nabla \\mathcal{L})^2$$
    $$\\theta \\leftarrow \\theta - \\frac{\\eta}{\\sqrt{g_t + \\epsilon}} \\cdot \\nabla \\mathcal{L}$$
    
    **Text Generation (Sampling):**
    
    Given context $c = [w_{t-n+1}, ..., w_{t-1}]$:
    $$w_t \\sim P(w | c) = \\text{softmax}(W_2 \\cdot h + b_2)$$
    
    Using temperature sampling for diversity:
    $$P(w_i | T) = \\frac{e^{z_i/T}}{\\sum_j e^{z_j/T}}$$
    where $T > 1$ increases randomness, $T < 1$ increases confidence
    """)
