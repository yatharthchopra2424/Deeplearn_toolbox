"""
Deep Learning Toolbox — Main Application
=========================================
Entry point for the Streamlit dashboard.
Run with: streamlit run app.py
"""

import streamlit as st

# ── Page config must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Deep Learning Toolbox",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Import all model modules ─────────────────────────────────────────────────
from models.perceptron        import run as run_perceptron
from models.mlp               import run as run_mlp
from models.backprop          import run as run_backprop
from models.cnn               import run as run_cnn
from models.rnn               import run as run_rnn
from models.sentiment_analysis import run as run_sentiment_analysis
from models.next_word_predictor import run as run_next_word_predictor
from models.face_detection    import run as run_face_detection
from models.face_counting     import run as run_face_counting
from models.attendance_v2     import run as run_attendance
from models.gradient_viz      import run as run_gradient_viz
from models.loss_explorer     import run as run_loss_explorer
from models.digit_predictor   import run as run_digit_predictor

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

  /* ── CSS Variables ── */
  :root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --card:      #161d2e;
    --border:    #1e2d45;
    --accent:    #00d4ff;
    --accent2:   #7c3aed;
    --accent3:   #10b981;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --danger:    #ef4444;
    --warning:   #f59e0b;
  }

  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
  }

  [data-testid="stHeader"]   { background: transparent !important; }
  [data-testid="stSidebar"]  { background: var(--surface) !important; }

  /* ── Hide default decoration ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar       { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* ── Hero banner ── */
  .hero {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b35 50%, #0a0e1a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 48px 40px 36px;
    margin-bottom: 40px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 50% at 80% 50%, rgba(0,212,255,.08) 0%, transparent 70%),
                radial-gradient(ellipse 40% 60% at 20% 80%, rgba(124,58,237,.06) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--accent);
    border: 1px solid rgba(0,212,255,.3);
    padding: 4px 12px;
    border-radius: 100px;
    margin-bottom: 16px;
    text-transform: uppercase;
  }
  .hero h1 {
    font-size: clamp(2rem, 5vw, 3.4rem);
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0 0 12px;
    background: linear-gradient(90deg, #e2e8f0, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .hero p {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 560px;
    line-height: 1.7;
    margin: 0;
  }
  .hero-stats {
    display: flex;
    gap: 32px;
    margin-top: 28px;
    flex-wrap: wrap;
  }
  .stat {
    text-align: left;
  }
  .stat-num {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
  }
  .stat-label {
    font-size: .75rem;
    color: var(--muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 4px;
  }

  /* ── Section header ── */
  .section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
  }
  .section-header h2 {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
    letter-spacing: .5px;
  }
  .section-line {
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* ── Cards ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px 22px 20px;
    cursor: pointer;
    transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
    height: 100%;
    position: relative;
    overflow: hidden;
  }
  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--card-accent, var(--accent)), transparent);
    opacity: 0;
    transition: opacity .3s;
  }
  .card:hover { transform: translateY(-4px); box-shadow: 0 12px 32px rgba(0,0,0,.4); }
  .card:hover::before { opacity: 1; }
  .card:hover { border-color: var(--card-accent, var(--accent)); }

  .card-icon {
    font-size: 2rem;
    margin-bottom: 12px;
    display: block;
  }
  .card-num {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2px;
    margin-bottom: 6px;
  }
  .card-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 8px;
  }
  .card-desc {
    font-size: .8rem;
    color: var(--muted);
    line-height: 1.6;
    margin-bottom: 16px;
  }
  .card-tag {
    display: inline-block;
    font-size: .7rem;
    padding: 3px 10px;
    border-radius: 100px;
    background: rgba(255,255,255,.05);
    color: var(--muted);
    border: 1px solid var(--border);
  }
  .card-arrow {
    position: absolute;
    bottom: 20px; right: 20px;
    font-size: .8rem;
    color: var(--muted);
    transition: color .2s, transform .2s;
  }
  .card:hover .card-arrow { color: var(--accent); transform: translate(2px, -2px); }

  /* ── Back button ── */
  .back-btn-wrapper { margin-bottom: 28px; }

  /* ── Module header ── */
  .module-header {
    background: linear-gradient(135deg, var(--surface), var(--card));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 28px 24px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
  }
  .module-header::after {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 200px; height: 100%;
    background: radial-gradient(circle at 100% 50%, rgba(0,212,255,.05) 0%, transparent 70%);
  }
  .module-header h2 {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0 0 8px;
    color: var(--text);
  }
  .module-header p {
    color: var(--muted);
    font-size: .9rem;
    margin: 0;
    max-width: 600px;
    line-height: 1.6;
  }

  /* ── Streamlit component overrides ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent2), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 8px 22px !important;
    transition: opacity .2s !important;
  }
  .stButton > button:hover { opacity: .85 !important; }

  [data-testid="stSlider"] label,
  [data-testid="stNumberInput"] label,
  [data-testid="stTextInput"] label,
  [data-testid="stSelectbox"] label,
  [data-testid="stTextArea"] label {
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: .85rem !important;
    font-weight: 600 !important;
  }

  [data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 16px !important;
  }
  [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace !important; }

  .stExpander { border: 1px solid var(--border) !important; border-radius: 10px !important; background: var(--card) !important; }
  .stExpander summary { color: var(--text) !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; }

  div[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px !important; }

  .stSuccess { background: rgba(16,185,129,.1) !important; border: 1px solid rgba(16,185,129,.3) !important; border-radius: 8px !important; }
  .stInfo    { background: rgba(0,212,255,.08) !important; border: 1px solid rgba(0,212,255,.2) !important; border-radius: 8px !important; }
  .stWarning { background: rgba(245,158,11,.08) !important; border: 1px solid rgba(245,158,11,.2) !important; border-radius: 8px !important; }
  .stError   { background: rgba(239,68,68,.08) !important; border: 1px solid rgba(239,68,68,.2) !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Card data ─────────────────────────────────────────────────────────────────
CARDS = [
    {
        "id": "perceptron",
        "icon": "⚡",
        "title": "Perceptron",
        "desc": "Binary classification with manual weight updates and step-by-step calculations.",
        "tag": "Supervised",
        "accent": "#00d4ff",
        "runner": run_perceptron,
    },
    {
        "id": "mlp",
        "icon": "🕸️",
        "title": "Multilayer Neural Network",
        "desc": "Configurable MLP with manual forward propagation and layer-wise output display.",
        "tag": "Deep Learning",
        "accent": "#7c3aed",
        "runner": run_mlp,
    },
    {
        "id": "backprop",
        "icon": "🔄",
        "title": "Backpropagation",
        "desc": "Step-by-step gradient calculation, error flow, and weight update visualization.",
        "tag": "Optimization",
        "accent": "#10b981",
    "runner": run_backprop,
    },
    {
        "id": "cnn",
        "icon": "🖼️",
        "title": "Convolutional Neural Network",
        "desc": "Manual convolution operation with kernel selection and feature map display.",
        "tag": "Computer Vision",
        "accent": "#f59e0b",
        "runner": run_cnn,
    },
    {
        "id": "digit_predictor",
        "icon": "🎨",
        "title": "Handwritten Digit Predictor",
        "desc": "Draw any digit (0-9) on canvas and AI predicts it in real-time using CNN.",
        "tag": "Computer Vision",
        "accent": "#06b6d4",
        "runner": run_digit_predictor,
    },
    {
        "id": "rnn",
        "icon": "🔁",
        "title": "Recurrent Neural Network",
        "desc": "Next-word prediction on text sequences with manual RNN logic.",
        "tag": "Sequence Model",
        "accent": "#ec4899",
        "runner": run_rnn,
    },
    {
        "id": "sentiment_analysis",
        "icon": "💬",
        "title": "Sentiment Analysis",
        "desc": "LSTM-based binary sentiment classification with real-time predictions.",
        "tag": "NLP",
        "accent": "#06b6d4",
        "runner": run_sentiment_analysis,
    },
    {
        "id": "next_word_predictor",
        "icon": "✍️",
        "title": "Next Word Predictor",
        "desc": "Language modeling with N-grams and neural networks for text generation.",
        "tag": "NLP",
        "accent": "#f59e0b",
        "runner": run_next_word_predictor,
    },
    {
        "id": "face_detection",
        "icon": "👁️",
        "title": "Face Detection",
        "desc": "Haar-cascade face detection on uploaded images using OpenCV.",
        "tag": "OpenCV",
        "accent": "#06b6d4",
        "runner": run_face_detection,
    },
    {
        "id": "face_counting",
        "icon": "👥",
        "title": "Face Counting",
        "desc": "Count and annotate every face found in an uploaded photograph.",
        "tag": "OpenCV",
        "accent": "#8b5cf6",
        "runner": run_face_counting,
    },
    {
        "id": "attendance",
        "icon": "📋",
        "title": "Smart Attendance + Face Recognition",
        "desc": "Register student faces with names, then recognize & mark attendance automatically.",
        "tag": "OpenCV + AI",
        "accent": "#14b8a6",
        "runner": run_attendance,
    },
    {
        "id": "gradient_viz",
        "icon": "📉",
        "title": "Gradient Visualization",
        "desc": "Plot how gradients evolve over epochs and visualize convergence.",
        "tag": "Analysis",
        "accent": "#f97316",
        "runner": run_gradient_viz,
    },
    {
        "id": "loss_explorer",
        "icon": "📊",
        "title": "Loss Function Explorer",
        "desc": "Interactively compare MSE and Cross-Entropy loss across training epochs.",
        "tag": "Analysis",
        "accent": "#84cc16",
        "runner": run_loss_explorer,
    },
]

# ── Session state ─────────────────────────────────────────────────────────────
if "active" not in st.session_state:
    st.session_state.active = None


def open_module(card_id: str):
    st.session_state.active = card_id


def go_home():
    st.session_state.active = None


# ── Dashboard ─────────────────────────────────────────────────────────────────
def render_dashboard():
    st.markdown("""
    <div class="hero">
      <div class="hero-tag">🧠 AI / ML Education Platform</div>
      <h1>Deep Learning Toolbox</h1>
      <p>An interactive laboratory for exploring deep learning algorithms — from raw perceptrons to convolutional networks — with every calculation visible and configurable.</p>
      <div class="hero-stats">
        <div class="stat"><div class="stat-num">10</div><div class="stat-label">Models</div></div>
        <div class="stat"><div class="stat-num">100%</div><div class="stat-label">Manual Impl.</div></div>
        <div class="stat"><div class="stat-num">∞</div><div class="stat-label">Experiments</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
      <h2>SELECT A MODULE</h2>
      <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Render 5 × 2 grid of cards ───────────────────────────────────────────
    for row_start in range(0, len(CARDS), 5):
        cols = st.columns(5, gap="medium")
        for i, col in enumerate(cols):
            card_idx = row_start + i
            if card_idx >= len(CARDS):
                break
            card = CARDS[card_idx]
            with col:
                # HTML card (visual only — not clickable via HTML in Streamlit)
                st.markdown(f"""
                <div class="card" style="--card-accent:{card['accent']}">
                  <span class="card-icon">{card['icon']}</span>
                  <div class="card-num">TASK {card_idx+1:02d}</div>
                  <div class="card-title">{card['title']}</div>
                  <div class="card-desc">{card['desc']}</div>
                  <span class="card-tag">{card['tag']}</span>
                  <div class="card-arrow">↗</div>
                </div>
                """, unsafe_allow_html=True)
                # Actual clickable button underneath
                if st.button(f"Open {card['title']}", key=f"btn_{card['id']}",
                             use_container_width=True):
                    open_module(card["id"])
                    st.rerun()


# ── Module view ───────────────────────────────────────────────────────────────
def render_module(card_id: str):
    card = next(c for c in CARDS if c["id"] == card_id)

    # Back button
    if st.button("← Back to Dashboard", key="back"):
        go_home()
        st.rerun()

    # Module header
    st.markdown(f"""
    <div class="module-header">
      <span style="font-size:2rem">{card['icon']}</span>
      <h2>{card['title']}</h2>
      <p>{card['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Run the module
    card["runner"]()


# ── Router ────────────────────────────────────────────────────────────────────
if st.session_state.active is None:
    render_dashboard()
else:
    render_module(st.session_state.active)
