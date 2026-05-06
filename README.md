# 🧠 Deep Learning Toolbox

An interactive Streamlit dashboard for exploring deep learning models with full manual implementations — every forward pass, backprop step, and weight update is visible.

## 🚀 Quick Start

```bash
# 1. Clone / open the folder
cd deep_learning_toolbox

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
deep_learning_toolbox/
├── app.py                  ← Main Streamlit dashboard
├── requirements.txt
├── README.md
├── data/                   ← Auto-created; stores attendance.csv
├── models/
│   ├── __init__.py
│   ├── perceptron.py       ← Task 1: Manual Perceptron
│   ├── mlp.py              ← Task 2: Multilayer Neural Network
│   ├── backprop.py         ← Task 3: Backpropagation visualizer
│   ├── cnn.py              ← Task 4: Manual CNN / Convolution
│   ├── rnn.py              ← Task 5: Vanilla RNN (next-word prediction)
│   ├── face_detection.py   ← Task 6: OpenCV Face Detection
│   ├── face_counting.py    ← Task 7: Face Counting
│   ├── attendance.py       ← Task 8: Attendance System
│   ├── gradient_viz.py     ← Task 9: Gradient Visualization
│   └── loss_explorer.py    ← Task 10: Loss Function Explorer
└── components/
    └── __init__.py
```

## 🧩 Models Overview

| # | Module | Key Concept |
|---|--------|-------------|
| 1 | Perceptron | Step activation, weight update rule |
| 2 | MLP | Forward prop, Xavier init, configurable depth |
| 3 | Backprop | Chain rule, BPTT, gradient flow |
| 4 | CNN | Manual 2-D convolution, max pooling |
| 5 | RNN | Hidden state, BPTT, next-word prediction |
| 6 | Face Detection | Haar cascade, sliding window |
| 7 | Face Counting | Bounding box annotation, stats |
| 8 | Attendance | CSV logging, download |
| 9 | Gradient Viz | Vanishing/exploding gradients, loss surface |
| 10 | Loss Explorer | MSE, BCE, Hinge, gradient shapes |

## 📦 Dependencies

- `streamlit` — UI framework  
- `numpy` — all manual math  
- `plotly` — interactive charts  
- `opencv-python-headless` — face detection  
- `Pillow` — image I/O  
- `pandas` — data tables  
- `scikit-learn` — optional utilities  

## ➕ Adding a New Model

1. Create `models/my_model.py` with a `run()` function.
2. Import it in `app.py`: `from models.my_model import run as run_my_model`
3. Add a card entry to the `CARDS` list in `app.py`.

That's it!
