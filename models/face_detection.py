"""
models/face_detection.py
=========================
Face detection using OpenCV Haar cascade classifier.
- Accepts uploaded image or uses a default synthetic test image
- Draws bounding boxes around detected faces
- Shows detection parameters (scaleFactor, minNeighbors)
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import os


# ── Helper: download / locate Haar cascade ───────────────────────────────────
def get_cascade_path() -> str:
    """Return path to haarcascade_frontalface_default.xml."""
    # OpenCV installs it alongside the package
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        st.error("Haar cascade XML not found. Ensure opencv-python is installed.")
        st.stop()
    return cascade_path


# ── Detection logic ──────────────────────────────────────────────────────────
def detect_faces(img_bgr: np.ndarray,
                 scale_factor: float,
                 min_neighbors: int,
                 min_size: int):
    """
    Detect faces in a BGR image.
    Returns: (annotated_image, face_rects)
      face_rects: list of (x, y, w, h)
    """
    cascade = cv2.CascadeClassifier(get_cascade_path())
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )

    annotated = img_bgr.copy()
    face_rects = []
    if len(faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 120), 2)
            cv2.putText(annotated, "Face", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)
            face_rects.append((x, y, w, h))

    return annotated, face_rects


# ── Convert BGR→RGB PIL for Streamlit display ────────────────────────────────
def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ── Create a synthetic test image with circles ──────────────────────────────
def make_test_image() -> np.ndarray:
    """Return a simple 300×300 BGR image with circular 'faces'."""
    img = np.ones((300, 400, 3), dtype=np.uint8) * 40
    # Draw two face-like ellipses
    for (cx, cy) in [(120, 150), (280, 150)]:
        cv2.ellipse(img, (cx, cy), (50, 60), 0, 0, 360, (200, 170, 130), -1)
        cv2.circle(img, (cx - 18, cy - 12), 8, (50, 50, 50), -1)
        cv2.circle(img, (cx + 18, cy - 12), 8, (50, 50, 50), -1)
        cv2.ellipse(img, (cx, cy + 15), (22, 10), 0, 0, 180, (150, 80, 80), -1)
    return img


# ── UI ────────────────────────────────────────────────────────────────────────
def run():
    st.info("📖 **Face Detection** — OpenCV's Haar cascade classifier uses a "
            "sliding window to scan the image at multiple scales, looking for "
            "patterns (features) that match a trained face model.")

    c1, c2, c3 = st.columns(3)
    with c1:
        scale_factor  = st.slider("Scale Factor", 1.05, 2.0, 1.1, step=0.05)
    with c2:
        min_neighbors = st.slider("Min Neighbors", 1, 10, 5)
    with c3:
        min_size      = st.slider("Min Face Size (px)", 10, 100, 30)

    source = st.radio("Image source", 
                      ["📷 Live Camera", "📤 Upload Image", "🎨 Synthetic Test"],
                      horizontal=True)

    img_bgr = None

    # ── Live Camera Input ─────────────────────────────────────────────────────
    if source == "📷 Live Camera":
        st.markdown("### 📹 Camera Stream")
        
        picture = st.camera_input("Take a picture")
        
        if picture is not None:
            # Convert uploaded image data to OpenCV format
            pil_img = Image.open(picture).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original Frame**")
                st.image(picture, use_column_width=True)
            
            annotated, faces = detect_faces(img_bgr, scale_factor,
                                           min_neighbors, min_size)
            
            with col_b:
                st.markdown(f"**Detected** — {len(faces)} face(s)")
                st.image(bgr_to_pil(annotated), use_column_width=True)
            
            if faces:
                st.success(f"✅ **{len(faces)} face(s) detected!**")
                import pandas as pd
                df = pd.DataFrame(faces, columns=["x", "y", "width", "height"])
                df["area (px²)"] = df["width"] * df["height"]
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Display face statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Faces", len(faces))
                with col2:
                    avg_size = int(df["area (px²)"].mean())
                    st.metric("Avg Face Size", f"{avg_size} px²")
                with col3:
                    largest = df["area (px²)"].max()
                    st.metric("Largest Face", f"{int(largest)} px²")
            else:
                st.warning("⚠️ No faces detected in camera frame. Try different lighting or angles.")
        else:
            st.info("👆 Click the camera button above to capture a frame from your webcam")

    # ── Upload Image Input ────────────────────────────────────────────────────
    elif source == "📤 Upload Image":
        uploaded = st.file_uploader("📁 Upload a photo", type=["jpg", "jpeg", "png"])
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original**")
                st.image(pil_img, use_column_width=True)
            
            annotated, faces = detect_faces(img_bgr, scale_factor,
                                           min_neighbors, min_size)
            
            with col_b:
                st.markdown(f"**Detected** — {len(faces)} face(s)")
                st.image(bgr_to_pil(annotated), use_column_width=True)
            
            if faces:
                st.success(f"✅ **{len(faces)} face(s) detected!**")
                import pandas as pd
                df = pd.DataFrame(faces, columns=["x", "y", "width", "height"])
                df["area (px²)"] = df["width"] * df["height"]
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Faces", len(faces))
                with col2:
                    avg_size = int(df["area (px²)"].mean())
                    st.metric("Avg Face Size", f"{avg_size} px²")
                with col3:
                    largest = df["area (px²)"].max()
                    st.metric("Largest Face", f"{int(largest)} px²")
            else:
                st.warning("⚠️ No faces detected. Try a clearer image or adjust parameters.")
        else:
            st.info("👆 Upload an image to detect faces")

    # ── Synthetic Test Image ──────────────────────────────────────────────────
    else:
        img_bgr = make_test_image()
        st.info("🎨 Using synthetic test image (circles). Upload a real photo for better results.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original**")
            st.image(bgr_to_pil(img_bgr), use_column_width=True)

        annotated, faces = detect_faces(img_bgr, scale_factor,
                                       min_neighbors, min_size)

        with col_b:
            st.markdown(f"**Detected** — {len(faces)} face(s)")
            st.image(bgr_to_pil(annotated), use_column_width=True)

        if faces:
            st.success(f"✅ {len(faces)} face(s) detected.")
        else:
            st.warning("No faces detected in synthetic image.")

    # ── How Haar Cascades Work ───────────────────────────────────────────────
    st.divider()
    with st.expander("📚 **How Haar Cascades Work** (Mathematics & Algorithm)"):
        st.markdown("""
        **1. Haar Features:**
        
        Simple rectangular brightness differences:
        $$\\text{Haar}(x,y) = \\sum_{\\text{white}} p(x,y) - \\sum_{\\text{black}} p(x,y)$$
        
        Types: edge-like, line-like, center-surround (4 different patterns)
        
        **2. Integral Image (Fast Computation):**
        
        Precomputed cumulative sum:
        $$I(x,y) = \\sum_{i=0}^{x}\\sum_{j=0}^{y} \\text{image}(i,j)$$
        
        Rectangle sum in $O(1)$:
        $$\\text{Sum}(x_1, y_1, x_2, y_2) = I(x_2,y_2) - I(x_1,y_2) - I(x_2,y_1) + I(x_1,y_1)$$
        
        **3. AdaBoost Classifier:**
        
        Trains weak classifiers sequentially:
        $$H(x) = \\text{sign}\\left(\\sum_{t=1}^{T} \\alpha_t h_t(x)\\right)$$
        
        Weights misclassified samples higher in each iteration.
        
        **4. Cascade of Classifiers:**
        
        Multiple stages arranged like:
        ```
        Stage 1 (fast) ──→ Stage 2 (medium) ──→ Stage 3 (slow) ──→ Face/Not Face
        ↓ Reject non-faces quickly
        ```
        
        Early stages reject ~99% of non-face windows. Only promising regions proceed.
        
        **5. Multi-Scale Detection:**
        
        Image pyramid creates multiple scales:
        $$I_k = I_{k-1} \\text{ downsampled by } \\text{scaleFactor}$$
        
        Fixed detector applies to each scale, enabling face detection at any size.
        
        **Parameters:**
        - `scaleFactor`: Image downsampling ratio (1.1 = slow/accurate, 1.5 = fast/rough)
        - `minNeighbors`: Grouping threshold (higher = fewer false positives)
        - `minSize` & `maxSize`: Bounding constraints
        """)
