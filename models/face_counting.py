"""
models/face_counting.py
========================
Count faces in an uploaded image using OpenCV Haar cascade.
- Annotates each face with a numbered bounding box
- Provides summary statistics (count, average face size, positions)
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import os
import plotly.graph_objects as go


def get_cascade_path() -> str:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        st.error("Haar cascade XML not found.")
        st.stop()
    return cascade_path


def count_and_annotate(img_bgr: np.ndarray,
                       scale_factor: float,
                       min_neighbors: int,
                       min_size: int):
    """
    Detect, number, and annotate all faces.
    Returns: (annotated_bgr, face_data_list)
    """
    cascade = cv2.CascadeClassifier(get_cascade_path())
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces   = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )

    annotated  = img_bgr.copy()
    face_data  = []
    colors = [
        (0, 255, 120), (0, 180, 255), (255, 100, 0),
        (255, 255, 0), (200, 0, 255), (0, 255, 200),
    ]

    if len(faces):
        for idx, (x, y, w, h) in enumerate(faces):
            color = colors[idx % len(colors)]
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            label = f"#{idx+1}"
            cv2.putText(annotated, label, (x, max(0, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cx, cy = x + w // 2, y + h // 2
            face_data.append({
                "ID": idx + 1,
                "x": x, "y": y,
                "width": w, "height": h,
                "center_x": cx, "center_y": cy,
                "area (px²)": w * h,
            })

    return annotated, face_data


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def plot_face_positions(face_data: list, img_h: int, img_w: int) -> go.Figure:
    """Scatter plot of face centres on a virtual canvas."""
    if not face_data:
        return None
    df = pd.DataFrame(face_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["center_x"], y=img_h - df["center_y"],
        mode="markers+text",
        marker=dict(size=12, color="#00d4ff",
                    line=dict(color="white", width=1)),
        text=[f"#{r}" for r in df["ID"]],
        textposition="top center",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,1)",
        font_family="Space Mono",
        xaxis=dict(range=[0, img_w], title="X pixel"),
        yaxis=dict(range=[0, img_h], title="Y pixel"),
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
        title="Face Centre Map",
    )
    return fig


def run():
    st.info("📖 **Face Counting** — Detect, number, and analyze all faces in an image")

    c1, c2, c3 = st.columns(3)
    with c1:
        scale_factor  = st.slider("Scale Factor", 1.05, 2.0, 1.1, step=0.05, key="fc_sf")
    with c2:
        min_neighbors = st.slider("Min Neighbors", 1, 10, 5, key="fc_mn")
    with c3:
        min_size      = st.slider("Min Face Size (px)", 10, 100, 30, key="fc_ms")

    # ── Image source selection ───────────────────────────────────────────────
    source = st.radio("Image source", 
                      ["📷 Live Camera", "📤 Upload Photo"],
                      horizontal=True, key="fc_source")

    img_bgr = None
    
    # ── Live Camera ──────────────────────────────────────────────────────────
    if source == "📷 Live Camera":
        st.markdown("### 📹 Capture Group Photo")
        
        picture = st.camera_input("Take a group photo", key="fc_camera")
        
        if picture is not None:
            pil_img = Image.open(picture).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            annotated, face_data = count_and_annotate(
                img_bgr, scale_factor, min_neighbors, min_size)

            # ── Side-by-side display ─────────────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Frame**")
                st.image(picture, use_column_width=True)
            with col2:
                st.markdown(f"**Annotated** — {len(face_data)} face(s)")
                st.image(bgr_to_pil(annotated), use_column_width=True)

            # ── Metrics ──────────────────────────────────────────────────────
            if face_data:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("👥 Faces Detected", len(face_data))
                areas = [f["area (px²)"] for f in face_data]
                m2.metric("📊 Avg Face Area", f"{np.mean(areas):.0f} px²")
                m3.metric("📈 Largest Face",  f"{max(areas)} px²")
                m4.metric("📉 Smallest Face", f"{min(areas)} px²")

                # ── Face inventory table ─────────────────────────────────────
                st.markdown("#### 📋 Face Inventory")
                st.dataframe(pd.DataFrame(face_data), use_container_width=True, hide_index=True)

                # ── Face position map ────────────────────────────────────────
                fig = plot_face_positions(face_data, img_bgr.shape[0], img_bgr.shape[1])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                # ── Face size distribution ───────────────────────────────────
                st.markdown("#### 📐 Face Size Distribution")
                fig_size = go.Figure()
                fig_size.add_trace(go.Box(
                    y=areas,
                    name="Face Area",
                    marker_color="#f59e0b",
                    boxmean="sd"
                ))
                fig_size.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(17,24,39,1)",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_size, use_container_width=True)
            else:
                st.warning("⚠️ No faces detected. Try a clearer photo or adjust parameters.")
        else:
            st.info("👆 Click the camera button above to capture a group photo")

    # ── Upload image ─────────────────────────────────────────────────────────
    else:
        uploaded = st.file_uploader("📁 Upload a group photo",
                                     type=["jpg", "jpeg", "png"],
                                     key="fc_upload")
        if uploaded is None:
            st.info("👆 Upload an image to begin face counting")
            return

        pil_img = Image.open(uploaded).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        annotated, face_data = count_and_annotate(
            img_bgr, scale_factor, min_neighbors, min_size)

        # ── Side-by-side display ─────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image(pil_img, use_column_width=True)
        with col2:
            st.markdown(f"**Annotated** — {len(face_data)} face(s)")
            st.image(bgr_to_pil(annotated), use_column_width=True)

        # ── Metrics ──────────────────────────────────────────────────────────
        if face_data:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("👥 Faces Detected", len(face_data))
            areas = [f["area (px²)"] for f in face_data]
            m2.metric("📊 Avg Face Area", f"{np.mean(areas):.0f} px²")
            m3.metric("📈 Largest Face",  f"{max(areas)} px²")
            m4.metric("📉 Smallest Face", f"{min(areas)} px²")

            # ── Face inventory table ─────────────────────────────────────────
            st.markdown("#### 📋 Face Inventory")
            st.dataframe(pd.DataFrame(face_data), use_container_width=True, hide_index=True)

            # ── Face position map ────────────────────────────────────────────
            fig = plot_face_positions(face_data, img_bgr.shape[0], img_bgr.shape[1])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
            # ── Face size distribution ───────────────────────────────────────
            st.markdown("#### 📐 Face Size Distribution")
            fig_size = go.Figure()
            fig_size.add_trace(go.Box(
                y=areas,
                name="Face Area",
                marker_color="#f59e0b",
                boxmean="sd"
            ))
            fig_size.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,1)",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_size, use_container_width=True)
        else:
            st.warning("⚠️ No faces detected — adjust the parameters and try again.")
