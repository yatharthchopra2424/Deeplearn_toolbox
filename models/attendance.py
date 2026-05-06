"""
models/attendance.py
=====================
Attendance system using OpenCV face detection.
- Upload a class photo
- Each detected face is assigned a sequential Student ID
- Attendance record saved as CSV (timestamped)
- Shows live log table and download link
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import os
from datetime import datetime
import io


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")


def get_cascade_path() -> str:
    p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(p):
        st.error("Haar cascade XML not found.")
        st.stop()
    return p


def detect_and_log(img_bgr: np.ndarray,
                   scale_factor: float,
                   min_neighbors: int,
                   min_size: int,
                   session_name: str):
    """
    Detect faces, assign Student IDs, write to CSV.
    Returns: (annotated_image, df_new_records)
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
    records    = []
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if len(faces):
        for idx, (x, y, w, h) in enumerate(faces):
            student_id = f"S{idx+1:03d}"
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 230, 130), 2)
            cv2.putText(annotated, student_id, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 130), 2)
            records.append({
                "Timestamp":   timestamp,
                "Session":     session_name,
                "Student_ID":  student_id,
                "Status":      "Present",
                "Face_X": x, "Face_Y": y,
                "Face_W": w, "Face_H": h,
            })

    df_new = pd.DataFrame(records)

    # Append to persistent CSV
    if not df_new.empty:
        if os.path.exists(ATTENDANCE_FILE):
            df_existing = pd.read_csv(ATTENDANCE_FILE)
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(ATTENDANCE_FILE, index=False)

    return annotated, df_new


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def run():
    st.info("📖 **Attendance System** — Detect faces, assign Student IDs, and auto-log to CSV")

    # ── Session info ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        session_name = st.text_input("📚 Session / Class Name", 
                                     value="CS101 – Lecture 1", 
                                     key="att_session")
    with col2:
        class_date = st.date_input("📅 Date", value=datetime.now().date())

    # ── Detection parameters ─────────────────────────────────────────────────
    c3, c4, c5 = st.columns(3)
    with c3:
        scale_factor  = st.slider("Scale Factor", 1.05, 2.0, 1.1, step=0.05, key="att_sf")
    with c4:
        min_neighbors = st.slider("Min Neighbors", 1, 10, 5, key="att_mn")
    with c5:
        min_size      = st.slider("Min Face Size (px)", 10, 100, 30, key="att_ms")

    # ── Image source selection ───────────────────────────────────────────────
    source = st.radio("Image source", 
                      ["📷 Live Camera (Real-time)", "📤 Upload Photo"],
                      horizontal=True, key="att_source")

    img_bgr = None
    
    # ── Live Camera Mode ─────────────────────────────────────────────────────
    if source == "📷 Live Camera (Real-time)":
        st.markdown("### 📹 Live Attendance Marking")
        st.write("📸 Capture a photo of the classroom to detect and log students")
        
        picture = st.camera_input("Take attendance photo", key="att_camera")
        
        if picture is not None:
            pil_img = Image.open(picture).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Mark attendance button
            if st.button("✅ Mark Attendance (Camera)", use_container_width=True, key="att_mark_cam"):
                annotated, df_new = detect_and_log(
                    img_bgr, scale_factor, min_neighbors, min_size, 
                    f"{session_name} - {class_date}")

                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Frame**")
                    st.image(picture, use_column_width=True)
                with col2:
                    st.markdown(f"**Detected** — {len(df_new)} student(s)")
                    st.image(bgr_to_pil(annotated), use_column_width=True)

                if df_new.empty:
                    st.warning("⚠️ No faces detected. Try different lighting or adjust parameters.")
                else:
                    st.success(f"✅ **Attendance marked for {len(df_new)} student(s)!**")
                    
                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("👥 Students Present", len(df_new))
                    with m2:
                        st.metric("🕐 Time", datetime.now().strftime("%H:%M:%S"))
                    with m3:
                        st.metric("📅 Session", session_name.split("–")[0].strip())
                    
                    # Display attendance table
                    st.markdown("#### 📋 Attendance Records (This Session)")
                    display_df = df_new[["Student_ID", "Status", "Timestamp"]].copy()
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    # Download button
                    csv_bytes = df_to_csv_bytes(df_new)
                    st.download_button(
                        "⬇️ Download Session Records (CSV)",
                        data=csv_bytes,
                        file_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.info("👆 Click the camera button to capture a photo of your class")

    # ── Upload Image Mode ────────────────────────────────────────────────────
    else:
        st.markdown("### 📤 Upload Class Photo")
        
        uploaded = st.file_uploader("📁 Upload class photo", 
                                    type=["jpg", "jpeg", "png"],
                                    key="att_upload")
        if uploaded is None:
            st.info("👆 Upload a class photo to mark attendance")
            return

        pil_img = Image.open(uploaded).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if st.button("✅ Mark Attendance (Upload)", use_container_width=True, key="att_mark_upload"):
            annotated, df_new = detect_and_log(
                img_bgr, scale_factor, min_neighbors, min_size, 
                f"{session_name} - {class_date}")

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Photo**")
                st.image(pil_img, use_container_width=True)
            with col2:
                st.markdown(f"**Annotated** — {len(df_new)} student(s)")
                st.image(bgr_to_pil(annotated), use_container_width=True)

            if df_new.empty:
                st.warning("⚠️ No faces detected. Adjust parameters and try again.")
            else:
                st.success(f"✅ **Attendance marked for {len(df_new)} student(s)!**")
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("👥 Students Present", len(df_new))
                with m2:
                    st.metric("🕐 Time", datetime.now().strftime("%H:%M:%S"))
                with m3:
                    st.metric("📅 Session", session_name.split("–")[0].strip())
                
                # Display attendance table
                st.markdown("#### 📋 Attendance Records (This Session)")
                display_df = df_new[["Student_ID", "Status", "Timestamp"]].copy()
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Download button
                csv_bytes = df_to_csv_bytes(df_new)
                st.download_button(
                    "⬇️ Download Session Records (CSV)",
                    data=csv_bytes,
                    file_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # ── Cumulative attendance log ────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 Cumulative Attendance Log")

    if os.path.exists(ATTENDANCE_FILE):
        df_all = pd.read_csv(ATTENDANCE_FILE)
        
        # Display metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("📝 Total Records", len(df_all))
        with m2:
            st.metric("👥 Unique Students", df_all["Student_ID"].nunique())
        with m3:
            st.metric("📚 Sessions", df_all["Session"].nunique() if "Session" in df_all.columns else "–")
        with m4:
            st.metric("✅ Present Count", len(df_all[df_all["Status"] == "Present"]))
        
        # Display recent records
        st.markdown("**Recent Records (Last 20):**")
        st.dataframe(df_all.tail(20), use_container_width=True, hide_index=True)

        # Download full log
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download Full Attendance Log",
                data=df_to_csv_bytes(df_all),
                file_name="full_attendance_log.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            if st.button("🗑️ Clear All Records", use_container_width=True):
                os.remove(ATTENDANCE_FILE)
                st.success("✅ All records cleared!")
                st.rerun()
    else:
        st.info("📝 No attendance records yet. Mark attendance to create the log.")

