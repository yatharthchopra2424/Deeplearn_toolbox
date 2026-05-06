"""
models/attendance_v2.py
========================
Advanced Attendance System with Face Recognition

Two-phase system:
1. REGISTRATION: Admin registers students' faces with their names
2. ATTENDANCE: System recognizes faces and marks attendance automatically

Uses OpenCV face detection + custom embedding for face matching.
Stores face data in pickle database for fast similarity comparison.
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import os
import json
from datetime import datetime
import pickle
from scipy.spatial.distance import euclidean

FACE_RECOGNITION_AVAILABLE = True  # We use OpenCV instead of dlib


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

FACE_DB_FILE = os.path.join(DATA_DIR, "face_database.pkl")
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Face Database Management
# ──────────────────────────────────────────────────────────────────────────────

def load_face_database():
    """Load registered face embeddings from database."""
    if os.path.exists(FACE_DB_FILE):
        with open(FACE_DB_FILE, 'rb') as f:
            return pickle.load(f)
    return {"names": [], "embeddings": []}


def save_face_database(db):
    """Save face encodings to database."""
    with open(FACE_DB_FILE, 'wb') as f:
        pickle.dump(db, f)


def extract_face_embedding(face_roi):
    """
    Extract robust numerical embedding from face ROI using multiple feature types.
    Uses histogram equalization, gradient features, and texture analysis.
    Returns: embedding (256-dim vector) for better discrimination
    """
    # Resize to standard size
    face_resized = cv2.resize(face_roi, (128, 128))
    
    # Convert to grayscale if not already
    if len(face_resized.shape) == 3:
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_resized
    
    # Preprocessing: histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    embeddings = []
    
    # 1. HISTOGRAM FEATURES (4 quadrants + full image)
    h, w = gray.shape
    quadrants = [
        gray[:h//2, :w//2],
        gray[:h//2, w//2:],
        gray[h//2:, :w//2],
        gray[h//2:, w//2:]
    ]
    
    for quad in quadrants + [gray]:
        hist = cv2.calcHist([quad], [0], None, [32], [0, 256])
        embeddings.extend(hist.flatten()[:32])
    
    # 2. EDGE/GRADIENT FEATURES (using Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    
    # Histogram of edge magnitudes
    edge_hist = cv2.calcHist([magnitude], [0], None, [16], [0, 256])
    embeddings.extend(edge_hist.flatten()[:16])
    
    # 3. TEXTURE FEATURES (using Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).astype(np.uint8)
    laplacian_hist = cv2.calcHist([laplacian], [0], None, [16], [0, 256])
    embeddings.extend(laplacian_hist.flatten()[:16])
    
    # 4. GRID-BASED LOCAL HISTOGRAMS (4x4 grid)
    cell_h, cell_w = h // 4, w // 4
    for i in range(4):
        for j in range(4):
            cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_hist = cv2.calcHist([cell], [0], None, [16], [0, 256])
            embeddings.extend(cell_hist.flatten()[:4])
    
    # 5. STATISTICAL FEATURES
    embeddings.extend([
        np.mean(gray),
        np.std(gray),
        np.min(gray),
        np.max(gray),
        np.median(gray)
    ])
    
    # Normalize to fixed dimensions
    embedding = np.array(embeddings)
    embedding = embedding[:256]  # Keep first 256 dimensions
    if len(embedding) < 256:
        embedding = np.pad(embedding, (0, 256 - len(embedding)), 'constant')
    
    # L2 normalization
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


def find_matching_student(test_embedding, db, tolerance=0.35):
    """
    Find matching student by comparing face embedding using euclidean distance.
    Returns: (student_name, distance, all_scores) or (None, None, all_scores)
    
    tolerance: Maximum euclidean distance for match (lower = stricter)
               Typical range: 0.25 (very strict) to 0.5 (loose)
    """
    if not db["embeddings"] or test_embedding is None:
        return None, None, {}
    
    # Calculate distances to all stored embeddings
    all_scores = {}
    distances = []
    
    for name, embedding in zip(db["names"], db["embeddings"]):
        dist = euclidean(test_embedding, np.array(embedding))
        distances.append(dist)
        all_scores[name] = dist
    
    # Find best match
    best_match_idx = np.argmin(distances)
    best_distance = distances[best_match_idx]
    
    # Check if within tolerance
    if best_distance < tolerance:
        return db["names"][best_match_idx], best_distance, all_scores
    
    return None, best_distance, all_scores


def get_cascade_path() -> str:
    """Get Haar cascade path for face detection backup."""
    p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(p):
        st.error("Haar cascade XML not found.")
        st.stop()
    return p


def detect_faces_haar(img_bgr, scale_factor=1.1, min_neighbors=5, min_size=30):
    """Fallback face detection using Haar cascades."""
    cascade = cv2.CascadeClassifier(get_cascade_path())
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )
    return faces


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert BGR image to PIL format."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

def run():
    st.markdown("""
    <div style="border-left:4px solid #06b6d4; padding-left:16px; margin-bottom:24px;">
    <h2 style="margin-top:0;">🎓 Smart Attendance System</h2>
    <p style="color:#64748b; margin-bottom:0;">AI-powered face recognition for accurate attendance marking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Tabs for Registration & Attendance ────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["👤 Register Students", "📋 Mark Attendance", "📊 View Records"])    
    # Check database compatibility
    db = load_face_database()
    if db["embeddings"] and len(db["embeddings"][0]) != 256:
        st.warning("⚠️ Database format changed (embeddings updated). Please re-register students for better accuracy.")
        if st.button("🗑️ Clear Old Database", use_container_width=True):
            if os.path.exists(FACE_DB_FILE):
                os.remove(FACE_DB_FILE)
            st.success("Database cleared! You can now register students with the new system.")
            st.rerun()    
    # ── TAB 1: REGISTRATION ────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Step 1: Register Student Faces")
        st.info("📸 Capture or upload a clear photo of each student's face with their name")
        
        col1, col2 = st.columns(2)
        with col1:
            student_name = st.text_input(
                "Student Name",
                placeholder="Enter full name (e.g., John Doe)",
                key="reg_name"
            )
        with col2:
            st.write("")
            st.write("")
            if st.button("🗑️ Clear Database", use_container_width=True):
                if os.path.exists(FACE_DB_FILE):
                    os.remove(FACE_DB_FILE)
                st.success("✅ Database cleared!")
                st.rerun()
        
        # Registration source
        reg_source = st.radio("Capture Method", 
                            ["📷 Live Camera", "📤 Upload Photo"],
                            horizontal=True, key="reg_source")
        
        img_bgr = None
        
        if reg_source == "📷 Live Camera":
            picture = st.camera_input("Capture face", key="reg_camera")
            if picture is not None:
                pil_img = Image.open(picture).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            uploaded = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"], key="reg_upload")
            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        if img_bgr is not None and student_name:
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(bgr_to_pil(img_bgr), caption="Captured Photo", use_column_width=True)
            
            with col_b:
                st.write("")
                if st.button("✅ Register This Face", use_container_width=True, key="reg_btn"):
                    # Detect face in image
                    cascade = cv2.CascadeClassifier(get_cascade_path())
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                    
                    if len(faces) == 0:
                        st.error("❌ No face detected! Please ensure the face is clear and visible.")
                    else:
                        # Extract face region and compute embedding
                        x, y, w, h = faces[0]
                        face_roi = img_bgr[y:y+h, x:x+w]
                        embedding = extract_face_embedding(face_roi)
                        
                        # Load database and add new student
                        db = load_face_database()
                        
                        # Check if student already exists
                        if student_name in db["names"]:
                            st.warning(f"⚠️ {student_name} already registered. Updating...")
                            idx = db["names"].index(student_name)
                            db["embeddings"][idx] = embedding
                        else:
                            db["names"].append(student_name)
                            db["embeddings"].append(embedding)
                        
                        save_face_database(db)
                        st.success(f"✅ **{student_name}** registered successfully!")
                        st.balloons()
        
        # Display registered students
        st.divider()
        st.markdown("### Registered Students")
        db = load_face_database()
        
        if db["names"]:
            reg_df = pd.DataFrame({
                "Student Name": db["names"],
                "Registration Status": ["✅ Registered"] * len(db["names"])
            })
            st.dataframe(reg_df, use_container_width=True, hide_index=True)
            st.metric("Total Registered", len(db["names"]))
        else:
            st.info("No students registered yet.")
    
    # ── TAB 2: MARK ATTENDANCE ────────────────────────────────────────────────
    with tab2:
        st.markdown("### Step 2: Mark Attendance")
        st.info("📸 Capture class photo or individual photos. Faces will be recognized and matched to registered students")
        
        col1, col2 = st.columns(2)
        with col1:
            session_name = st.text_input(
                "Session / Class Name",
                value="CS101 - Lecture",
                key="attend_session"
            )
        with col2:
            class_date = st.date_input("Date", value=datetime.now().date())
        
        # Matching tolerance (euclidean distance threshold)
        tolerance = st.slider("Recognition Strictness", 0.20, 0.50, 0.35, 0.02,
            help="Lower = stricter matching (0.20=very strict, 0.50=loose)")
        
        attend_source = st.radio("Capture Method",
                                ["📷 Live Camera", "📤 Upload Photo"],
                                horizontal=True, key="attend_source")
        
        img_bgr = None
        
        if attend_source == "📷 Live Camera":
            picture = st.camera_input("Capture attendance photo", key="attend_camera")
            if picture is not None:
                pil_img = Image.open(picture).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            uploaded = st.file_uploader("Upload class photo", type=["jpg", "jpeg", "png"], key="attend_upload")
            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        if img_bgr is not None:
            if st.button("🔍 Recognize & Mark Attendance", use_container_width=True, key="attend_mark"):
                db = load_face_database()
                
                if not db["names"]:
                    st.error("❌ No registered students! Please register students first in the 'Register Students' tab.")
                else:
                    # Detect all faces in image using Haar cascades
                    cascade = cv2.CascadeClassifier(get_cascade_path())
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                    
                    if len(faces) == 0:
                        st.warning("⚠️ No faces detected in the image.")
                    else:
                        # Recognize each face
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        attendance_records = []
                        annotated = img_bgr.copy()
                        recognized_faces = {}
                        
                        for x, y, w, h in faces:
                            # Extract face region
                            face_roi = img_bgr[y:y+h, x:x+w]
                            
                            # Compute embedding
                            test_embedding = extract_face_embedding(face_roi)
                            
                            # Find matching student
                            student_name, distance, all_scores = find_matching_student(test_embedding, db, tolerance)
                            
                            if student_name:
                                # Face recognized
                                color = (0, 255, 0)  # Green
                                label = f"✓ {student_name}"
                                recognized_faces[student_name] = True
                                
                                confidence = max(0, min(100, (1 - distance / 0.5) * 100))  # Convert distance to confidence
                                attendance_records.append({
                                    "Timestamp": timestamp,
                                    "Session": session_name,
                                    "Date": str(class_date),
                                    "Student_Name": student_name,
                                    "Status": "Present",
                                    "Confidence": f"{confidence:.1f}%",
                                    "Match_Distance": f"{distance:.3f}"
                                })
                            else:
                                # Face not recognized - show best match for debugging
                                color = (0, 0, 255)  # Red
                                if all_scores:
                                    best_match = min(all_scores.items(), key=lambda x: x[1])
                                    label = f"? {best_match[0]} ({best_match[1]:.2f})"
                                else:
                                    label = "? Unknown"
                            
                            # Draw bounding box
                            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(annotated, label, (x, y - 8),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Display results
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.image(bgr_to_pil(img_bgr), caption="Original", use_column_width=True)
                        with col_b:
                            st.image(bgr_to_pil(annotated), caption="Recognized", use_column_width=True)
                        
                        # Attendance summary
                        if attendance_records:
                            st.success(f"✅ **{len(attendance_records)} student(s) recognized!**")
                            
                            # Display attendance table
                            attend_df = pd.DataFrame(attendance_records)
                            st.dataframe(attend_df, use_container_width=True, hide_index=True)
                            
                            # Save to CSV
                            if os.path.exists(ATTENDANCE_FILE):
                                df_existing = pd.read_csv(ATTENDANCE_FILE)
                                df_all = pd.concat([df_existing, attend_df], ignore_index=True)
                            else:
                                df_all = attend_df
                            df_all.to_csv(ATTENDANCE_FILE, index=False)
                            
                            # Download button
                            csv_bytes = attend_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "⬇️ Download Session Records",
                                data=csv_bytes,
                                file_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # Metrics
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric("👥 Present", len(attendance_records))
                            with m2:
                                st.metric("❌ Absent", len(db["names"]) - len(attendance_records))
                            with m3:
                                st.metric("📊 Attendance %", 
                                         f"{(len(attendance_records)/len(db['names'])*100):.1f}%")
                        else:
                            st.warning("⚠️ No recognized faces. Try a clearer photo or adjust tolerance.")
                        
                        # Show matching debug scores for all students
                        st.divider()
                        st.markdown("#### 🔍 Matching Scores (Debug Info)")
                        st.info("These are the face distance scores for each detected face. Lower = better match. Adjust tolerance slider if needed.")
                        
                        # Collect all matching scores across all detected faces
                        all_matches = {}
                        for x, y, w, h in faces:
                            face_roi = img_bgr[y:y+h, x:x+w]
                            test_embedding = extract_face_embedding(face_roi)
                            _, _, scores = find_matching_student(test_embedding, db, 1.0)  # Use high tolerance to get all scores
                            
                            for name, score in scores.items():
                                if name not in all_matches:
                                    all_matches[name] = []
                                all_matches[name].append(score)
                        
                        if all_matches:
                            debug_data = []
                            for name in db["names"]:
                                if name in all_matches:
                                    best_score = min(all_matches[name])
                                    face_count = len(all_matches[name])
                                    debug_data.append({
                                        "Student": name,
                                        "Best Match Score": f"{best_score:.3f}",
                                        "Threshold": f"{tolerance:.3f}",
                                        "Status": "✅ Match" if best_score < tolerance else "❌ No Match",
                                        "Faces Detected": face_count
                                    })
                            
                            debug_df = pd.DataFrame(debug_data)
                            st.dataframe(debug_df, use_container_width=True, hide_index=True)
    
    # ── TAB 3: VIEW RECORDS ────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Attendance Records")
        
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_csv(ATTENDANCE_FILE)
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("📝 Total Records", len(df))
            with m2:
                st.metric("👥 Unique Students", df["Student_Name"].nunique())
            with m3:
                st.metric("📚 Sessions", df["Session"].nunique())
            with m4:
                st.metric("✅ Present Count", len(df[df["Status"] == "Present"]))
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                selected_student = st.selectbox(
                    "Filter by Student",
                    ["All Students"] + sorted(df["Student_Name"].unique().tolist())
                )
            with col2:
                selected_session = st.selectbox(
                    "Filter by Session",
                    ["All Sessions"] + sorted(df["Session"].unique().tolist())
                )
            
            # Apply filters
            filtered_df = df
            if selected_student != "All Students":
                filtered_df = filtered_df[filtered_df["Student_Name"] == selected_student]
            if selected_session != "All Sessions":
                filtered_df = filtered_df[filtered_df["Session"] == selected_session]
            
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            # Download full log
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Full Attendance Log",
                data=csv_bytes,
                file_name="attendance_log.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No attendance records yet. Start marking attendance!")
