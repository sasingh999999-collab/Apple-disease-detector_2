"""
================================================================================
APPLE DISEASE DETECTION SYSTEM - STREAMLIT APPLICATION
================================================================================
Deployment-ready version for Streamlit Cloud / GitHub
Classes: Healthy, Black Rot, Powdery Mildew, Black Pox, Anthracnose, Codling Moth
================================================================================
"""

# ================================================================================
# SECTION 1: IMPORTS
# ================================================================================

import streamlit as st
import os
from ultralytics import YOLO
import collections
from PIL import Image
import cv2
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path

# ================================================================================
# SECTION 2: CONFIGURATION
# ================================================================================

st.set_page_config(
    page_title="Apple Disease Detection",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths for deployment
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'
DEFAULT_MODEL_PATH = MODELS_DIR / 'best.pt'

# Class information
CLASS_INFO = {
    0: {'name': 'Healthy', 'color': '#4CAF50', 'emoji': '?'},
    1: {'name': 'Black Rot', 'color': '#f44336', 'emoji': '??'},
    2: {'name': 'Powdery Mildew', 'color': '#9C27B0', 'emoji': '??'},
    3: {'name': 'Black Pox', 'color': '#FF9800', 'emoji': '??'},
    4: {'name': 'Anthracnose', 'color': '#E91E63', 'emoji': '??'},
    5: {'name': 'Codling Moth', 'color': '#FFC107', 'emoji': '??'}
}

# ================================================================================
# SECTION 3: STYLING
# ================================================================================

st.markdown('''
<style>
.main-header {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.metric-box {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.disease-card {
    background-color: #fff;
    padding: 1.2rem;
    border-radius: 10px;
    margin: 0.8rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    border-left: 5px solid;
}
.info-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2196F3;
    margin: 1rem 0;
}
</style>
''', unsafe_allow_html=True)

# ================================================================================
# SECTION 4: SESSION STATE
# ================================================================================

if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'annotated_path' not in st.session_state:
    st.session_state.annotated_path = None

# ================================================================================
# SECTION 5: HELPER FUNCTIONS
# ================================================================================

def load_model(model_path):
    """Load YOLO model from path."""
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_detection(model, file_path, conf_threshold, iou_threshold):
    """Run YOLO detection and return results."""
    try:
        results = model.predict(
            source=file_path,
            conf=conf_threshold,
            iou=iou_threshold,
            save=True,
            name='streamlit_detection',
            exist_ok=True
        )
        
        if not results or not results[0].boxes:
            return None, "No objects detected."
        
        # Extract detections
        detected_classes = []
        confidences = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            detected_classes.append(model.names[class_id])
            confidences.append(float(box.conf[0]))
        
        # Calculate statistics
        counts = collections.Counter(detected_classes)
        total = sum(counts.values())
        diseased = sum(c for name, c in counts.items() if name.lower() != 'healthy')
        healthy = counts.get('Healthy', 0)
        infection_rate = (diseased / total * 100) if total > 0 else 0
        
        return {
            'total_detections': total,
            'counts': dict(counts),
            'diseased_count': diseased,
            'healthy_count': healthy,
            'infection_rate': infection_rate,
            'avg_confidence': np.mean(confidences),
            'annotated_dir': results[0].save_dir,
            'confidences': confidences,
            'class_names': detected_classes
        }, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def generate_text_report(results, filename, conf, iou):
    """Generate detailed text report."""
    lines = [
        "="*70,
        "APPLE DISEASE DETECTION REPORT",
        "="*70,
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"File: {filename}",
        f"Confidence: {conf} | IOU: {iou}",
        "\n" + "-"*70,
        "SUMMARY",
        "-"*70,
        f"Total Detected: {results['total_detections']}",
        f"Healthy: {results['healthy_count']}",
        f"Diseased: {results['diseased_count']}",
        f"Infection Rate: {results['infection_rate']:.2f}%",
        f"Avg Confidence: {results['avg_confidence']:.3f}",
        "\n" + "-"*70,
        "BREAKDOWN BY CLASS",
        "-"*70,
        f"{'Class':<25} {'Count':<10} {'Percentage'}",
        "-"*70
    ]
    
    for name, count in sorted(results['counts'].items(), key=lambda x: x[1], reverse=True):
        pct = (count / results['total_detections']) * 100
        lines.append(f"{name:<25} {count:<10} {pct:.2f}%")
    
    lines.extend(["\n" + "-"*70, "INDIVIDUAL DETECTIONS", "-"*70])
    for i, (name, conf) in enumerate(zip(results['class_names'], results['confidences']), 1):
        lines.append(f"{i}. {name} (Confidence: {conf:.3f})")
    
    lines.extend(["\n" + "="*70, "END OF REPORT", "="*70])
    return "\n".join(lines)

def get_annotated_file(annotated_dir, original_filename, is_video):
    """Locate annotated output file."""
    try:
        base_name = os.path.splitext(original_filename)[0]
        extensions = ['.avi', '.mp4', '.jpg', '.png'] if is_video else ['.jpg', '.png']
        
        for ext in extensions:
            path = os.path.join(annotated_dir, base_name + ext)
            if os.path.exists(path):
                return path
        
        files = os.listdir(annotated_dir)
        return os.path.join(annotated_dir, files[0]) if files else None
    except:
        return None

# ================================================================================
# SECTION 6: HEADER
# ================================================================================

st.markdown('''
<div class="main-header">
    <h1>?? Apple Disease Detection System</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
        AI-Powered Disease Detection using YOLOv8
    </p>
</div>
''', unsafe_allow_html=True)

# ================================================================================
# SECTION 7: SIDEBAR
# ================================================================================

st.sidebar.header("?? Configuration")

# Model loading
st.sidebar.subheader("1. Load Model")

# Check if model exists
if DEFAULT_MODEL_PATH.exists():
    model_path = st.sidebar.text_input(
        "Model Path",
        value=str(DEFAULT_MODEL_PATH),
        help="Path to YOLOv8 model (.pt file)"
    )
    uploaded_model = None
else:
    st.sidebar.warning("Default model not found")
    model_path = st.sidebar.text_input(
        "Model Path (optional)",
        value="",
        help="Enter path to model or upload below"
    )
    uploaded_model = st.sidebar.file_uploader(
        "Or Upload Model",
        type=['pt'],
        help="Upload your trained YOLOv8 model"
    )

# Load model button
if st.sidebar.button("?? Load Model", use_container_width=True):
    with st.spinner("Loading model..."):
        if uploaded_model:
            temp_path = Path(tempfile.gettempdir()) / 'uploaded_model.pt'
            with open(temp_path, 'wb') as f:
                f.write(uploaded_model.getvalue())
            model = load_model(str(temp_path))
        elif model_path and Path(model_path).exists():
            model = load_model(model_path)
        else:
            st.sidebar.error("No valid model path or upload")
            model = None
        
        if model:
            st.session_state.model = model
            st.sidebar.success("? Model loaded!")

# Model status
if st.session_state.model:
    st.sidebar.success("?? Model Ready")
else:
    st.sidebar.warning("?? No Model Loaded")

st.sidebar.markdown("---")

# Detection parameters
st.sidebar.subheader("2. Detection Parameters")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
iou = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.3, 0.05)

st.sidebar.markdown("---")

# Output options
st.sidebar.subheader("3. Output Options")
output_annotated = st.sidebar.checkbox("?? Annotated Media", True)
output_text = st.sidebar.checkbox("?? Text Report", True)

st.sidebar.markdown("---")

# Class info
st.sidebar.subheader("?? Detection Classes")
for info in CLASS_INFO.values():
    st.sidebar.markdown(
        f"<span style='color: {info['color']};'>{info['emoji']} {info['name']}</span>",
        unsafe_allow_html=True
    )

# ================================================================================
# SECTION 8: UPLOAD INTERFACE
# ================================================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.header("?? Upload & Process")
    
    file_type = st.radio("Select File Type:", ["Image", "Video"], horizontal=True)
    
    if file_type == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'bmp'])
    else:
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file:
        st.subheader("?? Preview")
        is_video = file_type == "Video"
        
        if is_video:
            st.video(uploaded_file)
        else:
            st.image(Image.open(uploaded_file), use_container_width=True)
        
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown(f'''
        <div class="info-box">
            <strong>?? File Info:</strong><br>
            Name: {uploaded_file.name}<br>
            Size: {file_size_mb:.2f} MB
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        process_btn = st.button(
            "?? Analyze File",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state.model is None)
        )
        
        if not st.session_state.model:
            st.warning("?? Load model from sidebar first")
        
        if process_btn:
            with st.spinner("?? Processing..."):
                try:
                    # Save uploaded file
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    # Run detection
                    results, error = process_detection(
                        st.session_state.model, tmp_path, confidence, iou
                    )
                    
                    if error:
                        st.error(f"? {error}")
                        st.session_state.results = None
                    else:
                        st.session_state.results = results
                        st.session_state.annotated_path = get_annotated_file(
                            results['annotated_dir'], uploaded_file.name, is_video
                        )
                        st.success("? Analysis complete! ?")
                    
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"? Error: {str(e)}")
                    st.session_state.results = None
    else:
        st.info("?? Upload a file to begin")

# ================================================================================
# SECTION 9: RESULTS DISPLAY
# ================================================================================

with col2:
    st.header("?? Results")
    
    if st.session_state.results:
        results = st.session_state.results
        
        # Summary metrics
        st.subheader("Summary Statistics")
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.markdown(f'''
            <div class="metric-box">
                <h2 style="color: #2196F3; margin: 0;">{results['total_detections']}</h2>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Total</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with m2:
            st.markdown(f'''
            <div class="metric-box">
                <h2 style="color: #f44336; margin: 0;">{results['diseased_count']}</h2>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Diseased</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with m3:
            rate_color = "#f44336" if results['infection_rate'] > 30 else "#FF9800" if results['infection_rate'] > 10 else "#4CAF50"
            st.markdown(f'''
            <div class="metric-box">
                <h2 style="color: {rate_color}; margin: 0;">{results['infection_rate']:.1f}%</h2>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Infection</p>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Breakdown
        st.subheader("?? Detection Breakdown")
        for name, count in sorted(results['counts'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / results['total_detections']) * 100
            class_id = next(k for k, v in CLASS_INFO.items() if v['name'] == name)
            color = CLASS_INFO[class_id]['color']
            emoji = CLASS_INFO[class_id]['emoji']
            
            st.markdown(f'''
            <div class="disease-card" style="border-left-color: {color};">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <strong>{emoji} {name}</strong><br>
                        <span style="color: #666;">Count: {count} | {pct:.1f}%</span>
                    </div>
                    <div style="font-size: 2rem; color: {color};">{count}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Downloads
        st.subheader("?? Download Results")
        d1, d2 = st.columns(2)
        
        if output_annotated and st.session_state.annotated_path:
            with d1:
                if os.path.exists(st.session_state.annotated_path):
                    with open(st.session_state.annotated_path, 'rb') as f:
                        mime = "video/mp4" if file_type == "Video" else "image/jpeg"
                        st.download_button(
                            f"?? Annotated {file_type}",
                            f.read(),
                            f"annotated_{uploaded_file.name}",
                            mime,
                            use_container_width=True
                        )
        
        if output_text:
            with d2:
                report = generate_text_report(results, uploaded_file.name, confidence, iou)
                st.download_button(
                    "?? Text Report",
                    report,
                    f"report_{os.path.splitext(uploaded_file.name)[0]}.txt",
                    "text/plain",
                    use_container_width=True
                )
        
        # Preview
        if output_annotated and st.session_state.annotated_path:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("??? Annotated Output")
            if os.path.exists(st.session_state.annotated_path):
                if file_type == "Video":
                    st.video(st.session_state.annotated_path)
                else:
                    st.image(Image.open(st.session_state.annotated_path), use_container_width=True)
        
        st.markdown(f'''
        <div class="info-box">
            <strong>?? Config:</strong> Confidence: {confidence} | IOU: {iou} | Avg: {results['avg_confidence']:.3f}
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.info("?? Upload and analyze a file")

# ================================================================================
# SECTION 10: FOOTER
# ================================================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown('''
<div style='text-align: center; color: #666; padding: 1rem 0;'>
    <h4>Apple Disease Detection System</h4>
    <p>Training: 4,170 images | Validation: 524 images | Test: 519 images</p>
    <p>Classes: Healthy, Black Rot, Powdery Mildew, Black Pox, Anthracnose, Codling Moth</p>
    <p style='margin-top: 1rem; font-size: 0.9rem;'>Model: YOLOv8s | Framework: Ultralytics</p>
</div>
''', unsafe_allow_html=True)