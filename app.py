import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import queue
import threading
import torch
torch.set_num_threads(1)

from config import (
    DEFAULT_WEIGHTS,
    DEFAULT_CONF,
    DEFAULT_STRIDE,
    DEFAULT_LINE_THICKNESS,
    DEFAULT_FONT_SCALE,
    MIN_PERSISTENCE_FRAMES,
    DEFAULT_VIDEO_IMGSZ
)
from core.detector import RTDETREngine
from core.video_engine import VideoInspectionPipeline
from core.exporter import (
    incidents_to_dataframe,
    stills_detections_to_dataframe,
    dataframe_to_csv_bytes
)
from utils.temp_manager import temporary_video_file

# Page configuration
st.set_page_config(
    page_title="RailSamiksha AI | AeroInspect RT-DETR",
    layout="wide",
    page_icon="🧭",
    initial_sidebar_state="expanded"
)

# Custom Styling (Inspired by RailSamiksha Aerospace Design System)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: #070b12 !important;
        color: #f1f5f9 !important;
    }
    
    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Navbar Header */
    .rs-navbar {
        background: linear-gradient(180deg, #0d1527 0%, #090e1a 100%);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        display: flex;
        flex-direction: column;
        gap: 16px;
    }

    @media (min-width: 768px) {
        .rs-navbar {
            flex-direction: row;
            align-items: center;
            justify-content: space-between;
        }
    }

    .rs-brand-block {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .rs-logo-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        flex-shrink: 0;
    }

    .rs-brand-title {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 8px;
        line-height: 1.2;
    }

    .rs-brand-ai {
        color: #60a5fa;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
        background: rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .rs-sub-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #94a3b8;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid #334155;
        padding: 3px 8px;
        border-radius: 9999px;
        text-transform: uppercase;
    }

    .rs-brand-tagline {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 3px;
        font-weight: 400;
    }

    /* Telemetry Chips */
    .rs-telemetry-strip {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }

    .rs-status-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        padding: 6px 12px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #0f172a;
        border: 1px solid #1e293b;
        color: #cbd5e1;
    }

    .chip-live {
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.25);
        color: #34d399;
    }

    .chip-gpu {
        background: rgba(59, 130, 246, 0.12);
        border-color: rgba(59, 130, 246, 0.3);
        color: #60a5fa;
    }

    .chip-cpu {
        background: rgba(245, 158, 11, 0.12);
        border-color: rgba(245, 158, 11, 0.3);
        color: #fbbf24;
    }

    .rs-pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Metric Cards */
    .rs-metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-bottom: 22px;
    }

    .rs-card {
        background: #0c121e;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .rs-card:hover {
        border-color: #334155;
        transform: translateY(-1px);
    }

    .rs-card-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }

    .bar-blue { background: linear-gradient(90deg, #2563eb, #38bdf8); }
    .bar-amber { background: linear-gradient(90deg, #ea580c, #f59e0b); }
    .bar-emerald { background: linear-gradient(90deg, #059669, #10b981); }
    .bar-cyan { background: linear-gradient(90deg, #0891b2, #06b6d4); }

    .rs-card-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .rs-card-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.1;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #080d17 !important;
        border-right: 1px solid #1e293b !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #1e293b !important;
    }

    .sidebar-workspace-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 20px;
    }

    .ws-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }

    .ws-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #94a3b8;
        text-transform: uppercase;
    }

    .ws-role-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 9999px;
        background: rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #60a5fa;
    }

    .ws-sub {
        font-size: 0.78rem;
        color: #cbd5e1;
        font-weight: 500;
    }

    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin-top: 18px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid #1e293b !important;
        background-color: transparent !important;
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #0c121e !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px 8px 0 0 !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        color: #60a5fa !important;
        border-color: #3b82f6 !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    /* Primary Action Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.35) !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 6px 18px 0 rgba(37, 99, 235, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Severity Badges for Taxonomy */
    .sev-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        margin-bottom: 5px;
        width: 100%;
        background: #0f172a;
        border: 1px solid #1e293b;
    }

    .sev-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing RT-DETR Vision Engine...")
def load_detector(weights_path: str = DEFAULT_WEIGHTS) -> RTDETREngine:
    """Load and cache the RT-DETR inference engine across application sessions."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}. "
            "If deployed on Hugging Face Spaces or GitHub, please ensure 'weights/best.pt' is included or 'git lfs pull' was executed."
        )
    if os.path.getsize(weights_path) < 1024 * 1024:
        raise ValueError(
            f"Model weights at '{weights_path}' is only {os.path.getsize(weights_path)} bytes (Git LFS pointer). "
            "Please run 'git lfs pull' to download the ~63MB binary model weights."
        )
    return RTDETREngine(weights_path=weights_path, default_conf=DEFAULT_CONF)


def main():
    # Load Model
    try:
        detector = load_detector(DEFAULT_WEIGHTS)
    except Exception as e:
        st.error(f"❌ Failed to load RT-DETR model from '{DEFAULT_WEIGHTS}': {e}")
        st.stop()

    device_is_cuda = (detector.device == "cuda")
    device_chip_class = "chip-gpu" if device_is_cuda else "chip-cpu"
    device_label = "GPU (CUDA)" if device_is_cuda else "CPU (ACCELERATED)"

    # RailSamiksha Brand Navbar Header
    st.markdown(f"""
    <div class="rs-navbar">
        <div class="rs-brand-block">
            <div class="rs-logo-box">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"></polygon>
                </svg>
            </div>
            <div>
                <div class="rs-brand-title">
                    RailSamiksha <span class="rs-brand-ai">AI</span>
                    <span class="rs-sub-pill">AEROINSPECT PLATFORM</span>
                </div>
                <div class="rs-brand-tagline">
                    Autonomous Aerial Drone Surveillance & Railway Defect Identification Pipeline
                </div>
            </div>
        </div>
        <div class="rs-telemetry-strip">
            <div class="rs-status-chip chip-live">
                <span class="rs-pulse-dot"></span>
                <span>SYSTEM ONLINE</span>
            </div>
            <div class="rs-status-chip chip-model">
                <span>RT-DETR-L DEEP VISION</span>
            </div>
            <div class="rs-status-chip {device_chip_class}">
                <span>⚡ {device_label}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls (RailSamiksha Telemetry Panel)
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-workspace-card">
            <div class="ws-header">
                <span class="ws-title">WORKSPACE</span>
                <span class="ws-role-badge">OPERATOR</span>
            </div>
            <div class="ws-sub">🛰️ AeroDrone Inspection Terminal</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">⚙️ INFERENCE CONTROLS</div>', unsafe_allow_html=True)
        conf_slider = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_CONF,
            step=0.05,
            help="Minimum confidence score required for an anomaly detection."
        )

        stride_slider = st.slider(
            "Video Frame Stride",
            min_value=1,
            max_value=6,
            value=DEFAULT_STRIDE,
            step=1,
            help="Process every Nth video frame to balance real-time FPS with detection density."
        )


        persistence_slider = st.slider(
            "Temporal Verification Filter",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help="Frames required to mark a defect as 'Verified'. Set to 1 for instant registration of every defect."
        )


        st.markdown('<div class="section-title">🎨 ANNOTATION DISPLAY</div>', unsafe_allow_html=True)
        box_thickness = st.slider(
            "Bounding Box Border (px)",
            min_value=1,
            max_value=10,
            value=DEFAULT_LINE_THICKNESS,
            step=1,
            help="Thickness of the defect bounding box borders in pixels."
        )

        font_scale_val = st.slider(
            "Defect Label Size",
            min_value=0.4,
            max_value=2.0,
            value=DEFAULT_FONT_SCALE,
            step=0.05,
            help="Size and prominence of the defect name and confidence badge."
        )

        auto_scale_box = st.checkbox(
            "Auto-Scale to Image Resolution",
            value=True,
            help="Dynamically scales box borders and label size for high-resolution 2K/4K drone imagery."
        )

        st.markdown("---")
        st.markdown('<div class="section-title">📋 DEFECT TAXONOMY & SEVERITY</div>', unsafe_allow_html=True)
        
        # RailSamiksha Severity Palette
        severity_map = {
            6: ("Rail Crack", "#ef4444", "CRITICAL"),
            2: ("Broken Sleeper", "#f43f5e", "CRITICAL"),
            4: ("Missing Sleeper", "#ec4899", "CRITICAL"),
            7: ("Rail Joint Damage", "#f97316", "HIGH"),
            1: ("Ballast Washout", "#fb923c", "HIGH"),
            5: ("Missing Fastener", "#f59e0b", "MEDIUM"),
            3: ("Damaged Fastener", "#eab308", "MEDIUM"),
            0: ("Ballast Deficiency", "#06b6d4", "LOW"),
            8: ("Rail Misalignment", "#3b82f6", "LOW"),
        }

        for cid in sorted(detector.names.keys()):
            cname, color_hex, sev_tier = severity_map.get(cid, (detector.names[cid], "#64748b", "INFO"))
            st.markdown(f"""
            <div class="sev-pill">
                <span class="sev-dot" style="background-color: {color_hex};"></span>
                <span style="flex: 1; color: #e2e8f0; font-size: 0.72rem;">{cname}</span>
                <span style="color: {color_hex}; font-size: 0.65rem; font-weight: 700;">{sev_tier}</span>
            </div>
            """, unsafe_allow_html=True)

        st.caption(f"🛡️ Persistence Filter: Min {MIN_PERSISTENCE_FRAMES} frames.")

    # Application Tabs
    tab1, tab2 = st.tabs(["📸 Aerial Still Inspection", "🎥 Flight Video Stream"])

    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    sample_img_path = os.path.join(samples_dir, "sample_track_image.jpg")
    sample_vid_path = os.path.join(samples_dir, "sample_track_flight.mp4")

    # --------------------------------------------------------------------------
    # TAB 1: Aerial Still Inspection
    # --------------------------------------------------------------------------
    with tab1:
        st.markdown("""
        <div style="margin-bottom: 16px;">
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #f8fafc;">
                High-Resolution Aerial Still Inspection
            </h3>
            <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #94a3b8;">
                Upload drone orthophotos or high-altitude track captures for multi-scale defect identification.
            </p>
        </div>
        """, unsafe_allow_html=True)

        img_mode = st.radio(
            "Select Imagery Input Mode:",
            ["Upload Drone Image", "⚡ Try Built-in Sample Image"],
            horizontal=True,
            key="img_source_mode"
        )

        pil_img = None
        img_name = ""

        if img_mode == "Upload Drone Image":
            uploaded_image = st.file_uploader(
                "Select Drone Image File",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                key="image_uploader"
            )
            if uploaded_image is not None:
                pil_img = Image.open(uploaded_image).convert("RGB")
                img_name = uploaded_image.name
        else:
            if os.path.exists(sample_img_path):
                pil_img = Image.open(sample_img_path).convert("RGB")
                img_name = "sample_track_image.jpg"
                st.caption("📸 Loaded sample aerial track frame with surface anomalies.")
            else:
                st.warning("Sample image not found in 'samples/sample_track_image.jpg'. Please upload an image.")

        if pil_img is not None:
            img_rgb = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            h, w = img_rgb.shape[:2]

            with st.spinner("Analyzing imagery via RT-DETR..."):
                annotated_img, detections = detector.infer_direct(
                    frame_bgr=img_bgr,
                    imgsz=640,
                    conf=conf_slider,
                    line_thickness=box_thickness,
                    font_scale=font_scale_val,
                    auto_scale=auto_scale_box
                )

            # Metric Cards
            unique_classes = len(set(d["class_name"] for d in detections))
            max_conf = max([d["confidence"] for d in detections], default=0.0)

            st.markdown(f"""
            <div class="rs-metric-grid">
                <div class="rs-card">
                    <div class="rs-card-bar bar-blue"></div>
                    <div class="rs-card-label">Defects Identified</div>
                    <div class="rs-card-value">{len(detections)}</div>
                </div>
                <div class="rs-card">
                    <div class="rs-card-bar bar-amber"></div>
                    <div class="rs-card-label">Defect Categories</div>
                    <div class="rs-card-value">{unique_classes}</div>
                </div>
                <div class="rs-card">
                    <div class="rs-card-bar bar-emerald"></div>
                    <div class="rs-card-label">Peak Confidence</div>
                    <div class="rs-card-value">{max_conf * 100:.1f}%</div>
                </div>
                <div class="rs-card">
                    <div class="rs-card-bar bar-cyan"></div>
                    <div class="rs-card-label">Imagery Resolution</div>
                    <div class="rs-card-value">{w} &times; {h}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Side-by-Side Comparison
            col_orig, col_annot = st.columns(2)
            with col_orig:
                st.markdown("**Original Drone Imagery**")
                st.image(img_rgb, use_container_width=True)

            with col_annot:
                st.markdown("**RT-DETR Annotated Output**")
                st.image(annotated_img, use_container_width=True)

            # Findings Table
            st.markdown("### 🔍 Detected Anomaly Registry")
            df_stills = stills_detections_to_dataframe(detections)
            if not df_stills.empty:
                st.dataframe(df_stills, use_container_width=True)
                csv_bytes = dataframe_to_csv_bytes(df_stills)
                st.download_button(
                    label="📥 Export Findings Report (CSV)",
                    data=csv_bytes,
                    file_name=f"rail_still_inspection_{os.path.splitext(img_name)[0]}.csv",
                    mime="text/csv"
                )
            else:
                st.success("✅ Clean Inspection: Zero rail anomalies detected above the selected confidence threshold.")

    # --------------------------------------------------------------------------
    # TAB 2: Flight Video Stream
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown("""
        <div style="margin-bottom: 16px;">
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #f8fafc;">
                Real-Time Drone Flight Video Pipeline
            </h3>
            <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #94a3b8;">
                Continuous video stream analysis with persistent spatial tracking and temporal verification.
            </p>
        </div>
        """, unsafe_allow_html=True)

        vid_mode = st.radio(
            "Select Video Input Mode:",
            ["Upload Drone Flight Video", "⚡ Try Built-in Demo Video"],
            horizontal=True,
            key="vid_source_mode"
        )

        target_video = None
        video_display_name = ""

        if vid_mode == "Upload Drone Flight Video":
            uploaded_video = st.file_uploader(
                "Select Drone Flight Video Recording",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_uploader"
            )
            if uploaded_video is not None:
                target_video = uploaded_video
                video_display_name = uploaded_video.name
                size_mb = getattr(uploaded_video, "size", 0) / (1024 * 1024)
                st.info(f"📁 Video ready: **{video_display_name}** ({size_mb:.1f} MB)")
        else:
            if os.path.exists(sample_vid_path):
                target_video = sample_vid_path
                video_display_name = "sample_track_flight.mp4"
                size_mb = os.path.getsize(sample_vid_path) / (1024 * 1024)
                st.info(f"📁 Built-in Demo Video ready: **{video_display_name}** ({size_mb:.1f} MB) - contains rail surface anomalies.")
            else:
                st.warning("Demo video not found in 'samples/sample_track_flight.mp4'. Please upload a video.")

        if target_video is not None:
            col_launch, col_stop_btn = st.columns([2, 1])
            with col_launch:
                start_btn = st.button("🚀 Launch Real-Time Flight Inspection", type="primary")

            if start_btn:
                with temporary_video_file(target_video) as temp_video_path:
                    pipeline = VideoInspectionPipeline(
                        video_path=temp_video_path,
                        detector=detector,
                        conf=conf_slider,
                        stride=stride_slider,
                        min_persistence_frames=persistence_slider,
                        imgsz=DEFAULT_VIDEO_IMGSZ,
                        line_thickness=box_thickness,
                        font_scale=font_scale_val,
                        auto_scale=auto_scale_box
                    )

                    col_vid, col_panel = st.columns([3, 2])
                    
                    with col_vid:
                        st.markdown("**Live Aerial Telemetry Stream**")
                        video_placeholder = st.empty()
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()

                    with col_panel:
                        st.markdown("**Live Incident Registry (Auto-Updating)**")
                        metrics_placeholder = st.empty()
                        table_placeholder = st.empty()

                    try:
                        frame_queue = queue.Queue(maxsize=3)
                        worker_errors = []
                        stop_event = threading.Event()

                        def video_stream_worker():
                            try:
                                for stream_item in pipeline.process_stream(stop_event=stop_event):
                                    while not stop_event.is_set():
                                        try:
                                            frame_queue.put(stream_item, timeout=0.2)
                                            break
                                        except queue.Full:
                                            continue
                                    if stop_event.is_set():
                                        break
                            except Exception as stream_err:
                                worker_errors.append(stream_err)
                            finally:
                                frame_queue.put(None)

                        # Set thread stack safely
                        old_stack = threading.stack_size(16 * 1024 * 1024)
                        worker_thread = threading.Thread(target=video_stream_worker, daemon=True)
                        worker_thread.start()
                        threading.stack_size(old_stack)

                        last_reported_frame = 0

                        while True:
                            try:
                                item = frame_queue.get(timeout=0.1)
                            except queue.Empty:
                                if not worker_thread.is_alive():
                                    break
                                continue

                            if item is None:
                                break

                            current_idx, total_frames, annotated_frame, active_detections = item
                            last_reported_frame = current_idx
                            
                            if total_frames > 0:
                                progress_ratio = min(1.0, current_idx / total_frames)
                                progress_bar.progress(progress_ratio)
                                status_text.text(f"Scanning frame {current_idx} of {total_frames} ({(progress_ratio * 100):.1f}%)")
                            else:
                                progress_bar.progress(0.5)
                                status_text.text(f"Scanning frame {current_idx}...")
                            
                            ret_enc, enc_buf = cv2.imencode('.jpg', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
                            if ret_enc:
                                video_placeholder.image(enc_buf.tobytes(), output_format="JPEG", use_container_width=True)
                            else:
                                video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                            del annotated_frame

                            # Refresh metrics and table periodically or when new defects are spotted
                            if current_idx % 4 == 0 or (len(active_detections) > 0 and current_idx % 2 == 0):
                                all_incidents = pipeline.get_all_incidents(sort_by_recent=True)
                                verified = pipeline.get_verified_incidents()

                                metrics_placeholder.markdown(f"""
                                <div style="display: flex; gap: 8px; margin-bottom: 12px;">
                                    <div class="rs-card" style="flex: 1; padding: 10px 14px;">
                                        <div class="rs-card-bar bar-amber"></div>
                                        <div class="rs-card-label">In-Frame</div>
                                        <div class="rs-card-value">{len(active_detections)}</div>
                                    </div>
                                    <div class="rs-card" style="flex: 1; padding: 10px 14px;">
                                        <div class="rs-card-bar bar-blue"></div>
                                        <div class="rs-card-label">Total Logged</div>
                                        <div class="rs-card-value">{len(all_incidents)}</div>
                                    </div>
                                    <div class="rs-card" style="flex: 1; padding: 10px 14px;">
                                        <div class="rs-card-bar bar-emerald"></div>
                                        <div class="rs-card-label">Verified</div>
                                        <div class="rs-card-value">{len(verified)}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                if all_incidents:
                                    df_live = incidents_to_dataframe(all_incidents, min_persistence=persistence_slider)
                                    cols_to_show = [c for c in ["Track ID", "Defect Category", "Severity", "Status", "Peak Confidence", "Frames Tracked", "Timestamp (s)"] if c in df_live.columns]
                                    table_placeholder.dataframe(
                                        df_live[cols_to_show],
                                        use_container_width=True,
                                        height=320
                                    )
                                    del df_live
                                else:
                                    table_placeholder.info("Tracking active... Results update immediately when defects appear.")

                        stop_event.set()
                        worker_thread.join(timeout=1.0)
                        if worker_errors:
                            raise worker_errors[0]

                        progress_bar.progress(1.0)
                        status_text.success("🎉 Video flight scan completed successfully!")

                        # Final Verified Report
                        all_final_incidents = pipeline.get_all_incidents(sort_by_recent=False)
                        verified_incidents = pipeline.get_verified_incidents()
                        st.markdown("---")
                        st.markdown("### 📑 Final Verified Anomaly Report")
                        st.write(f"Displaying all logged incidents across flight (Verified filter: $\ge$ {persistence_slider} frame(s)).")

                        df_final = incidents_to_dataframe(all_final_incidents, min_persistence=persistence_slider)
                        if not df_final.empty:
                            st.dataframe(df_final, use_container_width=True)
                            final_csv = dataframe_to_csv_bytes(df_final)
                            clean_stem = os.path.splitext(video_display_name)[0]
                            st.download_button(
                                label="📥 Export Complete Anomaly Report (CSV)",
                                data=final_csv,
                                file_name=f"rail_anomaly_report_{clean_stem}.csv",
                                mime="text/csv",
                                key="download_csv_btn"
                            )
                        else:
                            st.success("✅ Clean Flight: Zero rail defects detected across this flight.")

                    except Exception as err:
                        st.error(f"❌ Video analysis encountered an error: {err}")


if __name__ == "__main__":
    main()
