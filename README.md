---
title: RailSamiksha AI - AeroInspect RT-DETR
emoji: 🚂
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# RailSamiksha AI | AeroInspect RT-DETR Railway Inspection Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Ultralytics RT-DETR](https://img.shields.io/badge/Model-RT--DETR-blue.svg)](https://github.com/ultralytics/ultralytics)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)

Autonomous aerial drone surveillance and railway track defect identification pipeline powered by **Ultralytics RT-DETR (Real-Time Detection Transformer)**, persistent spatial tracking, and temporal verification filtering.

---

## 🌟 Key Capabilities

- **High-Resolution Aerial Still Inspection**: Multi-scale detection across 2K/4K drone orthophotos with automatic bounding box & label font scaling.
- **Flight Video Pipeline with Persistent Tracking**: Real-time video demuxing with frame stride sampling, spatial IoU continuity matching, and collision-free track assignment.
- **Temporal Verification Filter**: Deduplicates repeated detections across frames and suppresses single-frame transient false positives.
- **Telemetry HUD**: Embedded aerospace telemetry bar with real-time frame index, flight time, in-view defect count, and cumulative verified incidents.
- **Auto-Updating Incident Registry & Export**: Live incident table with peak confidence, defect severity, start/end frames, and 1-click CSV report export.
- **Built-in Demo Assets**: Ships with sample aerial orthophoto and video clip for immediate testing on Hugging Face Spaces.

---

## 📋 Defect Taxonomy & Severity Matrix

The model detects 9 critical railway track defect classes:

| Class ID | Defect Category | Default Severity | Visual Indicator |
| :---: | :--- | :---: | :---: |
| **6** | Rail Crack | `CRITICAL` | 🔴 Laser Red |
| **2** | Broken Sleeper | `CRITICAL` | 🔴 Crimson Rose |
| **4** | Missing Sleeper | `CRITICAL` | 🔴 Magenta |
| **7** | Rail Joint Damage | `HIGH` | 🟠 Solar Orange |
| **1** | Ballast Washout | `HIGH` | 🟠 Amber Orange |
| **5** | Missing Fastener | `MEDIUM` | 🟡 Vivid Amber |
| **3** | Damaged Fastener | `MEDIUM` | 🟡 Electric Gold |
| **0** | Ballast Deficiency | `LOW` | 🔵 Cyber Cyan |
| **8** | Rail Misalignment | `LOW` | 🔵 Aerospace Blue |

---

## 🚀 Quickstart: Running Locally

### 1. Prerequisites
- Python 3.9 - 3.11 recommended
- (Optional) Git LFS for downloading large model weights

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/<your-username>/rtdetr_inspection_app.git
cd rtdetr_inspection_app

# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Weights
Verify that `weights/best.pt` exists (~63MB). If cloning from GitHub with Git LFS:
```bash
git lfs pull
```

### 4. Launch Application
```bash
# Direct Streamlit launch
streamlit run app.py

# Or run via launcher script (optimized for CPU thread stability)
python run_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deployment Guides

### Deploy to Hugging Face Spaces (Free Tier)

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Choose **Streamlit** as the Space SDK and **CPU (Free)**.
3. Clone your Space repo or initialize Git:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push space main
   ```
4. **Git LFS Note**: Hugging Face Spaces supports Git LFS natively. Ensure `.gitattributes` tracks `*.pt` so `weights/best.pt` is uploaded cleanly.

### Deploy to GitHub

```bash
git init
git add .gitattributes
git add .gitignore requirements.txt packages.txt README.md app.py config.py run_app.py core/ utils/ samples/
git add weights/best.pt
git commit -m "feat: initial commit of RailSamiksha RT-DETR inspection prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

---

## 🏗️ Project Architecture

```
rtdetr_inspection_app/
│
├── app.py                   # Main Streamlit web application & telemetry dashboard
├── config.py                # Pipeline hyperparameters, model paths, defaults
├── run_app.py               # Production launcher script with CPU thread guards
├── requirements.txt         # Python dependencies
├── packages.txt             # Debian/Ubuntu system packages (ffmpeg, etc.)
├── README.md                # Documentation & Hugging Face Spaces configuration
├── .gitattributes           # Git LFS configuration for binary model weights
├── .gitignore               # Ignored files (virtualenvs, temporary media)
│
├── core/
│   ├── detector.py          # RT-DETR Ultralytics wrapper & annotation rendering
│   ├── video_engine.py      # Spatial tracking, frame stride sampling & HUD engine
│   └── exporter.py          # Incident aggregation, severity logic & CSV exporter
│
├── utils/
│   └── temp_manager.py      # Safe video file context manager & chunk streamer
│
├── samples/
│   ├── sample_track_image.jpg  # Built-in sample drone orthophoto
│   └── sample_track_flight.mp4 # Built-in sample drone flight video
│
└── weights/
    └── best.pt              # Trained RT-DETR-L railway defect weights (~63MB)
```

---

## 📄 License
This project is licensed under the MIT License.
