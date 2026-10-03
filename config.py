from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
DEFAULT_WEIGHTS = str(WEIGHTS_DIR / "best.pt")

# Inference defaults
DEFAULT_CONF = 0.35
DEFAULT_STRIDE = 3
DEFAULT_LINE_THICKNESS = 4
DEFAULT_FONT_SCALE = 0.85
DEFAULT_VIDEO_IMGSZ = 480

# Temporal filtering defaults
MIN_PERSISTENCE_FRAMES = 3  # Filters out single/two-frame false-alarm noise
