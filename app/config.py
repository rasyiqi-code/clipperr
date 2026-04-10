"""
Centralized configuration for ClipperR.
All hardcoded values live here so they can be changed in one place.
"""

import os



# ── Paths ──────────────────────────────────────────────
import sys

# Detect if the app is run as a bundle (PyInstaller)
if getattr(sys, 'frozen', False):
    # Running in a frozen bundle (.exe)
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Running in a normal Python environment
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "exports")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

WHISPER_MODEL_PATH = os.path.join(MODELS_DIR, "whisper", "base")
LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_MODEL_PATH = os.path.join(MODELS_DIR, "llm", "qwen")
BLAZEFACE_MODEL_PATH = os.path.join(MODELS_DIR, "mediapipe", "blaze_face_short_range.tflite")
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "opencv", "face_detection_yunet_2023mar.onnx")

# ── Face Tracking ────────────────────────────────────
FACE_TRACKING_SAMPLES = 20  # Increased slightly for better KLT stability

# Detection Gates (defaults — overridden by scene classifier)
FACE_Y_GATE_DEFAULT = 0.65         # Max Y-position for face detection
FACE_MIN_AREA_DEFAULT = 0.003      # Min normalized area (0.3% of frame)
FACE_UNCONFIRMED_SCORE_MIN = 0.75  # Min score for detection

# Clustering & Scoring
FACE_CLUSTER_THRESHOLD = 0.07      # Spatial clustering distance (normalized)
FACE_TEMPORAL_ALPHA = 0.3          # EMA smoothing factor (0.3 = 70% prev, 30% new)

# Camera Cut Detection
CAMERA_CUT_AREA_RATIO = 2.0        # Area change ratio to detect a cut
CAMERA_CUT_POSITION_SHIFT = 0.15   # Min simultaneous position shift for a cut
CAMERA_CUT_MIN_SEGMENT = 1.5       # Min sub-segment duration (seconds)

# ── Transcription ──────────────────────────────────────
TRANSCRIPTION_LANGUAGE = "id"       # ISO 639-1 code
TRANSCRIPTION_BEAM_SIZE = 5

# ── Video Rendering ───────────────────────────────────
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "fast"
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "128k"

# ── LLM Analysis ──────────────────────────────────────
LLM_MAX_TOKENS = 2048
LLM_TOP_K = 40
LLM_TEMPERATURE = 0.7
MAX_VIRAL_CLIPS = 10

ANALYSIS_CHUNK_DURATION = 600  # 10 minutes per chunk
ANALYSIS_CHUNK_OVERLAP = 30    # 30 seconds overlap

import json

# ── UI ─────────────────────────────────────────────────
APP_NAME = "clipperr"
WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 550
WINDOW_DEFAULT_WIDTH = 1000
WINDOW_DEFAULT_HEIGHT = 700

# ── User Preferences ──────────────────────────────────
USER_SETTINGS_FILE = os.path.join(BASE_DIR, "user_settings.json")

class UserSettings:
    """Simple JSON-based persistent dynamic settings."""
    def __init__(self):
        self.watermark_path = ""
        self.auto_thumbnail = False
        self.watermark_type = "image" # 'image' or 'text'
        self.watermark_text = ""
        self.watermark_pos = "top_left"
        self.watermark_opacity = 1.0
        self.llm_provider = "local"  # 'local' or 'api'
        self.openrouter_key = ""
        self.openrouter_model = "google/gemini-2.0-flash-lite:free"
        self.llm_max_tokens = 2048
        self.llm_top_k = 40
        self.llm_temperature = 0.7
        self.max_viral_clips = 10
        self.load()

    def load(self):
        if os.path.exists(USER_SETTINGS_FILE):
            try:
                with open(USER_SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                    self.watermark_path = data.get("watermark_path", "")
                    self.auto_thumbnail = data.get("auto_thumbnail", False)
                    self.watermark_type = data.get("watermark_type", "image")
                    self.watermark_text = data.get("watermark_text", "")
                    self.watermark_pos = data.get("watermark_pos", "top_left")
                    self.watermark_opacity = data.get("watermark_opacity", 1.0)
                    self.llm_provider = data.get("llm_provider", "local")
                    self.openrouter_key = data.get("openrouter_key", "")
                    self.openrouter_model = data.get("openrouter_model", "google/gemini-2.0-flash-lite:free")
                    self.llm_max_tokens = data.get("llm_max_tokens", 2048)
                    self.llm_top_k = data.get("llm_top_k", 40)
                    self.llm_temperature = data.get("llm_temperature", 0.7)
                    self.max_viral_clips = data.get("max_viral_clips", 10)
            except Exception:
                pass

    def save(self):
        try:
            with open(USER_SETTINGS_FILE, "w") as f:
                json.dump({
                    "watermark_path": self.watermark_path,
                    "auto_thumbnail": self.auto_thumbnail,
                    "watermark_type": self.watermark_type,
                    "watermark_text": self.watermark_text,
                    "watermark_pos": self.watermark_pos,
                    "watermark_opacity": self.watermark_opacity,
                    "llm_provider": self.llm_provider,
                    "openrouter_key": self.openrouter_key,
                    "openrouter_model": self.openrouter_model,
                    "llm_max_tokens": self.llm_max_tokens,
                    "llm_top_k": self.llm_top_k,
                    "llm_temperature": self.llm_temperature,
                    "max_viral_clips": self.max_viral_clips,
                }, f)
        except Exception:
            pass  # Fail silently — settings are non-critical

# Global singleton
prefs = UserSettings()
