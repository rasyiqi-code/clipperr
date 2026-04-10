import os
import requests
import certifi
import zipfile
import shutil

from huggingface_hub import hf_hub_download, snapshot_download
from PySide6.QtCore import QObject, QThread, Signal

import config
from logger import get_logger

log = get_logger(__name__)


class ProgressBridge:
    """Bridges tqdm progress from huggingface_hub to PySide6 signals.
    Implements full tqdm interface to avoid AttributeErrors."""
    def __init__(self, filename, signal, *args, **kwargs):
        self.filename = filename
        self.signal = signal
        self.total = 0
        self.current = 0

    def update(self, n=0):
        self.current += n
        if self.total > 0:
            pct = int((self.current / self.total) * 100)
            scaled_pct = 5 + int(pct * 0.9)
            self.signal.emit(f"Downloading {self.filename}: {pct}%", scaled_pct)

    def set_description(self, desc, refresh=True): pass
    def close(self): pass
    def clear(self, *args, **kwargs): pass
    def refresh(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass


class DownloadWorker(QObject):
    """Performs file downloads (HuggingFace or direct URL)."""

    progress_signal = Signal(str, int)
    finished_signal = Signal(str, bool)

    def download_model(self, repo_id: str, filenames: list[str], local_dir: str, token: str | None = None):
        try:
            os.makedirs(local_dir, exist_ok=True)
            log.info("Starting granular download for %s into %s", repo_id, local_dir)
            
            # If no files specified, use snapshot_download (likely a complex repo like Qwen)
            if not filenames:
                self.progress_signal.emit(f"Verifying {repo_id} (Snapshot)...", 30)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    token=None,
                    local_dir_use_symlinks=False,
                )
            else:
                # Download files one by one for granular progress
                for i, fname in enumerate(filenames):
                    self.progress_signal.emit(f"Fetching {fname}...", 5 + int((i / len(filenames)) * 90))
                    
                    # Create a bridge for tqdm
                    bridge = ProgressBridge(fname, self.progress_signal)
                    
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=fname,
                        local_dir=local_dir,
                        token=None,
                        local_dir_use_symlinks=False,
                        tqdm_class=lambda **kwargs: bridge # Inject our bridge as the tqdm class
                    )

            self.progress_signal.emit("Setup complete!", 100)
            self.finished_signal.emit(repo_id, True)
            log.info("Model download complete: %s", repo_id)

        except Exception as exc:
            error_msg = str(exc)
            log.error("Download failed for %s: %s", repo_id, error_msg)
            
            if "401" in error_msg or "Unauthorized" in error_msg:
                display_msg = "Error: Auth Required (Gated Model?)"
            else:
                display_msg = f"Error: {error_msg[:60]}..." if len(error_msg) > 60 else f"Error: {error_msg}"

            self.progress_signal.emit(display_msg, 0)
            self.finished_signal.emit(repo_id, False)

    def download_url(self, url: str, filename: str, local_dir: str):
        try:
            os.makedirs(local_dir, exist_ok=True)
            filepath = os.path.join(local_dir, filename)
            self.progress_signal.emit(f"Connecting to {filename}...", 10)
            
            with requests.get(url, stream=True, verify=certifi.where()) as r:
                r.raise_for_status()
                total_length = r.headers.get('content-length')
                
                with open(filepath, 'wb') as f:
                    if total_length is None:
                        f.write(r.content)
                    else:
                        dl = 0
                        total_length = int(total_length)
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                dl += len(chunk)
                                f.write(chunk)
                                done = int(70 * dl / total_length) # 10-80% for download
                                self.progress_signal.emit(f"Downloading {filename}...", 10 + done)
            
            # Auto-extract if it's a ZIP
            if filename.endswith(".zip"):
                self.progress_signal.emit("Extracting files...", 85)
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    # Extract to a temp dir then move files if needed, or to local_dir directly
                    zip_ref.extractall(local_dir)
                
                # Cleanup zip file
                os.remove(filepath)
                
            self.progress_signal.emit("Download and setup complete!", 100)
            self.finished_signal.emit(filename, True)
            log.info("URL download complete: %s", local_dir)

        except Exception as exc:
            log.error("URL download failed: %s", exc)
            self.progress_signal.emit(f"Error: {exc}", 0)
            self.finished_signal.emit(filename, False)


class DownloadThread(QThread):
    """Convenience QThread wrapper around DownloadWorker for HuggingFace downloads."""

    progress_signal = Signal(str, int)
    finished_signal = Signal(str, bool)

    def __init__(self, repo_id: str, filenames: list[str], local_dir: str, token: str | None = None):
        super().__init__()
        self.repo_id = repo_id
        self.filenames = filenames
        self.local_dir = local_dir
        self.token = token

    def run(self):
        self._worker = DownloadWorker()
        self._worker.progress_signal.connect(self.progress_signal.emit)
        self._worker.finished_signal.connect(self.finished_signal.emit)
        self._worker.download_model(self.repo_id, self.filenames, self.local_dir, self.token)


class ModelManager:
    """Registry of downloadable AI models with local status checks."""

    def __init__(self):
        self.models = {
            "whisper-base": {
                "repo": "Systran/faster-whisper-base",
                "files": ["model.bin", "config.json", "vocabulary.txt", "tokenizer.json"],
                "path": config.WHISPER_MODEL_PATH,
            },
            "llm-analysis": {
                "repo": config.LLM_MODEL_ID,
                "files": ["config.json", "model.safetensors"], # Explicit check for Qwen
                "path": os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", f"models--{config.LLM_MODEL_ID.replace('/', '--')}", "snapshots"),
            },
            "blazeface": {
                "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
                "files": ["blaze_face_short_range.tflite"],
                "path": os.path.dirname(config.BLAZEFACE_MODEL_PATH),
            },
            "yunet-face": {
                "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                "files": ["face_detection_yunet_2023mar.onnx"],
                "path": os.path.dirname(config.YUNET_MODEL_PATH),
            },
        }

    def check_status(self, model_id: str) -> bool:
        if model_id not in self.models:
            return False
            
        model_info = self.models[model_id]
        target_dir = model_info["path"]
        
        # Special handling for snapshot-based paths (Qwen)
        if "snapshots" in target_dir:
            if not os.path.exists(target_dir):
                return False
            # Check if any snapshot subdirectory exists and contains required files
            try:
                for snapshot_id in os.listdir(target_dir):
                    snap_path = os.path.join(target_dir, snapshot_id)
                    if all(os.path.exists(os.path.join(snap_path, f)) for f in model_info["files"]):
                        return True
            except Exception:
                pass
            return False
            
        if not os.path.exists(target_dir):
            return False
            
        return all(
            os.path.exists(os.path.join(target_dir, f))
            for f in model_info["files"]
        )
