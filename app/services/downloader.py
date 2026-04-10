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
    def __init__(self, filename, signal, file_idx=0, total_files=1, weights=None, **kwargs):
        self.filename = filename
        self.signal = signal
        self.file_idx = file_idx
        self.total_files = total_files
        self.weights = weights or {}
        self.total = kwargs.get('total', 0)
        self.current = 0
        
        # Calculate base progress from previous files
        all_files = list(self.weights.keys()) if self.weights else []
        self.base_progress = 0
        if all_files and filename in all_files:
            idx = all_files.index(filename)
            sum_prev = sum(self.weights[f] for f in all_files[:idx])
            total_sum = sum(self.weights.values())
            self.base_progress = (sum_prev / total_sum) if total_sum > 0 else 0
            self.file_weight = (self.weights[filename] / total_sum) if total_sum > 0 else (1.0 / total_files)
        else:
            self.base_progress = file_idx / total_files
            self.file_weight = 1.0 / total_files

    def update(self, n=0):
        self.current += n
        if self.total > 0:
            # Local file percentage (0.0 to 1.0)
            file_pct = (self.current / self.total)
            
            # Weighted overall progress
            overall_pct_float = (self.base_progress + (file_pct * self.file_weight)) * 100
            overall_pct = int(overall_pct_float)
            
            # Clamp to 99% during download, 100% is reserved for final success
            display_pct = max(1, min(99, overall_pct))
            
            self.signal.emit(f"Downloading {self.filename}: {int(file_pct*100)}% ({display_pct}%)", display_pct)

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
                # Define weights for progress reporting (model.safetensors is 988MB, others are small)
                weights = {f: 1 for f in filenames}
                if "model.safetensors" in weights:
                    weights["model.safetensors"] = 1000 # 1000x heavier
                if "model.bin" in weights:
                    weights["model.bin"] = 1000

                # Download files one by one for granular progress
                for i, fname in enumerate(filenames):
                    # Emit immediate status so it doesn't stay at "Starting..."
                    self.progress_signal.emit(f"Connecting to {fname}...", 5 + int((i / len(filenames)) * 5))

                    # Create a bridge for tqdm with multi-file context and byte-weighting
                    bridge = ProgressBridge(fname, self.progress_signal, i, len(filenames), weights)
                    
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
            
            # DO NOT clean up directory anymore—we want to support resuming large files.
            # hf_hub_download handles its own .incomplete files.
            
            if "401" in error_msg or "Unauthorized" in error_msg:
                display_msg = "Error: Auth Required (Gated Model?)"
            else:
                elif "disk" in error_msg.lower() or "space" in error_msg.lower():
                display_msg = "Error: Disk Full!"
            else:
                display_msg = f"Error: {error_msg[:100]}" if len(error_msg) > 100 else f"Error: {error_msg}"

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
                "files": [
                    "config.json", "generation_config.json", "model.safetensors",
                    "tokenizer.json", "tokenizer_config.json", "vocab.json",
                    "merges.txt", "LICENSE", "README.md", ".gitattributes"
                ],
                "path": config.LLM_MODEL_PATH,
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
        
        # Simplified check for local non-nested models (Whisper, LLM)
        if not os.path.exists(target_dir):
            return False
            
        # If it's the LLM (no specific files listed), just check if the directory is non-empty
        # and contains a config.json which is standard for HF models.
        if not model_info["files"]:
            return os.path.exists(os.path.join(target_dir, "config.json"))
            
        if not os.path.exists(target_dir):
            return False
            
        return all(
            os.path.exists(os.path.join(target_dir, f))
            for f in model_info["files"]
        )
