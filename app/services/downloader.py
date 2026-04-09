import os
import requests
import certifi

from huggingface_hub import hf_hub_download, snapshot_download
from PySide6.QtCore import QObject, QThread, Signal

import config
from logger import get_logger

log = get_logger(__name__)


class DownloadWorker(QObject):
    """Performs file downloads (HuggingFace or direct URL)."""

    progress_signal = Signal(str, int)
    finished_signal = Signal(str, bool)

    def download_model(self, repo_id: str, filenames: list[str], local_dir: str, token: str | None = None):
        try:
            os.makedirs(local_dir, exist_ok=True)
            
            # Use snapshot_download for better reliability (handles partials and full repos)
            self.progress_signal.emit(f"Verifying/Downloading: {repo_id}...", 30)
            
            # For faster-whisper and LLMs, we want specific weight formats if possible
            # but snapshot handles directory mapping much better than hf_hub_download manually
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=token,
                local_dir_use_symlinks=False, # Essential for Windows non-admin
                # avoid redundant weight formats for whisper
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.ckpt", "*.safetensors"] if "whisper" in repo_id else []
            )

            self.progress_signal.emit("All files verified and ready!", 100)
            self.finished_signal.emit(repo_id, True)
            log.info("Model download complete: %s", repo_id)

        except Exception as exc:
            error_msg = str(exc)
            log.error("Download failed for %s: %s", repo_id, error_msg)
            
            # User-friendly mapping for common errors
            if "401" in error_msg or "Unauthorized" in error_msg:
                display_msg = "Error: Auth Failed (Check Token)"
            elif "1314" in error_msg or "required privilege" in error_msg:
                display_msg = "Error: Permission Denied (Disable Symlinks)"
            elif "CERTIFICATE_VERIFY_FAILED" in error_msg:
                display_msg = "Error: SSL Verification Failed"
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
                                done = int(80 * dl / total_length)
                                self.progress_signal.emit(f"Downloading {filename}...", 10 + done)
            
            self.progress_signal.emit("Download complete!", 100)
            self.finished_signal.emit(filename, True)
            log.info("URL download complete: %s", filepath)

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
        worker = DownloadWorker()
        worker.progress_signal.connect(self.progress_signal.emit)
        worker.finished_signal.connect(self.finished_signal.emit)
        worker.download_model(self.repo_id, self.filenames, self.local_dir, self.token)


class ModelManager:
    """Registry of downloadable AI models with local status checks."""

    def __init__(self):
        self.models = {
            "whisper-base": {
                "repo": "Systran/faster-whisper-base",
                "files": ["model.bin", "config.json", "vocabulary.txt"],
                "path": config.WHISPER_MODEL_PATH,
            },
            "llm-analysis": {
                "repo": config.LLM_MODEL_ID,
                "files": [],  # snapshot check
                "path": os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", f"models--{config.LLM_MODEL_ID.replace('/', '--')}"),
            },
            "blazeface": {
                "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
                "files": ["blaze_face_full_range.tflite"],
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
        
        if not os.path.exists(target_dir):
            return False
            
        if not model_info.get("files"):
            return True  # Directory check is sufficient for transformers cache
            
        return all(
            os.path.exists(os.path.join(target_dir, f))
            for f in model_info["files"]
        )
