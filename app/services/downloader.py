import os
import requests
import certifi

from huggingface_hub import hf_hub_download
from PySide6.QtCore import QObject, QThread, Signal

from logger import get_logger

log = get_logger(__name__)


class DownloadWorker(QObject):
    """Performs file downloads (HuggingFace or direct URL)."""

    progress_signal = Signal(str, int)
    finished_signal = Signal(str, bool)

    def download_model(self, repo_id: str, filenames: list[str], local_dir: str, token: str | None = None):
        try:
            os.makedirs(local_dir, exist_ok=True)
            
            if not filenames:
                # If no specific files are listed, download the whole repo to the HF cache
                from huggingface_hub import snapshot_download
                self.progress_signal.emit(f"Downloading full repository: {repo_id} (this may take a while)...", 50)
                snapshot_download(
                    repo_id=repo_id,
                    token=token,
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.ckpt"] # avoid redundant weight formats
                )
            else:
                total = len(filenames)
                for i, filename in enumerate(filenames):
                    self.progress_signal.emit(f"Downloading {i+1}/{total}: {filename}...", int((i / total) * 100))
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=local_dir,
                        token=token,
                    )

            self.progress_signal.emit("All files downloaded!", 100)
            self.finished_signal.emit(repo_id, True)
            log.info("Model download complete: %s", repo_id)

        except Exception as exc:
            error_msg = str(exc)
            if "401" in error_msg or "403" in error_msg:
                error_msg = "Auth Failed (Token Required)"
            log.error("Download failed for %s: %s", repo_id, error_msg)
            self.progress_signal.emit(f"Error: {error_msg}", 0)
            self.finished_signal.emit(repo_id, False)

    def download_url(self, url: str, filename: str, local_dir: str):
        try:
            os.makedirs(local_dir, exist_ok=True)
            filepath = os.path.join(local_dir, filename)
            self.progress_signal.emit(f"Connecting to {filename}...", 10)
            
            # Use requests with certifi for better SSL support in frozen apps
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
                                # Progress between 10% and 90%
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

    def __init__(self, base_path: str = "models"):
        self.base_path = base_path
        self.models = {
            "whisper-base": {
                "repo": "Systran/faster-whisper-base",
                "files": ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"],
                "path": os.path.join(base_path, "whisper/base"),
            },
            "llm-analysis": {
                "repo": "Qwen/Qwen2.5-0.5B-Instruct",
                "files": [],  # Handled natively by transformers snapshot/cache
                "path": os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"),
            },
            "blazeface": {
                "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
                "files": ["blaze_face_full_range.tflite"], # Standardizing name
                "path": os.path.join(base_path, "mediapipe"),
            },
            "yunet-face": {
                "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                "files": ["face_detection_yunet_2023mar.onnx"],
                "path": os.path.join(base_path, "opencv"),
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
