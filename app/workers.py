"""
Background workers for heavy processing tasks.
Runs in QThread to keep the UI responsive.
"""

import threading
from PySide6.QtCore import QThread, Signal

from services.video_processor import VideoProcessor
from logger import get_logger

log = get_logger(__name__)

# UI-1 fix: Shared VideoProcessor instance to avoid reloading models (10-30s) per run
_shared_processor: VideoProcessor | None = None
_processor_lock = threading.Lock()


def _get_processor() -> VideoProcessor:
    """Get or create the shared VideoProcessor (models loaded once). Thread-safe."""
    global _shared_processor
    with _processor_lock:
        if _shared_processor is None:
            log.info("Creating shared VideoProcessor (first run — models will load)")
            _shared_processor = VideoProcessor()
        return _shared_processor


class ProcessingThread(QThread):
    """Runs the full video-processing pipeline in a background thread."""

    progress_signal = Signal(str, int)
    finished_signal = Signal(dict)

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self._cancelled = False

    # ── Cancel support ───────────────────────────────
    def request_cancel(self):
        log.info("Cancel requested for processing thread")
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    # ── Thread entry ─────────────────────────────────
    def run(self):
        try:
            log.info("ProcessingThread started for %s", self.video_path)
            processor = _get_processor()
            

            results = processor.process_file(
                self.video_path,
                progress_callback=self.progress_signal.emit,
                cancel_check=lambda: self._cancelled,
            )

            if self._cancelled:
                self.progress_signal.emit("Cancelled.", 0)
                self.finished_signal.emit({"clips": [], "segments": [], "errors": ["Processing cancelled by user."]})
            else:
                log.info("ProcessingThread finished. Clips: %d", len(results.get("clips", [])))
                self.finished_signal.emit(results)

        except Exception as exc:
            log.exception("ProcessingThread failed")
            self.progress_signal.emit(f"Error: {exc}", 0)
            self.finished_signal.emit({"clips": [], "segments": [], "errors": [str(exc)]})
