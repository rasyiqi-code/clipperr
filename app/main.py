"""
ClipperR — AI-powered video clipper.
Entry point and main window orchestrator.
"""

import sys
import os
import warnings
import certifi

# Fix SSL certificates for frozen apps - must be done BEFORE other imports that use networking
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['HTTPS_CA_BUNDLE'] = certifi.where()

# Suppress all non-essential UserWarnings (like torchcodec/cuda/FFmpeg version mismatches)
# so the terminal stays clean for the user.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchcodec.*")

# Suppress AV1 hardware decoding warnings from Qt Multimedia (FFmpeg backend)
os.environ["QT_MEDIA_BACKEND"] = "ffmpeg"
os.environ["AV_LOG_LEVEL"] = "quiet"

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from dotenv import load_dotenv

from config import APP_NAME, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT, WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT
from styles import MAIN_STYLE
from pages import HomePage, ResultsPage, SettingsPage, MonitorPage
from workers import ProcessingThread
from logger import get_logger, setup_logging

# Bootstrap logging before anything else
setup_logging()
log = get_logger(__name__)

# Note: HF_TOKEN is now handled by the 'prefs' singleton in config.py


class clipperrApp(QMainWindow):
    """Main application window — orchestrates pages and processing."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon("app/assets/icon.png"))
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)
        self.setStyleSheet(MAIN_STYLE)

        self._thread: ProcessingThread | None = None

        # Root layout
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        self._nav_btns: list[QPushButton] = []
        root.addWidget(self._build_sidebar())

        # Page stack
        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)

        # Pages
        self._home = HomePage()
        self._results = ResultsPage()
        self._settings = SettingsPage()
        self._monitor = MonitorPage()

        self._stack.addWidget(self._home)
        self._stack.addWidget(self._results)
        self._stack.addWidget(self._settings)
        self._stack.addWidget(self._monitor)

        # Connections
        self._home.file_selected.connect(self._process_video)
        self._home.cancel_requested.connect(self._cancel_processing)

    # ══════════════════════════════════════════════════
    #  Sidebar
    # ══════════════════════════════════════════════════
    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        layout = QVBoxLayout(sidebar)

        title = QLabel(APP_NAME)
        title.setObjectName("SidebarTitle")
        layout.addWidget(title)

        nav_items = [
            ("🏠", "Dashboard"), 
            ("📜", "Library"), 
            ("⚙", "Settings"),
            ("📈", "Monitor"),
        ]
        layout.addSpacing(10)
        for i, (icon, text) in enumerate(nav_items):
            btn = QPushButton(f"{icon}  {text}")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setObjectName("NavButton")
            btn.setProperty("active", "true" if i == 0 else "false")
            btn.clicked.connect(lambda _checked, idx=i: self._switch_page(idx))
            layout.addWidget(btn)
            self._nav_btns.append(btn)

        layout.addStretch()
        return sidebar

    def _switch_page(self, index: int):
        self._stack.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_btns):
            btn.setProperty("active", "true" if i == index else "false")
        # Re-apply stylesheet so dynamic properties take effect
        self.setStyleSheet(self.styleSheet())

    # ══════════════════════════════════════════════════
    #  Processing
    # ══════════════════════════════════════════════════
    def _process_video(self, path: str):
        log.info("process_video called for %s", path)

        # Check models
        missing = self._settings.missing_models()
        if missing:
            log.warning("Missing models: %s", missing)
            self._home.set_status(f"⚠️ Missing: {', '.join(missing)}")
            self._switch_page(2)  # Settings
            return

        self._home.set_status(f"🚀 Processing: {os.path.basename(path)}")
        self._home.set_progress(0)
        self._home.set_processing(True)

        self._thread = ProcessingThread(path)
        self._thread.progress_signal.connect(self._on_progress)
        self._thread.finished_signal.connect(self._on_finished)
        self._thread.start()

    def _cancel_processing(self):
        if self._thread and self._thread.isRunning():
            self._thread.request_cancel()
            self._home.set_status("⏹ Cancelling...")

    def _on_progress(self, msg: str, val: int):
        self._home.set_status(f"⚡ {msg}")
        self._home.set_progress(val)

    def _on_finished(self, results: dict):
        clips = results.get("clips", [])
        errors = results.get("errors", [])
        log.info("Processing finished. Clips: %d, Errors: %d", len(clips), len(errors))

        if not clips:
            if errors:
                self._home.set_status("❌ Failed! Check Library for errors.")
            else:
                self._home.set_status("ℹ️ No Viral Moments Found.")
        else:
            if errors:
                self._home.set_status("⚠️ Completed with Errors.")
            else:
                self._home.set_status("✅ Done!")

        self._home.set_progress(100)
        self._home.set_processing(False)

        self._results.show_results(results)
        self._switch_page(1)  # Results


# ── Entry point ──────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("app/assets/icon.png"))
    window = clipperrApp()
    window.show()
    sys.exit(app.exec())
