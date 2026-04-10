import os
import subprocess

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QPushButton, QMessageBox, QApplication
)
from PySide6.QtCore import Qt

from logger import get_logger

from services.history_manager import HistoryManager
from widgets import VideoPlayer

log = get_logger(__name__)


class ResultsPage(QWidget):
    """Page with a historical clip list on the left and a video player on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_manager = HistoryManager()

        # ── Main Layout (Horizontal) ──
        self._main_layout = QHBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # 1. Left Side: Scrollable List
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet("background-color: #020617;")
        
        self._container = QWidget()
        self._container.setStyleSheet("background-color: #020617;")
        self._scroll.setWidget(self._container)

        self._list_layout = QVBoxLayout(self._container)
        self._list_layout.setContentsMargins(30, 25, 30, 25)
        self._list_layout.setSpacing(15)

        self._header = QLabel("Results History")
        self._header.setStyleSheet("font-size: 22px; font-weight: 800;")
        self._list_layout.addWidget(self._header)

        self._info = QLabel("Process a video to see results here.")
        self._info.setStyleSheet("color: #64748b; font-size: 13px;")
        self._list_layout.addWidget(self._info)

        self._list_layout.addStretch()
        
        self._main_layout.addWidget(self._scroll, 3) # Weight 3

        # 2. Right Side: Video Player Sidebar
        self._player = VideoPlayer()
        self._main_layout.addWidget(self._player, 2) # Weight 2
        
        self._playing_path = None

        # Load existing history on startup
        self.refresh_history()

    # ── Public API ───────────────────────────────────
    def show_results(self, results: dict):
        """Save new results to history and refresh the page."""
        clips = results.get("clips", [])
        errors = results.get("errors", [])

        if clips:
            self.history_manager.add_clips(clips)
        
        # Always refresh to rebuild the list cleanly (removes stale error labels too)
        self.refresh_history()
        
        if errors:
            self._info.setText(f"Found {len(clips)} clips. Some errors occurred:")
            for err in errors:
                err_label = QLabel(f"⚠️ {err}")
                err_label.setWordWrap(True)
                err_label.setStyleSheet("color: #f59e0b; font-size: 12px;")
                # Insert before the trailing stretch
                self._list_layout.insertWidget(self._list_layout.count() - 1, err_label)
        elif not clips and not self.history_manager.clips:
            self._info.setText("No clips found. Try a different video.")

    def refresh_history(self):
        """Reload all cards from HistoryManager."""
        self._clear_cards()
        clips = self.history_manager.clips
        
        if clips:
            self._info.setText(f"Total history: {len(clips)} clips.")
            for clip in clips:
                self._add_clip_card(clip)
        else:
            self._info.setText("No history available. Process a video to get started.")

        self._list_layout.addStretch()

    # ── Internal helpers ─────────────────────────────
    def _clear_cards(self):
        """Remove all items after header and info label (index 0 and 1)."""
        while self._list_layout.count() > 2:
            item = self._list_layout.takeAt(2)
            w = item.widget()
            if w:
                w.deleteLater()
            # Also handles spacers and sub-layouts that accumulate

    def _add_clip_card(self, clip: dict):
        card = QFrame()
        card.setObjectName("ClipCard")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(8)

        # Title row
        title_row = QHBoxLayout()
        title = QLabel(clip.get("title", "Untitled"))
        title.setObjectName("ClipTitle")
        time_label = QLabel(f"{clip['start']:.1f}s — {clip['end']:.1f}s")
        time_label.setObjectName("ClipDuration")
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(time_label)
        card_layout.addLayout(title_row)

        # Explanation
        if clip.get("explanation"):
            desc = QLabel(clip["explanation"])
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #94a3b8; font-size: 12px;")
            card_layout.addWidget(desc)

        # --- Social Metadata (Integrated from Social Page) ---
        if any([clip.get("title"), clip.get("description"), clip.get("hashtags")]):
            social_area = QFrame()
            social_area.setStyleSheet("background-color: rgba(2, 6, 23, 0.4); border-radius: 12px; margin: 5px 0; border: 1px solid rgba(56, 189, 248, 0.1);")
            social_layout = QVBoxLayout(social_area)
            social_layout.setContentsMargins(15, 12, 15, 12)
            social_layout.setSpacing(10)

            def add_social_field(label, value, copy_text):
                f_layout = QVBoxLayout()
                h_layout = QHBoxLayout()
                
                lbl = QLabel(label)
                lbl.setStyleSheet("color: #38bdf8; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;")
                h_layout.addWidget(lbl)
                h_layout.addStretch()
                
                cp_btn = QPushButton("Copy")
                cp_btn.setFixedSize(45, 20)
                cp_btn.setCursor(Qt.PointingHandCursor)
                cp_btn.setStyleSheet("font-size: 9px; font-weight: 800; background-color: #1e293b; color: #f8fafc; border-radius: 4px;")
                cp_btn.clicked.connect(lambda _=None, t=copy_text: QApplication.clipboard().setText(t))
                h_layout.addWidget(cp_btn)
                f_layout.addLayout(h_layout)

                val = QLabel(value)
                val.setWordWrap(True)
                val.setTextInteractionFlags(Qt.TextSelectableByMouse)
                val.setStyleSheet("color: #cbd5e1; font-size: 13px; line-height: 1.4;")
                f_layout.addWidget(val)
                social_layout.addLayout(f_layout)

            if clip.get("title"):
                add_social_field("VIRAL TITLE", clip["title"], clip["title"])
            if clip.get("description"):
                add_social_field("DESCRIPTION", clip["description"], clip["description"])
            if clip.get("hashtags"):
                add_social_field("HASHTAGS", clip["hashtags"], clip["hashtags"])

            card_layout.addWidget(social_area)

        # Action row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        
        # Delete button
        del_btn = QPushButton("🗑️")
        del_btn.setToolTip("Delete from history and disk")
        del_btn.setFixedSize(32, 32)
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.setStyleSheet("background-color: #450a0a; color: #ef4444; border: 1px solid #7f1d1d; border-radius: 8px;")
        output_path = clip.get("output_path", "")
        del_btn.clicked.connect(lambda _checked, p=output_path: self._delete_clip(p))
        btn_row.addWidget(del_btn)

        btn_row.addStretch()

        # Preview & Open buttons
        if output_path and os.path.exists(output_path):
            prev_btn = QPushButton("🎬 Preview")
            prev_btn.setObjectName("SecondaryButton")
            prev_btn.setCursor(Qt.PointingHandCursor)
            prev_btn.setFixedWidth(100)
            prev_btn.clicked.connect(lambda _checked, p=output_path: self._preview_clip(p))
            btn_row.addWidget(prev_btn)

            open_btn = QPushButton("📂 Open File")
            open_btn.setObjectName("ActionButton")
            open_btn.setCursor(Qt.PointingHandCursor)
            open_btn.setFixedWidth(100)
            open_btn.clicked.connect(lambda _checked, p=output_path: self._open_file(p))
            btn_row.addWidget(open_btn)
        else:
            status = QLabel("(File missing)")
            status.setStyleSheet("color: #475569; font-style: italic; font-size: 11px;")
            btn_row.addWidget(status)

        card_layout.addLayout(btn_row)
        self._list_layout.addWidget(card)

    def _preview_clip(self, path: str):
        """Load clip into the sidebar player."""
        self._playing_path = path
        self._player.load(path)

    def _delete_clip(self, path: str):
        """Confirm deletion, then remove from disk and history."""
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete this clip and its video file?\n\n{os.path.basename(path)}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return

        if self._playing_path == path:
            self._player.stop()
            self._playing_path = None
            
        # 1. Physical file deletion
        if os.path.exists(path):
            try:
                os.remove(path)
                log.info("Deleted physical file: %s", path)
            except Exception as e:
                log.error("Failed to delete physical file: %s", e)

        # 2. History removal
        self.history_manager.remove_clip(path)
        self.refresh_history()

    @staticmethod
    def _open_file(path: str):
        try:
            subprocess.Popen(["xdg-open", path])
        except Exception:
            log.exception("Failed to open file: %s", path)
