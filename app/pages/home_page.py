import os
import shutil
import re
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QProgressBar, 
                               QPushButton, QHBoxLayout, QScrollArea, QFrame, QGridLayout, QSizePolicy)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QIcon

from widgets.drop_zone import DropZone
from services.history_manager import HistoryManager
from logger import get_logger

log = get_logger(__name__)


class HomePage(QScrollArea):
    """Dashboard page with Analytics, Quick Presets, and Recent Projects Gallery."""

    file_selected = Signal(str)
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("background-color: #020617;")

        self.history_manager = HistoryManager()
        
        container = QWidget()
        container.setStyleSheet("background-color: #020617;")
        self.setWidget(container)

        self.layout = QVBoxLayout(container)
        self.layout.setContentsMargins(30, 25, 30, 25)
        self.layout.setSpacing(20)

        # ── Drop Zone (Now smaller height) ──
        dz_container = QWidget()
        dz_layout = QVBoxLayout(dz_container)
        dz_layout.setContentsMargins(0, 0, 0, 0)
        self.drop_zone = DropZone()
        self.drop_zone.setFixedHeight(180)  # Make it shorter
        self.drop_zone.file_selected.connect(self.file_selected.emit)
        dz_layout.addWidget(self.drop_zone)
        self.layout.addWidget(dz_container)

        # ── Processing Status Row ──
        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #38bdf8; font-weight: 600;")
        status_row.addWidget(self.status_label, 1)

        self.cancel_btn = QPushButton("⏹ Cancel")
        self.cancel_btn.setObjectName("ActionButton")
        self.cancel_btn.setFixedWidth(100)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        status_row.addWidget(self.cancel_btn)
        self.layout.addLayout(status_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)  # Hide overlapping text
        self.layout.addWidget(self.progress_bar)

        # ── Dynamic content starts after this point ──
        # Track where dynamic widgets begin so we can refresh them
        self._dynamic_start_index = self.layout.count()

        self._build_analytics_section()
        self._build_gallery_section()

        self.layout.addStretch()

    def _refresh_dashboard(self):
        """Rebuild analytics and gallery sections with fresh data."""
        # Remove all widgets/layouts after the dynamic start index
        while self.layout.count() > self._dynamic_start_index:
            item = self.layout.takeAt(self._dynamic_start_index)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # Recursively delete child widgets in sub-layouts
                self._clear_layout(item.layout())

        self._build_analytics_section()
        self._build_gallery_section()
        self.layout.addStretch()

    @staticmethod
    def _clear_layout(layout):
        """Recursively remove all items from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                HomePage._clear_layout(item.layout())

    def _build_analytics_section(self):
        title = QLabel("System Analytics")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #38bdf8; margin-top: 15px;")
        self.layout.addWidget(title)

        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)

        # Get stats
        groups = self._get_project_groups()
        total_projects = len(groups)
        total_clips = len(self.history_manager.clips)
        
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)

        # Cards
        stats_layout.addWidget(self._stat_card("🎬 Total Projects", str(total_projects)))
        stats_layout.addWidget(self._stat_card("✂️ Clips Produced", str(total_clips)))
        stats_layout.addWidget(self._stat_card("💾 Free Space", f"{free_gb} GB"))

        self.layout.addLayout(stats_layout)

    def _stat_card(self, title: str, value: str) -> QFrame:
        card = QFrame()
        card.setObjectName("ClipCard")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 15, 20, 15)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;")
        lbl_title.setAlignment(Qt.AlignCenter)
        
        lbl_val = QLabel(value)
        lbl_val.setStyleSheet("color: #38bdf8; font-size: 26px; font-weight: 900;")
        lbl_val.setAlignment(Qt.AlignCenter)
        
        cl.addWidget(lbl_title)
        cl.addWidget(lbl_val)
        return card

    def _build_gallery_section(self):
        title = QLabel("Recent Projects")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #38bdf8; margin-top: 15px;")
        self.layout.addWidget(title)

        groups = self._get_project_groups()
        if not groups:
            info = QLabel("Process a video to build your gallery.")
            info.setStyleSheet("color: #64748b; font-style: italic;")
            self.layout.addWidget(info)
            return

        grid = QGridLayout()
        grid.setSpacing(15)

        row, col = 0, 0
        for proj_name, clips in list(groups.items())[:6]:  # Show max 6 recent
            card = self._gallery_card(proj_name, clips)
            grid.addWidget(card, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1

        self.layout.addLayout(grid)

    def _gallery_card(self, proj_name: str, clips: list) -> QFrame:
        card = QFrame()
        card.setObjectName("ClipCard")
        card.setFixedSize(280, 240)
        
        cl = QVBoxLayout(card)
        cl.setContentsMargins(15, 15, 15, 15)
        cl.setSpacing(12)
        
        # Determine thumbnail
        thumb_path = None
        for clip in clips:
            if "thumbnail_path" in clip and os.path.exists(clip["thumbnail_path"]):
                thumb_path = clip["thumbnail_path"]
                break
        
        # Image area
        img_lbl = QLabel()
        img_lbl.setAlignment(Qt.AlignCenter)
        img_lbl.setStyleSheet("background-color: #030712; border-radius: 12px; border: 1px solid #1e293b;")
        img_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        if thumb_path:
            pixmap = QPixmap(thumb_path)
            scaled = pixmap.scaled(250, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_lbl.setPixmap(scaled)
        else:
            img_lbl.setText("No Thumbnail")
            img_lbl.setStyleSheet("color: #475569; background-color: #030712; border-radius: 12px; font-weight: 800; font-size: 10px; text-transform: uppercase;")
            
        cl.addWidget(img_lbl)
        
        # Info row
        name_lbl = QLabel(proj_name[:25] + "..." if len(proj_name) > 25 else proj_name)
        name_lbl.setStyleSheet("color: #f8fafc; font-weight: 800; font-size: 14px;")
        name_lbl.setToolTip(proj_name)
        cl.addWidget(name_lbl)
        
        info_row = QHBoxLayout()
        info_row.setSpacing(8)
        
        badge = QLabel(f"{len(clips)} SHORTS")
        badge.setStyleSheet("color: #38bdf8; font-weight: 900; font-size: 10px; background-color: rgba(56, 189, 248, 0.1); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.2);")
        info_row.addWidget(badge)
        
        info_row.addStretch()
        
        time_badge = QLabel("READY")
        time_badge.setStyleSheet("color: #10b981; font-weight: 900; font-size: 9px; letter-spacing: 0.5px;")
        info_row.addWidget(time_badge)
        
        cl.addLayout(info_row)
        return card

    def _get_project_groups(self) -> dict:
        groups = {}
        for clip in self.history_manager.clips:
            out_path = clip.get("output_path", "")
            base = os.path.basename(out_path)
            # Match pattern: clip_0_OriginalName.mp4
            match = re.match(r"clip_\d+_(.*)", base)
            if match:
                proj_name = match.group(1)
            else:
                proj_name = base

            if proj_name not in groups:
                groups[proj_name] = []
            groups[proj_name].append(clip)
        return groups

    # ── Public helpers ───────────────────────────────
    def set_status(self, message: str):
        self.status_label.setText(message)

    def set_progress(self, value: int):
        self.progress_bar.setValue(value)

    def set_processing(self, active: bool):
        """Toggle UI between idle and processing states."""
        self.cancel_btn.setVisible(active)
        self.drop_zone.setEnabled(not active)
        if not active:
            # Refresh gallery when processing finishes
            self.history_manager.clips = self.history_manager.load()
            self._refresh_dashboard()
