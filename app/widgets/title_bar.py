from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt

from logger import get_logger

log = get_logger(__name__)


class TitleBar(QFrame):
    """Custom window title bar with minimize / maximize / close buttons."""

    def __init__(self, parent):
        super().__init__(parent)
        self._window = parent
        self.setObjectName("TitleBar")
        self.setFixedHeight(40)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 5, 0)

        logo = QLabel("🌀 ClipperR")
        logo.setStyleSheet("font-weight: 900; color: #38bdf8; font-size: 14px;")
        layout.addWidget(logo)
        layout.addStretch()

        self.min_btn = QPushButton("—")
        self.max_btn = QPushButton("🗖")
        self.close_btn = QPushButton("✕")

        for btn in [self.min_btn, self.max_btn, self.close_btn]:
            btn.setObjectName("WindowBtn")
            btn.setFixedSize(30, 30)
            layout.addWidget(btn)

        self.close_btn.setObjectName("CloseBtn")

        self.min_btn.clicked.connect(self._window.showMinimized)
        self.max_btn.clicked.connect(self.toggle_max)
        self.close_btn.clicked.connect(self._window.close)

        self._start_pos = None
        self._is_maximized = False
        self._normal_geometry = None

    # ── Maximize / Restore ───────────────────────────
    def toggle_max(self):
        if self._is_maximized:
            self._window.setGeometry(self._normal_geometry)
            self.max_btn.setText("🗖")
            self._is_maximized = False
        else:
            self._normal_geometry = self._window.geometry()
            screen = self._window.screen().availableGeometry()
            self._window.move(screen.x(), screen.y())
            self._window.resize(screen.width(), screen.height())
            self.max_btn.setText("🗗")
            self._is_maximized = True

    # ── Drag / Double-click ──────────────────────────
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_max()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._start_pos = event.globalPosition().toPoint() - self._window.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._start_pos:
            self._window.move(event.globalPosition().toPoint() - self._start_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._start_pos = None
        event.accept()
