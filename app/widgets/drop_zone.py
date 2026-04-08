from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QFileDialog
from PySide6.QtCore import Qt, Signal

from logger import get_logger

log = get_logger(__name__)


class DropZone(QFrame):
    """Drag-and-drop zone that emits *file_selected* with the chosen path."""

    file_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setMinimumSize(300, 200)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        self._label = QLabel("Drag & Drop Video Here\nor Click to Browse")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("font-size: 16px; color: #94a3b8;")
        layout.addWidget(self._label)

    # ── Click to browse ──────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "", "Videos (*.mp4 *.mov *.mkv *.avi *.webm)"
            )
            if file_path:
                log.info("File selected via dialog: %s", file_path)
                self.file_selected.emit(file_path)

    # ── Drag & drop ──────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.setStyleSheet("border: 2px dashed #38bdf8; background-color: #1e2944;")
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")

    def dropEvent(self, event):
        self.setStyleSheet("")
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            log.info("File dropped: %s", file_path)
            self.file_selected.emit(file_path)
