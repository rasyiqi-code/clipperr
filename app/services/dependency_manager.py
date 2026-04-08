import sys

from PySide6.QtCore import QObject, Signal, QProcess

from logger import get_logger

log = get_logger(__name__)


class DependencyManager(QObject):
    """Manages system-level Python dependency installation (e.g. PyTorch)."""

    status_signal = Signal(str, bool)
    output_signal = Signal(str)
    finished_signal = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._handle_stdout)
        self._process.readyReadStandardError.connect(self._handle_stderr)
        self._process.finished.connect(self._handle_finished)

    @staticmethod
    def check_torch() -> tuple[bool, str]:
        try:
            import torch
            return True, f"PyTorch {torch.__version__}"
        except ImportError:
            return False, "PyTorch Missing"

    def install_torch(self):
        python_exe = sys.executable
        args = [
            "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ]
        log.info("Starting torch install: %s %s", python_exe, " ".join(args))
        self.output_signal.emit(f"Starting installation: {python_exe} {' '.join(args)}\n")
        self._process.start(python_exe, args)

    # ── Internal handlers ────────────────────────────
    def _handle_stdout(self):
        data = self._process.readAllStandardOutput().data().decode()
        self.output_signal.emit(data)

    def _handle_stderr(self):
        data = self._process.readAllStandardError().data().decode()
        self.output_signal.emit(f"Error: {data}")

    def _handle_finished(self, exit_code, _exit_status):
        success = exit_code == 0
        log.info("Torch install %s (exit code %d)", "succeeded" if success else "failed", exit_code)
        self.finished_signal.emit(success)
        if success:
            self.output_signal.emit("\n✅ Installation complete!")
        else:
            self.output_signal.emit(f"\n❌ Installation failed with exit code {exit_code}")
