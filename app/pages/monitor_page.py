import os
import psutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QFrame, QProgressBar, QScrollArea
)
from PySide6.QtCore import QTimer, Qt
from logger import get_logger

log = get_logger(__name__)

class MonitorPage(QScrollArea):
    """Real-time system telemetry and hardware monitoring page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("background-color: #020617;")

        page = QWidget()
        page.setStyleSheet("background-color: #020617;")
        self.setWidget(page)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(20)

        # Title
        title = QLabel("System Monitor")
        title.setStyleSheet("font-size: 28px; font-weight: 800; color: white;")
        layout.addWidget(title)

        subtitle = QLabel("Real-time hardware telemetry and process statistics.")
        subtitle.setStyleSheet("color: #94a3b8; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(subtitle)

        # ── Hardware Cards Grid ───────────────────────
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(15)

        self.cpu_card = self._create_monitor_card("CPU USAGE", "processor")
        self.ram_card = self._create_monitor_card("MEMORY (RAM)", "memory")
        self.gpu_card = self._create_monitor_card("GPU (VRAM)", "gpu")

        grid_layout.addWidget(self.cpu_card)
        grid_layout.addWidget(self.ram_card)
        grid_layout.addWidget(self.gpu_card)
        layout.addLayout(grid_layout)

        # ── Process Details ────────────────────────────
        proc_title = QLabel("Process Information")
        proc_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #38bdf8; margin-top: 10px;")
        layout.addWidget(proc_title)

        self.proc_card = QFrame()
        self.proc_card.setObjectName("ClipCard")
        proc_layout = QVBoxLayout(self.proc_card)
        proc_layout.setContentsMargins(20, 20, 20, 20)
        
        self.proc_details = QLabel("Initializing telemetry...")
        self.proc_details.setWordWrap(True)
        self.proc_details.setStyleSheet("color: #94a3b8; font-family: 'JetBrains Mono', monospace; font-size: 13px; line-height: 1.8;")
        proc_layout.addWidget(self.proc_details)
        
        layout.addWidget(self.proc_card)
        layout.addStretch()

        # ── Refresh Timer ──────────────────────────────
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_stats)
        self.timer.start(2000) # Refresh every 2 seconds

        # Initial update
        self._update_stats()

    def _create_monitor_card(self, title_text, card_type):
        card = QFrame()
        card.setObjectName("ClipCard") # Standardize to ClipCard
        card.setMinimumHeight(180)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)

        header = QLabel(title_text)
        header.setStyleSheet("color: #94a3b8; font-weight: 800; font-size: 11px; letter-spacing: 1px;")
        layout.addWidget(header)

        val_label = QLabel("0%")
        val_label.setObjectName("ValueLabel")
        val_label.setStyleSheet("font-size: 32px; font-weight: 800; color: white; margin: 5px 0;")
        layout.addWidget(val_label)

        sub_val = QLabel("0.0 / 0.0 GB")
        sub_val.setObjectName("SubValueLabel")
        sub_val.setStyleSheet("color: #64748b; font-size: 12px;")
        layout.addWidget(sub_val)

        pbar = QProgressBar()
        pbar.setObjectName("MonitorBar")
        pbar.setFixedHeight(6)
        pbar.setTextVisible(False)
        pbar.setStyleSheet("""
            QProgressBar#MonitorBar {
                background-color: #1e293b;
                border-radius: 3px;
                border: none;
            }
            QProgressBar#MonitorBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38bdf8, stop:1 #818cf8);
                border-radius: 3px;
            }
        """)
        layout.addStretch()
        layout.addWidget(pbar)

        # Store references
        card.val_label = val_label
        card.sub_val = sub_val
        card.pbar = pbar
        
        return card

    def _update_stats(self):
        try:
            # 1. CPU
            cpu_pct = psutil.cpu_percent()
            self.cpu_card.val_label.setText(f"{cpu_pct}%")
            self.cpu_card.pbar.setValue(int(cpu_pct))
            self.cpu_card.sub_val.setText(f"{psutil.cpu_count()} Logical Cores")

            # 2. RAM
            mem = psutil.virtual_memory()
            mem_used_gb = mem.used / (1024**3)
            mem_total_gb = mem.total / (1024**3)
            self.ram_card.val_label.setText(f"{mem.percent}%")
            self.ram_card.pbar.setValue(int(mem.percent))
            self.ram_card.sub_val.setText(f"{mem_used_gb:.1f} GB / {mem_total_gb:.1f} GB")

            # 3. GPU (lazy import — torch may not be installed yet)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    vram_used = torch.cuda.memory_allocated(0) / (1024**3)
                    vram_pct = (vram_used / vram_total) * 100 if vram_total > 0 else 0
                    
                    self.gpu_card.val_label.setText(f"{vram_pct:.1f}%")
                    self.gpu_card.pbar.setValue(int(vram_pct))
                    self.gpu_card.sub_val.setText(f"{vram_used:.2f} / {vram_total:.1f} GB ({gpu_name[:15]}...)")
                else:
                    self.gpu_card.val_label.setText("N/A")
                    self.gpu_card.pbar.setValue(0)
                    self.gpu_card.sub_val.setText("No CUDA Device Found")
            except ImportError:
                self.gpu_card.val_label.setText("N/A")
                self.gpu_card.pbar.setValue(0)
                self.gpu_card.sub_val.setText("PyTorch not installed")

            # ── Process Details ─────────────────────────────
            proc = psutil.Process(os.getpid())
            with proc.oneshot():
                p_mem = proc.memory_info().rss / (1024**2) # MB
                p_cpu = proc.cpu_percent()
                p_threads = proc.num_threads()
                
            details = (
                f"• PID: {os.getpid()}\n"
                f"• Process Memory: {p_mem:.1f} MB\n"
                f"• Process CPU: {p_cpu:.1f}%\n"
                f"• Active Threads: {p_threads}\n"
                f"• Working Directory: {os.getcwd()}"
            )
            self.proc_details.setText(details)

        except Exception as e:
            log.error("Telemetry update failed: %s", e)
