from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QSlider, QLabel, QFrame, QSizePolicy
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import Qt, QUrl, QTime
import os

class VideoPlayer(QFrame):
    """A self-contained video player widget with controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VideoPlayerContainer")
        self.setMinimumWidth(300)

        # ── Setup Multimedia ──
        self._player = QMediaPlayer()
        self._video_widget = QVideoWidget()
        self._audio_output = QAudioOutput()
        
        self._player.setVideoOutput(self._video_widget)
        self._player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(1.0)

        # ── UI Layout ──
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Video Area
        self._video_widget.setStyleSheet("background-color: black; border-radius: 10px;")
        layout.addWidget(self._video_widget, 1)

        # Controls Container
        controls = QFrame()
        controls.setObjectName("PlayerControls")
        c_layout = QVBoxLayout(controls)
        c_layout.setContentsMargins(10, 5, 10, 10)

        # Progress Slider
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.sliderMoved.connect(self._set_position)
        c_layout.addWidget(self._slider)

        # Buttons and Time
        btn_row = QHBoxLayout()
        
        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(32, 32)
        self._play_btn.setObjectName("PlayerBtn")
        self._play_btn.clicked.connect(self._toggle_playback)
        
        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setStyleSheet("color: #94a3b8; font-size: 11px;")

        btn_row.addWidget(self._play_btn)
        btn_row.addSpacing(10)
        btn_row.addWidget(self._time_label)
        btn_row.addStretch()
        
        c_layout.addLayout(btn_row)
        layout.addWidget(controls)

        # ── Signals ──
        self._player.positionChanged.connect(self._update_position)
        self._player.durationChanged.connect(self._update_duration)
        self._player.playbackStateChanged.connect(self._update_btn)

    # ══════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════
    def load(self, path: str):
        """Load a video file and prepare for playback."""
        if not os.path.exists(path):
            return
        
        self.stop()
        self._player.setSource(QUrl.fromLocalFile(path))
        self.play()

    def play(self):
        self._player.play()

    def pause(self):
        self._player.pause()

    def stop(self):
        self._player.stop()

    # ══════════════════════════════════════════════════
    #  Internal Handlers
    # ══════════════════════════════════════════════════
    def _toggle_playback(self):
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self.pause()
        else:
            self.play()

    def _update_btn(self, state):
        if state == QMediaPlayer.PlayingState:
            self._play_btn.setText("⏸")
        else:
            self._play_btn.setText("▶")

    def _update_position(self, pos):
        self._slider.setValue(pos)
        self._update_time_label(pos, self._player.duration())

    def _update_duration(self, dur):
        self._slider.setRange(0, dur)
        self._update_time_label(self._player.position(), dur)

    def _set_position(self, pos):
        self._player.setPosition(pos)

    def _update_time_label(self, pos, dur):
        p_time = QTime(0, 0).addMSecs(pos).toString("mm:ss")
        d_time = QTime(0, 0).addMSecs(dur).toString("mm:ss")
        self._time_label.setText(f"{p_time} / {d_time}")
