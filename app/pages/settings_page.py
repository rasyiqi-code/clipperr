import os

from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QPushButton, QLineEdit,
)
from PySide6.QtCore import QThread, Qt

from services.downloader import ModelManager, DownloadWorker, DownloadThread
from services.dependency_manager import DependencyManager
from logger import get_logger

log = get_logger(__name__)


class SettingsPage(QScrollArea):
    """System configuration page: dependency installer + model manager."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("background-color: #020617;")

        self.model_manager = ModelManager()
        self.dep_manager = DependencyManager()
        self._active_downloads: list = []
        self._model_ui: dict = {}  # model_id -> (status_label, btn)

        # Connect dep_manager signals ONCE (thread-safety fix)
        self.dep_manager.output_signal.connect(self._on_dep_log)
        self.dep_manager.finished_signal.connect(self._on_dep_install_finished)

        page = QWidget()
        page.setStyleSheet("background-color: #020617;")
        self.setWidget(page)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(15)

        title = QLabel("System Configuration")
        title.setStyleSheet("font-size: 28px; font-weight: 800;")
        layout.addWidget(title)

        # ── System Environment ───────────────────────
        try:
            self._build_env_section(layout)
        except Exception as e:
            layout.addWidget(QLabel(f"❌ Error loading Environment: {e}"))

        # ── Render Preferences ─────────────────────────
        try:
            self._build_render_prefs_section(layout)
        except Exception as e:
            layout.addWidget(QLabel(f"❌ Error loading Render Prefs: {e}"))

        # ── AI Engine Configuration ───────────────────
        try:
            self._build_ai_config_section(layout)
        except Exception as e:
            layout.addWidget(QLabel(f"❌ Error loading AI Config: {e}"))

        # ── LLM Analysis Parameters ──────────────────
        try:
            self._build_llm_analysis_section(layout)
        except Exception as e:
            layout.addWidget(QLabel(f"❌ Error loading LLM Analysis Params: {e}"))

        # ── AI Model Manager ─────────────────────────
        try:
            self._build_model_section(layout)
        except Exception as e:
            layout.addWidget(QLabel(f"❌ Error loading Model Manager: {e}"))

        layout.addStretch()

    # ══════════════════════════════════════════════════
    #  Build helpers
    # ══════════════════════════════════════════════════
    def _build_render_prefs_section(self, layout: QVBoxLayout):
        from config import prefs
        from PySide6.QtWidgets import (QCheckBox, QFileDialog, QRadioButton, QButtonGroup, 
                                       QSlider, QComboBox, QStackedWidget, QWidget)
        from PySide6.QtCore import Qt

        pref_title = QLabel("Render Preferences")
        pref_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #38bdf8; margin-top: 20px;")
        layout.addWidget(pref_title)

        pref_card = QFrame()
        pref_card.setObjectName("ClipCard")
        pc_layout = QVBoxLayout(pref_card)

        # Basic Checkbox
        self.thumb_cb = QCheckBox("Generate AI Thumbnails (Clickbait Cover)")
        self.thumb_cb.setStyleSheet("""
            QCheckBox { color: #e2e8f0; font-weight: bold; font-size: 13px; }
            QCheckBox::indicator { width: 18px; height: 18px; background-color: #1e293b; border: 1px solid #334155; border-radius: 4px; }
            QCheckBox::indicator:checked { background-color: #38bdf8; border: 1px solid #0ea5e9; image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%230f172a' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='20 6 9 17 4 12'%3E%3C/polyline%3E%3C/svg%3E"); }
        """)
        self.thumb_cb.setChecked(prefs.auto_thumbnail)
        self.thumb_cb.stateChanged.connect(self._on_thumb_toggled)
        pc_layout.addWidget(self.thumb_cb)

        # Advanced Watermark UI
        wm_title = QLabel("Watermark settings:")
        wm_title.setStyleSheet("color: #94a3b8; font-size: 12px; margin-top: 15px;")
        pc_layout.addWidget(wm_title)

        # Type Selection
        type_layout = QHBoxLayout()
        self.btn_group_type = QButtonGroup(self)
        self.rb_image = QRadioButton("Image")
        self.rb_text = QRadioButton("Text")
        for rb in [self.rb_image, self.rb_text]:
            rb.setStyleSheet("""
                QRadioButton { color: #cbd5e1; font-size: 12px; }
                QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; background-color: #1e293b; border: 1px solid #334155; }
                QRadioButton::indicator:checked { background-color: #38bdf8; border: 3px solid #0f172a; }
            """)
            self.btn_group_type.addButton(rb)
        
        self.rb_image.setChecked(prefs.watermark_type == "image")
        self.rb_text.setChecked(prefs.watermark_type == "text")
        self.rb_image.toggled.connect(self._on_wm_type_changed)
        type_layout.addWidget(self.rb_image)
        type_layout.addWidget(self.rb_text)
        type_layout.addStretch()
        pc_layout.addLayout(type_layout)

        # Stacked Widget for Type Inputs
        self.wm_stack = QStackedWidget()
        
        # --- Image Page ---
        self.page_image = QWidget()
        pimg_layout = QHBoxLayout(self.page_image)
        pimg_layout.setContentsMargins(0, 5, 0, 5)
        self.wm_label = QLabel(f"Selected: {os.path.basename(prefs.watermark_path) if prefs.watermark_path else 'None'}")
        self.wm_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        btn_wm = QPushButton("Browse (.png)")
        btn_wm.setStyleSheet("background-color: #1e293b; color: white; padding: 5px; border-radius: 4px;")
        btn_wm.clicked.connect(self._select_watermark)
        btn_clear_wm = QPushButton("Clear")
        btn_clear_wm.setStyleSheet("background-color: #334155; color: white; border-radius: 4px;")
        btn_clear_wm.clicked.connect(self._clear_watermark)
        pimg_layout.addWidget(self.wm_label)
        pimg_layout.addStretch()
        pimg_layout.addWidget(btn_wm)
        pimg_layout.addWidget(btn_clear_wm)
        self.wm_stack.addWidget(self.page_image)

        # --- Text Page ---
        self.page_text = QWidget()
        ptxt_layout = QHBoxLayout(self.page_text)
        ptxt_layout.setContentsMargins(0, 5, 0, 5)
        self.txt_input = QLineEdit()
        self.txt_input.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 4px; padding: 5px; color: white;")
        self.txt_input.setPlaceholderText("Enter watermark text (e.g. @MyChannel)")
        self.txt_input.setText(prefs.watermark_text)
        self.txt_input.textChanged.connect(self._on_wm_text_changed)
        ptxt_layout.addWidget(self.txt_input)
        self.wm_stack.addWidget(self.page_text)

        pc_layout.addWidget(self.wm_stack)
        self.wm_stack.setCurrentWidget(self.page_image if prefs.watermark_type == "image" else self.page_text)

        # Position and Opacity Options
        opt_layout = QHBoxLayout()
        pos_label = QLabel("Pos:")
        pos_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.cb_pos = QComboBox()
        self.cb_pos.addItems(["Top Left", "Top Right", "Bottom Left", "Bottom Right", "Center"])
        self.cb_pos.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 4px;")
        
        pos_map_inv = {"top_left": 0, "top_right": 1, "bottom_left": 2, "bottom_right": 3, "center": 4}
        self.cb_pos.setCurrentIndex(pos_map_inv.get(prefs.watermark_pos, 0))
        self.cb_pos.currentIndexChanged.connect(self._on_wm_pos_changed)
        
        opc_label = QLabel("Opacity:")
        opc_label.setStyleSheet("color: #94a3b8; font-size: 12px; margin-left: 20px;")
        self.slider_opc = QSlider(Qt.Horizontal)
        self.slider_opc.setRange(10, 100)
        self.slider_opc.setValue(int(prefs.watermark_opacity * 100))
        self.slider_opc.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #334155; height: 4px; background: #1e293b; margin: 2px 0; border-radius: 2px; }
            QSlider::handle:horizontal { background: #38bdf8; border: 1px solid #0ea5e9; width: 14px; margin: -5px 0; border-radius: 7px; }
        """)
        self.slider_opc.valueChanged.connect(self._on_wm_opc_changed)
        
        self.lbl_opc_val = QLabel(f"{int(prefs.watermark_opacity * 100)}%")
        self.lbl_opc_val.setStyleSheet("color: #38bdf8; font-size: 12px; font-weight: bold;")
        
        opt_layout.addWidget(pos_label)
        opt_layout.addWidget(self.cb_pos)
        opt_layout.addWidget(opc_label)
        opt_layout.addWidget(self.slider_opc)
        opt_layout.addWidget(self.lbl_opc_val)
        
        pc_layout.addLayout(opt_layout)
        layout.addWidget(pref_card)

    def _on_thumb_toggled(self, state):
        from config import prefs
        prefs.auto_thumbnail = (state == 2)
        prefs.save()

    def _on_wm_type_changed(self):
        from config import prefs
        prefs.watermark_type = "image" if self.rb_image.isChecked() else "text"
        prefs.save()
        self.wm_stack.setCurrentWidget(self.page_image if prefs.watermark_type == "image" else self.page_text)

    def _on_wm_text_changed(self, text):
        from config import prefs
        prefs.watermark_text = text
        prefs.save()

    def _on_wm_pos_changed(self, idx):
        from config import prefs
        pos_map = {0: "top_left", 1: "top_right", 2: "bottom_left", 3: "bottom_right", 4: "center"}
        prefs.watermark_pos = pos_map.get(idx, "top_left")
        prefs.save()

    def _on_wm_opc_changed(self, val):
        from config import prefs
        prefs.watermark_opacity = val / 100.0
        prefs.save()
        self.lbl_opc_val.setText(f"{val}%")

    def _select_watermark(self):
        from config import prefs
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Select Watermark", "", "Images (*.png)")
        if path:
            prefs.watermark_path = path
            prefs.save()
            self.wm_label.setText(f"Selected: {os.path.basename(path)}")

    def _clear_watermark(self):
        from config import prefs
        prefs.watermark_path = ""
        prefs.save()
        self.wm_label.setText("Selected: None")

    def _build_ai_config_section(self, layout: QVBoxLayout):
        from config import prefs
        from PySide6.QtWidgets import QComboBox, QLineEdit
        
        ai_title = QLabel("AI Engine Configuration")
        ai_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #38bdf8; margin-top: 20px;")
        layout.addWidget(ai_title)

        ai_card = QFrame()
        ai_card.setObjectName("ClipCard")
        ac_layout = QVBoxLayout(ai_card)

        # Provider Selection
        prov_layout = QHBoxLayout()
        prov_label = QLabel("AI Provider:")
        prov_label.setStyleSheet("color: #e2e8f0; font-weight: bold; font-size: 13px;")
        self.cb_provider = QComboBox()
        self.cb_provider.addItems(["Local LLM (Qwen)", "OpenRouter API (Cloud)"])
        self.cb_provider.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 4px; padding: 5px;")
        self.cb_provider.setCurrentIndex(0 if prefs.llm_provider == "local" else 1)
        self.cb_provider.setEnabled(False) # Locked by default
        self.cb_provider.currentIndexChanged.connect(self._on_ai_provider_changed)
        
        self.btn_unlock_prov = QPushButton("🔓 Unlock")
        self.btn_unlock_prov.setFixedSize(80, 32)
        self.btn_unlock_prov.setCursor(Qt.PointingHandCursor)
        self.btn_unlock_prov.setStyleSheet("""
            QPushButton { background-color: #1e293b; color: #38bdf8; border: 1px solid #334155; border-radius: 6px; font-size: 11px; font-weight: bold; }
            QPushButton:hover { background-color: #334155; color: #7dd3fc; }
        """)
        self.btn_unlock_prov.clicked.connect(self._toggle_provider_lock)

        prov_layout.addWidget(prov_label)
        prov_layout.addWidget(self.cb_provider, 1)
        prov_layout.addWidget(self.btn_unlock_prov)
        ac_layout.addLayout(prov_layout)

        # Container for API specific settings
        self.api_settings_container = QWidget()
        api_layout = QVBoxLayout(self.api_settings_container)
        api_layout.setContentsMargins(0, 10, 0, 0)
        
        # API Key
        key_label = QLabel("OpenRouter API Key:")
        key_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("sk-or-v1-...")
        self.api_key_input.setText(prefs.openrouter_key)
        self.api_key_input.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 4px; padding: 8px; color: white;")
        self.api_key_input.textChanged.connect(self._on_api_key_changed)
        api_layout.addWidget(key_label)
        api_layout.addWidget(self.api_key_input)

        # API Model
        model_label = QLabel("API Model Name:")
        model_label.setStyleSheet("color: #94a3b8; font-size: 12px; margin-top: 5px;")
        self.api_model_input = QLineEdit()
        self.api_model_input.setPlaceholderText("google/gemini-2.0-flash-lite:free")
        self.api_model_input.setText(prefs.openrouter_model)
        self.api_model_input.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 4px; padding: 8px; color: white;")
        self.api_model_input.textChanged.connect(self._on_api_model_changed)
        api_layout.addWidget(model_label)
        api_layout.addWidget(self.api_model_input)

        ac_layout.addWidget(self.api_settings_container)
        self.api_settings_container.setVisible(prefs.llm_provider == "api")
        
        layout.addWidget(ai_card)

    def _build_llm_analysis_section(self, layout: QVBoxLayout):
        from config import prefs
        from PySide6.QtWidgets import QSlider, QLineEdit
        
        llm_title = QLabel("LLM Analysis Parameters")
        llm_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #38bdf8; margin-top: 20px;")
        layout.addWidget(llm_title)

        llm_card = QFrame()
        llm_card.setObjectName("ClipCard")
        lc_layout = QVBoxLayout(llm_card)

        # Max Clips
        mc_layout = QHBoxLayout()
        mc_label = QLabel("Max Clips per Video:")
        mc_label.setStyleSheet("color: #e2e8f0; font-size: 13px;")
        self.max_clips_input = QLineEdit()
        self.max_clips_input.setFixedWidth(60)
        self.max_clips_input.setText(str(prefs.max_viral_clips))
        self.max_clips_input.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 4px; padding: 4px; color: white;")
        self.max_clips_input.textChanged.connect(self._on_max_clips_changed)
        mc_layout.addWidget(mc_label)
        mc_layout.addStretch()
        mc_layout.addWidget(self.max_clips_input)
        lc_layout.addLayout(mc_layout)

        # Temperature
        temp_layout = QHBoxLayout()
        temp_label = QLabel("AI Creativity (Temperature):")
        temp_label.setStyleSheet("color: #e2e8f0; font-size: 13px;")
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(1, 15) # 0.1 to 1.5
        self.temp_slider.setValue(int(prefs.llm_temperature * 10))
        self.temp_slider.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #334155; height: 4px; background: #1e293b; border-radius: 2px; }
            QSlider::handle:horizontal { background: #38bdf8; border: 1px solid #0ea5e9; width: 14px; margin: -5px 0; border-radius: 7px; }
        """)
        self.temp_val_label = QLabel(f"{prefs.llm_temperature:.1f}")
        self.temp_val_label.setStyleSheet("color: #38bdf8; font-weight: bold; width: 30px;")
        self.temp_slider.valueChanged.connect(self._on_temp_changed)
        temp_layout.addWidget(temp_label)
        temp_layout.addStretch()
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_val_label)
        lc_layout.addLayout(temp_layout)

        # Max Tokens
        tok_layout = QHBoxLayout()
        tok_label = QLabel("Max Tokens (Response Length):")
        tok_label.setStyleSheet("color: #e2e8f0; font-size: 13px;")
        self.tokens_input = QLineEdit()
        self.tokens_input.setFixedWidth(80)
        self.tokens_input.setText(str(prefs.llm_max_tokens))
        self.tokens_input.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 4px; padding: 4px; color: white;")
        self.tokens_input.textChanged.connect(self._on_tokens_changed)
        tok_layout.addWidget(tok_label)
        tok_layout.addStretch()
        tok_layout.addWidget(self.tokens_input)
        lc_layout.addLayout(tok_layout)

        layout.addWidget(llm_card)

    def _on_max_clips_changed(self, text):
        from config import prefs
        try:
            prefs.max_viral_clips = int(text)
            prefs.save()
        except ValueError: pass

    def _on_temp_changed(self, val):
        from config import prefs
        temp = val / 10.0
        prefs.llm_temperature = temp
        self.temp_val_label.setText(f"{temp:.1f}")
        prefs.save()

    def _on_tokens_changed(self, text):
        from config import prefs
        try:
            prefs.llm_max_tokens = int(text)
            prefs.save()
        except ValueError: pass

    def _on_ai_provider_changed(self, idx):
        from config import prefs
        prefs.llm_provider = "local" if idx == 0 else "api"
        prefs.save()
        self.api_settings_container.setVisible(prefs.llm_provider == "api")
        # Auto-relock after change
        self.cb_provider.setEnabled(False)
        self.btn_unlock_prov.setText("🔓 Unlock")

    def _toggle_provider_lock(self):
        is_locked = not self.cb_provider.isEnabled()
        self.cb_provider.setEnabled(is_locked)
        self.btn_unlock_prov.setText("🔒 Lock" if is_locked else "🔓 Unlock")

    def _on_api_key_changed(self, text):
        from config import prefs
        prefs.openrouter_key = text
        prefs.save()

    def _on_api_model_changed(self, text):
        from config import prefs
        prefs.openrouter_model = text
        prefs.save()

    def _build_env_section(self, layout: QVBoxLayout):
        env_title = QLabel("System Environment")
        env_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #38bdf8; margin-top: 20px;")
        layout.addWidget(env_title)

        env_card = QFrame()
        env_card.setObjectName("ClipCard")
        ec_layout = QHBoxLayout(env_card)

        v_info = QVBoxLayout()
        name = QLabel("PYTORCH ENGINE")
        name.setStyleSheet("font-weight: 800; font-size: 14px;")
        repo = QLabel("Essential for Whisper & AI Inference")
        repo.setStyleSheet("color: #64748b; font-size: 11px;")
        v_info.addWidget(name)
        v_info.addWidget(repo)
        ec_layout.addLayout(v_info)
        ec_layout.addStretch()

        self._torch_status = QLabel("Checking...")
        self._install_btn = QPushButton("Install Tools")
        self._install_btn.setObjectName("ActionButton")
        self._install_btn.setFixedWidth(120)
        self._install_btn.clicked.connect(self._start_dependency_install)

        ec_layout.addWidget(self._torch_status)
        ec_layout.addWidget(self._install_btn)
        layout.addWidget(env_card)

        self._refresh_torch_ui()

    def _build_model_section(self, layout: QVBoxLayout):
        from config import prefs
        desc = QLabel("Manage your local AI models and application preferences.")
        desc.setStyleSheet("color: #94a3b8; margin-bottom: 20px;")
        layout.addWidget(desc)

        section_title = QLabel("AI Model Manager")
        section_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #38bdf8;")
        layout.addWidget(section_title)

        # Informational note about models
        info_label = QLabel("Note: AI models will be downloaded automatically from public repositories (HuggingFace/GitHub).")
        info_label.setStyleSheet("color: #64748b; font-size: 11px; font-style: italic; margin-top: 10px;")
        layout.addWidget(info_label)

        # Model cards
        for model_id, info in self.model_manager.models.items():
            card = QFrame()
            card.setObjectName("ClipCard")
            c_layout = QHBoxLayout(card)

            v_info = QVBoxLayout()
            name = QLabel(model_id.upper().replace("-", " "))
            name.setStyleSheet("font-weight: 800; font-size: 14px;")
            repo_label = QLabel(info.get("repo", info.get("url", "")))
            repo_label.setStyleSheet("color: #64748b; font-size: 11px;")
            v_info.addWidget(name)
            v_info.addWidget(repo_label)

            c_layout.addLayout(v_info)
            c_layout.addStretch()

            status = QLabel("Checking...")
            btn = QPushButton("Install")
            btn.setObjectName("ActionButton")
            btn.setFixedWidth(120)

            c_layout.addWidget(status)
            c_layout.addWidget(btn)
            layout.addWidget(card)

            self._model_ui[model_id] = (status, btn)
            btn.clicked.connect(
                lambda _checked, m=model_id: self._start_download(m)
            )
            self._refresh_model_ui(model_id)


    # ══════════════════════════════════════════════════
    #  Torch dependency
    # ══════════════════════════════════════════════════
    def _refresh_torch_ui(self):
        ready, _msg = self.dep_manager.check_torch()
        self._torch_status.setText("✅ Ready" if ready else "❌ Missing")
        self._torch_status.setStyleSheet(
            "color: #10b981; font-weight: bold;" if ready else "color: #ef4444; font-weight: bold;"
        )
        self._install_btn.setText("Reinstall" if ready else "Install Tools")
        self._install_btn.setStyleSheet(
            "background-color: #1e293b; color: #94a3b8;" if ready else ""
        )
        self._install_btn.setEnabled(True)

    def _start_dependency_install(self):
        self._install_btn.setEnabled(False)
        self._torch_status.setText("📥 Starting...")
        self.dep_manager.install_torch()

    def _on_dep_log(self, data: str):
        if "Collecting" in data:
            lib = data.split("Collecting")[1].split("\n")[0].strip()
            self._torch_status.setText(f"📥 Installing {lib}...")

    def _on_dep_install_finished(self, success: bool):
        self._refresh_torch_ui()
        if not success:
            self._install_btn.setEnabled(True)

    # ══════════════════════════════════════════════════
    #  Model downloads
    # ══════════════════════════════════════════════════
    def _refresh_model_ui(self, model_id: str):
        status_label, btn = self._model_ui[model_id]
        exists = self.model_manager.check_status(model_id)
        status_label.setText("✅ Ready" if exists else "⚠️ Missing")
        status_label.setStyleSheet(
            "color: #10b981; font-weight: bold;" if exists else "color: #f59e0b; font-weight: bold;"
        )
        btn.setText("Reinstall" if exists else "Download")
        btn.setStyleSheet("background-color: #1e293b; color: #94a3b8;" if exists else "")
        btn.setEnabled(True)

    def _start_download(self, model_id: str):
        status_label, btn = self._model_ui[model_id]
        info = self.model_manager.models[model_id]
        token = None # All downloads are now anonymous

        btn.setEnabled(False)
        status_label.setText("📥 Starting...")

        if "url" in info:
            thread = QThread()
            worker = DownloadWorker()
            worker.moveToThread(thread)
            # Store worker reference on thread to prevent GC collection during download
            thread._worker = worker
            worker.progress_signal.connect(lambda msg, pct: status_label.setText(f"📥 [{pct}%] {msg}"))
            worker.finished_signal.connect(
                lambda _f, success: self._download_finished(model_id, success, thread)
            )
            thread.started.connect(lambda: worker.download_url(info["url"], info["files"][0], info["path"]))
            self._active_downloads.append(thread)
            thread.start()
        else:
            thread = DownloadThread(info["repo"], info["files"], info["path"], token)
            self._active_downloads.append(thread)
            thread.progress_signal.connect(lambda msg, pct: status_label.setText(f"📥 [{pct}%] {msg}"))
            thread.finished_signal.connect(
                lambda _f, success: self._download_finished(model_id, success, thread)
            )
            thread.start()

    def _download_finished(self, model_id: str, success: bool, thread: QThread):
        if thread in self._active_downloads:
            self._active_downloads.remove(thread)
        thread.quit()
        thread.wait()

        self._refresh_model_ui(model_id)
        if not success:
            status_label, _btn = self._model_ui[model_id]
            # Don't overwrite if it already contains a specific error message
            current_text = status_label.text()
            if "Error:" not in current_text:
                status_label.setText("❌ Download Failed")
            status_label.setStyleSheet("color: #ef4444; font-weight: bold;")

    # ── Public helpers ───────────────────────────────
    def has_all_models(self) -> bool:
        return all(self.model_manager.check_status(m) for m in self.model_manager.models)

    def missing_models(self) -> list[str]:
        return [m for m in self.model_manager.models if not self.model_manager.check_status(m)]
