MAIN_STYLE = """
/* Global Theme — Deep Space / Glassmorphism */
QMainWindow {
    background-color: #030712;
}

QWidget {
    color: #f1f5f9;
    font-family: 'Outfit', 'Inter', 'Segoe UI', sans-serif;
}

/* Sidebar — Frosted Glass effect */
#Sidebar {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
    min-width: 220px;
    max-width: 220px;
}

#SidebarTitle {
    font-size: 22px;
    font-weight: 900;
    margin: 25px 20px;
    color: #38bdf8;
    letter-spacing: 1px;
}

#NavButton {
    background-color: transparent;
    color: #94a3b8;
    text-align: left;
    padding: 12px 25px;
    border-radius: 12px;
    margin: 4px 15px;
    font-size: 14px;
    font-weight: 600;
}

#NavButton:hover {
    background-color: rgba(56, 189, 248, 0.1);
    color: #f8fafc;
}

#NavButton[active="true"] {
    background-color: #1e293b;
    color: #38bdf8;
    font-weight: 800;
}

/* Content Area */
#Content {
    background-color: #030712;
}

#DropZone {
    border: 2px dashed #1e293b;
    border-radius: 24px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f172a, stop:1 #020617);
    margin: 20px;
}

#DropZone:hover {
    border: 2px dashed #38bdf8;
    background-color: #1e2944;
}

/* Premium Cards */
#ClipCard {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 18px;
    padding: 20px;
}

#ClipCard:hover {
    border: 1px solid rgba(56, 189, 248, 0.5);
    background-color: #151e33;
}

#ClipTitle {
    font-size: 17px;
    font-weight: 800;
    color: #f8fafc;
}

#ClipDuration {
    color: #64748b;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}

/* Action Buttons — Gradient */
QPushButton#ActionButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #38bdf8, stop:1 #0284c7);
    color: #ffffff;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 800;
    font-size: 13px;
    border: none;
}

QPushButton#ActionButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7dd3fc, stop:1 #0ea5e9);
}

QPushButton#ActionButton:pressed {
    background-color: #0369a1;
}

/* Secondary Buttons */
QPushButton#SecondaryButton {
    background-color: #1e293b;
    color: #cbd5e1;
    border-radius: 12px;
    padding: 10px 18px;
    font-weight: 700;
    border: 1px solid #334155;
}

QPushButton#SecondaryButton:hover {
    background-color: #334155;
    color: #f1f5f9;
}

/* Progress & Sliders */
QProgressBar {
    background-color: #1e293b;
    border-radius: 8px;
    height: 12px;
    text-align: center;
    border: 1px solid #0f172a;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38bdf8, stop:1 #818cf8);
    border-radius: 8px;
}

/* ScrollBar — Minimalist */
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #1e293b;
    min-height: 20px;
    border-radius: 3px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Slider */
QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #38bdf8;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

/* Text Inputs fallback */
QLineEdit, QTextEdit {
    background-color: #020617;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 10px;
    color: #f1f5f9;
}

QLineEdit:focus {
    border: 1px solid #38bdf8;
}

/* Dialogs & Message Boxes */
QMessageBox {
    background-color: #0f172a;
    border: 1px solid #1e293b;
}

QMessageBox QLabel {
    color: #f1f5f9;
    font-size: 14px;
    font-weight: 500;
}

QMessageBox QPushButton {
    background-color: #1e293b;
    color: #cbd5e1;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 700;
    min-width: 80px;
    border: 1px solid #334155;
}

QMessageBox QPushButton:hover {
    background-color: #334155;
    color: #f1f5f9;
}

QMessageBox QPushButton[default="true"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #38bdf8, stop:1 #0284c7);
    color: #ffffff;
    border: none;
}
"""
