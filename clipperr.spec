# -*- mode: python ; coding: utf-8 -*-

import os
import clipperr_core

block_cipher = None

# Get the path to the clipperr_core binary dynamically
core_dir = os.path.dirname(clipperr_core.__file__)
core_binaries = []
for f in os.listdir(core_dir):
    if f.startswith("clipperr_core") and (f.endswith(".so") or f.endswith(".pyd")):
        core_binaries.append((os.path.join(core_dir, f), 'clipperr_core'))

binaries = core_binaries

# Datas: (source, destination_folder_in_bundle)
datas = [
    ('app/assets/icon.png', 'app/assets'),
    ('.env', '.'),
]

a = Analysis(
    ['app/main.py'],
    pathex=['.', 'app'],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'faster_whisper',
        'mediapipe',
        'cv2',
        'numpy',
        'torch',
        'torchcodec',
        'clipperr_core',
        'PySide6.QtMultimedia',
        'PySide6.QtMultimediaWidgets',
        'psutil',
        'dotenv',
        'requests',
        'huggingface_hub'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['maturin', 'pip', 'setuptools'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='clipperr',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app/assets/icon.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='clipperr',
)
