# -*- mode: python ; coding: utf-8 -*-

import os
import clipperr_core

block_cipher = None

binaries = []

# FFmpeg discovery for bundling
for f in ["ffmpeg", "ffprobe"]:
    ext = ".exe" if os.name == 'nt' else ""
    local_f = f + ext
    if os.path.exists(local_f):
        binaries.append((local_f, '.'))
    else:
        # Fallback for build environment (AppVeyor or Dev Machine)
        import shutil
        sys_f = shutil.which(f)
        if sys_f:
            binaries.append((sys_f, '.'))

# Datas: (source, destination_folder_in_bundle)
datas = [
    ('app/assets/icon.png', 'app/assets'),
]

# Robust .env handling
if not os.path.exists('.env'):
    with open('.env', 'w') as f: f.write("# Bundled Env\n")
datas.append(('.env', '.'))

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
    version='version_info.txt',
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
