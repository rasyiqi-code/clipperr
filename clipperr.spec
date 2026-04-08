# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Binaries: (source, destination_folder_in_bundle)
# We need to include the Rust core .so file
binaries = [
    ('venv/lib/python3.12/site-packages/clipperr_core/clipperr_core.cpython-312-x86_64-linux-gnu.so', 'clipperr_core')
]

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
