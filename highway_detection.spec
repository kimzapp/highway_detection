# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Highway Detection GUI
Build command: pyinstaller highway_detection.spec
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all necessary data files
datas = [
    # Include model files if they exist
    ('best.pt', '.'),
    ('best_lva.pt', '.'),
    ('best_lva.onnx', '.'),
    ('yolov8n.pt', '.'),
]

# Filter out non-existent files
datas = [(src, dst) for src, dst in datas if os.path.exists(src)]

# Hidden imports for dynamic imports
hidden_imports = [
    # PyQt5
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.sip',
    
    # OpenCV
    'cv2',
    
    # NumPy
    'numpy',
    
    # Torch and related
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torchvision',
    
    # Ultralytics YOLO
    'ultralytics',
    'ultralytics.nn',
    'ultralytics.nn.tasks',
    'ultralytics.engine',
    'ultralytics.engine.model',
    'ultralytics.engine.predictor',
    'ultralytics.engine.results',
    'ultralytics.utils',
    'ultralytics.utils.ops',
    'ultralytics.utils.torch_utils',
    
    # Supervision
    'supervision',
    'supervision.detection',
    'supervision.tracker',
    
    # ONNX Runtime
    'onnxruntime',
    
    # Project modules
    'gui',
    'gui.main_window',
    'gui.config_panel',
    'gui.source_selector',
    'gui.styles',
    'gui.video_preview',
    'gui.zone_selector_widget',
    'lane_detection',
    'lane_detection.bird_eye_view',
    'lane_detection.road_zone',
    'models',
    'models.base',
    'models.loader',
    'models.onnx_handler',
    'models.pt_handler',
    'process',
    'process.video',
    'tracking',
    'tracking.bytetrack',
    'violations',
    'violations.detector',
]

# Collect submodules
hidden_imports += collect_submodules('ultralytics')
hidden_imports += collect_submodules('supervision')

a = Analysis(
    ['run_gui.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
    ],
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
    name='HighwayDetection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HighwayDetection',
)
