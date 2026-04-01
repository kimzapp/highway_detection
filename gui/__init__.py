"""
GUI Module cho Highway Detection System
Cung cấp giao diện người dùng thân thiện để cấu hình và chạy hệ thống
"""

from .main_window import MainWindow
from .source_selector import SourceSelector
from .config_panel import ConfigPanel
from .video_preview import VideoPreviewWidget
from .zone_selector_widget import ZoneSelectorWidget
from .violation_history_page import ViolationHistoryPage
from .styles import apply_stylesheet, GLOBAL_STYLESHEET

__all__ = [
    'MainWindow',
    'SourceSelector', 
    'ConfigPanel',
    'VideoPreviewWidget',
    'ZoneSelectorWidget',
    'ViolationHistoryPage',
    'apply_stylesheet',
    'GLOBAL_STYLESHEET'
]
