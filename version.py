"""
Highway Detection - Version Information
"""

__version__ = "1.0.0"
__app_name__ = "Highway Detection"
__author__ = ""
__description__ = "Highway violation detection system using YOLO and computer vision"

VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}

def get_version():
    """Return full version string"""
    return __version__

def get_version_info():
    """Return version info dict"""
    return VERSION_INFO
