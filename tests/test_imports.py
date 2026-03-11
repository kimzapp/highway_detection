"""
Unit tests for GUI components
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGUIComponents:
    """Test suite for GUI components"""
    
    def test_source_selector_import(self):
        """Test that SourceSelector can be imported"""
        from gui.source_selector import SourceSelector, SourceConfig, SourceType
        assert SourceSelector is not None
        assert SourceConfig is not None
        assert SourceType is not None
    
    def test_config_panel_import(self):
        """Test that ConfigPanel can be imported"""
        from gui.config_panel import ConfigPanel, ProcessingConfig
        assert ConfigPanel is not None
        assert ProcessingConfig is not None
    
    def test_styles_import(self):
        """Test that styles module can be imported"""
        from gui import styles
        assert hasattr(styles, 'apply_stylesheet')


class TestModelLoader:
    """Test suite for model loading"""
    
    def test_model_loader_import(self):
        """Test that model loader can be imported"""
        from models.loader import ModelLoader
        assert ModelLoader is not None
    
    def test_base_handler_import(self):
        """Test that base handler can be imported"""
        from models.base import BaseModelHandler
        assert BaseModelHandler is not None


class TestProcessing:
    """Test suite for video processing"""
    
    def test_video_processor_import(self):
        """Test that VideoProcessor can be imported"""
        from process.video import VideoProcessor
        assert VideoProcessor is not None


class TestTracking:
    """Test suite for tracking"""
    
    def test_bytetrack_import(self):
        """Test that ByteTrack can be imported"""
        from tracking.bytetrack import ByteTrack
        assert ByteTrack is not None


class TestViolations:
    """Test suite for violation detection"""
    
    def test_violation_detector_import(self):
        """Test that ViolationDetector can be imported"""
        from violations.detector import ViolationDetector
        assert ViolationDetector is not None


class TestLaneDetection:
    """Test suite for lane detection"""
    
    def test_road_zone_import(self):
        """Test that road zone module can be imported"""
        from lane_detection.road_zone import MultiRoadZoneOverlay
        assert MultiRoadZoneOverlay is not None
    
    def test_bird_eye_view_import(self):
        """Test that bird eye view module can be imported"""
        from lane_detection.bird_eye_view import BirdEyeView
        assert BirdEyeView is not None
