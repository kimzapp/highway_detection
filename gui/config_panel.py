"""
Configuration Panel Widget
Widget cho phép người dùng cấu hình các tham số của hệ thống
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSlider,
    QFileDialog, QScrollArea, QFrame, QTabWidget,
    QGridLayout, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, Qt, QThread
from PyQt5.QtGui import QFont


# Danh sách các model Ultralytics được hỗ trợ tải tự động
ULTRALYTICS_MODELS = [
    # YOLOv8 Detection
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    # YOLOv8 Segmentation
    "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt",
    # YOLOv8 Pose
    "yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt",
    # YOLOv8 Classification
    "yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt",
    # YOLOv5
    "yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt",
    "yolov5n6.pt", "yolov5s6.pt", "yolov5m6.pt", "yolov5l6.pt", "yolov5x6.pt",
    # YOLOv9
    "yolov9c.pt", "yolov9e.pt", "yolov9t.pt", "yolov9s.pt", "yolov9m.pt",
    # YOLOv10
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
    # YOLO11
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    # RT-DETR
    "rtdetr-l.pt", "rtdetr-x.pt",
]


def is_ultralytics_model(model_name: str) -> bool:
    """Kiểm tra xem model name có phải là model Ultralytics hỗ trợ không"""
    name = os.path.basename(model_name).lower()
    return name in [m.lower() for m in ULTRALYTICS_MODELS]


class ModelLoaderThread(QThread):
    """Thread để load model trong background"""
    
    # Signals
    load_started = pyqtSignal()
    load_finished = pyqtSignal(bool, str, object)  # success, message, model_handler
    status_updated = pyqtSignal(str)  # status message for download progress
    
    def __init__(self, model_path: str, device: str, parent=None):
        super().__init__(parent)
        self._model_path = model_path
        self._device = device
        
    def run(self):
        """Load model trong thread riêng"""
        try:
            model_name = os.path.basename(self._model_path)
            
            # Kiểm tra nếu là model Ultralytics và file chưa tồn tại
            if not os.path.exists(self._model_path):
                if is_ultralytics_model(model_name):
                    # Tải model từ Ultralytics
                    self.status_updated.emit(f"⬇️ Đang tải xuống {model_name}...")
                    try:
                        from ultralytics import YOLO
                        # YOLO sẽ tự động download nếu model chưa có
                        model = YOLO(model_name)
                        model.to(self._device)
                        
                        # Wrap trong PTModelHandler để đồng nhất interface
                        from models.pt_handler import PTModelHandler
                        handler = PTModelHandler(model_name, self._device)
                        handler._ultralytics_model = model
                        handler._model = model
                        handler.names = model.names
                        
                        class_count = len(model.names) if model.names else 0
                        self.load_finished.emit(
                            True, 
                            f"Đã tải và load thành công ({class_count} classes)", 
                            handler
                        )
                        return
                        
                    except Exception as e:
                        self.load_finished.emit(
                            False, 
                            f"Không thể tải model {model_name}: {str(e)}", 
                            None
                        )
                        return
                else:
                    # Không phải model Ultralytics và file không tồn tại
                    supported_models = ", ".join(ULTRALYTICS_MODELS[:5]) + "..."
                    self.load_finished.emit(
                        False, 
                        f"File không tồn tại. Các model tự động tải: {supported_models}", 
                        None
                    )
                    return
            
            # File tồn tại - load bình thường
            from models import load_model
            
            self.status_updated.emit("⏳ Đang load model...")
            model_handler = load_model(self._model_path, self._device)
            
            # Kiểm tra model có class names không
            if model_handler.names:
                class_count = len(model_handler.names)
                self.load_finished.emit(True, f"Đã load thành công ({class_count} classes)", model_handler)
            else:
                self.load_finished.emit(True, "Đã load thành công", model_handler)
                
        except FileNotFoundError as e:
            self.load_finished.emit(False, f"File không tồn tại: {self._model_path}", None)
        except ValueError as e:
            self.load_finished.emit(False, f"Định dạng không hỗ trợ: {str(e)}", None)
        except Exception as e:
            self.load_finished.emit(False, f"Lỗi load model: {str(e)}", None)


@dataclass 
class ProcessingConfig:
    """Cấu hình xử lý video"""
    # Model settings
    model_path: str = "yolov8n.pt"
    device: str = "cpu"
    img_size: int = 640
    
    # Detection settings
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    classes: List[int] = field(default_factory=list)  # Empty = detect all classes (compatible with custom models)
    
    # Tracker settings  
    max_age: int = 90
    trace_length: int = 25
    
    # Visualization settings
    show_boxes: bool = True
    show_labels: bool = True
    show_traces: bool = True
    
    # BEV settings
    enable_bev: bool = True
    bev_width: int = 400
    bev_height: int = 600
    bev_method: str = "ipm"
    camera_height: float = 1.5
    
    # Output settings
    output_path: str = "output.mp4"
    output_format: str = "mp4"
    save_video: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_path': self.model_path,
            'device': self.device,
            'img_size': self.img_size,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'classes': self.classes,
            'max_age': self.max_age,
            'trace_length': self.trace_length,
            'show_boxes': self.show_boxes,
            'show_labels': self.show_labels,
            'show_traces': self.show_traces,
            'enable_bev': self.enable_bev,
            'bev_width': self.bev_width,
            'bev_height': self.bev_height,
            'bev_method': self.bev_method,
            'camera_height': self.camera_height,
            'output_path': self.output_path,
            'output_format': self.output_format,
            'save_video': self.save_video
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# COCO class names cho vehicle classes
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle", 
    5: "Bus",
    7: "Truck",
    0: "Person",
    1: "Bicycle"
}


class ConfigPanel(QWidget):
    """
    Panel cấu hình các tham số xử lý
    
    Signals:
        config_changed: Phát ra khi cấu hình thay đổi
        config_confirmed: Phát ra khi người dùng xác nhận cấu hình
        model_status_changed: Phát ra khi trạng thái model thay đổi (ready, error message)
    """
    
    config_changed = pyqtSignal(object)  # ProcessingConfig
    config_confirmed = pyqtSignal(object)  # ProcessingConfig
    model_status_changed = pyqtSignal(bool, str)  # is_ready, status_message
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config = ProcessingConfig()
        self._model_handler = None  # Pre-loaded model handler
        self._model_loader_thread: Optional[ModelLoaderThread] = None
        self._model_ready = False
        self._setup_ui()
        self._connect_signals()
        self._load_config_to_ui()
        
    def _setup_ui(self):
        """Thiết lập giao diện"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("Cấu Hình Tham Số")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create scroll area for settings
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)
        
        # Tab widget for organized settings
        tab_widget = QTabWidget()
        
        # Model tab
        model_tab = self._create_model_tab()
        tab_widget.addTab(model_tab, "🤖 Model")
        
        # Detection tab
        detection_tab = self._create_detection_tab()
        tab_widget.addTab(detection_tab, "🎯 Detection")
        
        # Tracking tab
        tracking_tab = self._create_tracking_tab()
        tab_widget.addTab(tracking_tab, "📍 Tracking")
        
        # Visualization tab
        viz_tab = self._create_visualization_tab()
        tab_widget.addTab(viz_tab, "👁️ Hiển thị")
        
        scroll_layout.addWidget(tab_widget)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 1)
        
        # Button row
        button_layout = QHBoxLayout()
        
        self._reset_btn = QPushButton("🔄 Reset Mặc Định")
        self._reset_btn.setMinimumHeight(40)
        button_layout.addWidget(self._reset_btn)
        
        button_layout.addStretch()
        
        self._back_btn = QPushButton("← Quay Lại")
        self._back_btn.setMinimumHeight(40)
        button_layout.addWidget(self._back_btn)
        
        self._confirm_btn = QPushButton("Tiếp Tục →")
        self._confirm_btn.setMinimumHeight(40)
        self._confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_layout.addWidget(self._confirm_btn)
        
        main_layout.addLayout(button_layout)
        
    def _create_model_tab(self) -> QWidget:
        """Tạo tab cấu hình model"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Hint label
        hint_label = QLabel("💡 Chọn preset hoặc duyệt file model - Model sẽ tự động được load")
        hint_label.setStyleSheet("color: #666666; font-style: italic; padding: 5px;")
        layout.addWidget(hint_label)
        
        # Model file selection
        model_group = QGroupBox("Model File")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Model:"), 0, 0)
        
        self._model_path_edit = QLineEdit()
        self._model_path_edit.setPlaceholderText("VD: yolov8n.pt hoặc đường dẫn đến file model")
        self._model_path_edit.setMinimumHeight(35)
        self._model_path_edit.setToolTip(
            "Nhập tên model Ultralytics (yolov8n.pt, yolov8s.pt,...) để tự động tải,\n"
            "hoặc đường dẫn đến file model đã có (.pt, .onnx)"
        )
        model_layout.addWidget(self._model_path_edit, 0, 1)
        
        self._browse_model_btn = QPushButton("Duyệt...")
        self._browse_model_btn.setMinimumHeight(35)
        model_layout.addWidget(self._browse_model_btn, 0, 2)
        
        # Preset models
        model_layout.addWidget(QLabel("Preset:"), 1, 0)
        self._model_preset_combo = QComboBox()
        self._model_preset_combo.addItem("YOLOv8n (Nhanh)", "yolov8n.pt")
        self._model_preset_combo.addItem("YOLOv8s (Cân bằng)", "yolov8s.pt")
        self._model_preset_combo.addItem("YOLOv8m (Chính xác)", "yolov8m.pt")
        self._model_preset_combo.addItem("Custom Model", "")
        self._model_preset_combo.setMinimumHeight(35)
        model_layout.addWidget(self._model_preset_combo, 1, 1, 1, 2)
        
        # Model status row
        model_layout.addWidget(QLabel("Trạng thái:"), 2, 0)
        
        model_status_widget = QWidget()
        model_status_layout = QHBoxLayout(model_status_widget)
        model_status_layout.setContentsMargins(0, 0, 0, 0)
        
        self._model_status_label = QLabel("⚪ Chọn preset hoặc nhập tên model để tự động tải")
        self._model_status_label.setStyleSheet("color: #999999;")
        self._model_status_label.setWordWrap(True)
        model_status_layout.addWidget(self._model_status_label, 1)
        
        self._load_model_btn = QPushButton("🔄 Reload")
        self._load_model_btn.setMinimumHeight(35)
        self._load_model_btn.setMinimumWidth(100)
        self._load_model_btn.setToolTip("Load lại model (dùng khi thay đổi device)")
        self._load_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        model_status_layout.addWidget(self._load_model_btn)
        
        model_layout.addWidget(model_status_widget, 2, 1, 1, 2)
        
        layout.addWidget(model_group)
        
        # Device selection
        device_group = QGroupBox("Thiết Bị Xử Lý")
        device_layout = QHBoxLayout(device_group)
        
        device_layout.addWidget(QLabel("Device:"))
        
        self._device_combo = QComboBox()
        self._device_combo.addItem("CPU", "cpu")
        self._device_combo.addItem("GPU (CUDA)", "cuda")
        self._device_combo.addItem("GPU 0", "cuda:0")
        self._device_combo.addItem("GPU 1", "cuda:1")
        self._device_combo.setMinimumHeight(35)
        device_layout.addWidget(self._device_combo)
        
        self._detect_gpu_btn = QPushButton("Kiểm tra GPU")
        self._detect_gpu_btn.setMinimumHeight(35)
        device_layout.addWidget(self._detect_gpu_btn)
        
        self._gpu_info_label = QLabel("")
        self._gpu_info_label.setStyleSheet("color: #666666;")
        device_layout.addWidget(self._gpu_info_label)
        
        device_layout.addStretch()
        
        layout.addWidget(device_group)
        
        # Image size
        size_group = QGroupBox("Kích Thước Input")
        size_layout = QHBoxLayout(size_group)
        
        size_layout.addWidget(QLabel("Image Size:"))
        
        self._img_size_combo = QComboBox()
        self._img_size_combo.addItem("320 (Nhanh)", 320)
        self._img_size_combo.addItem("416", 416)
        self._img_size_combo.addItem("512", 512)
        self._img_size_combo.addItem("640 (Mặc định)", 640)
        self._img_size_combo.addItem("832", 832)
        self._img_size_combo.addItem("1024 (Chính xác)", 1024)
        self._img_size_combo.setCurrentIndex(3)  # Default 640
        self._img_size_combo.setMinimumHeight(35)
        size_layout.addWidget(self._img_size_combo)
        
        size_layout.addStretch()
        
        layout.addWidget(size_group)
        
        layout.addStretch()
        
        return widget
        
    def _create_detection_tab(self) -> QWidget:
        """Tạo tab cấu hình detection"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Thresholds
        thresh_group = QGroupBox("Ngưỡng Phát Hiện")
        thresh_layout = QGridLayout(thresh_group)
        
        # Confidence threshold
        thresh_layout.addWidget(QLabel("Confidence:"), 0, 0)
        
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(5, 95)
        self._conf_slider.setValue(25)
        thresh_layout.addWidget(self._conf_slider, 0, 1)
        
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)
        self._conf_spin.setMinimumWidth(80)
        thresh_layout.addWidget(self._conf_spin, 0, 2)
        
        # IOU threshold
        thresh_layout.addWidget(QLabel("IOU:"), 1, 0)
        
        self._iou_slider = QSlider(Qt.Horizontal)
        self._iou_slider.setRange(10, 90)
        self._iou_slider.setValue(50)
        thresh_layout.addWidget(self._iou_slider, 1, 1)
        
        self._iou_spin = QDoubleSpinBox()
        self._iou_spin.setRange(0.1, 0.9)
        self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setValue(0.5)
        self._iou_spin.setMinimumWidth(80)
        thresh_layout.addWidget(self._iou_spin, 1, 2)
        
        layout.addWidget(thresh_group)
        
        # Class selection
        class_group = QGroupBox("Loại Đối Tượng Phát Hiện")
        class_layout = QVBoxLayout(class_group)
        
        class_hint = QLabel("Chọn các loại đối tượng cần theo dõi:")
        class_hint.setStyleSheet("color: #666666;")
        class_layout.addWidget(class_hint)
        
        self._class_checkboxes = {}
        class_grid = QGridLayout()
        
        for i, (class_id, class_name) in enumerate(VEHICLE_CLASSES.items()):
            checkbox = QCheckBox(f"{class_name} (ID: {class_id})")
            checkbox.setChecked(True)  # Default: detect all classes (compatible with custom models)
            self._class_checkboxes[class_id] = checkbox
            class_grid.addWidget(checkbox, i // 3, i % 3)
            
        class_layout.addLayout(class_grid)
        
        # Quick select buttons
        quick_layout = QHBoxLayout()
        
        self._select_vehicles_btn = QPushButton("Chọn Xe Cộ")
        self._select_vehicles_btn.clicked.connect(self._select_vehicle_classes)
        quick_layout.addWidget(self._select_vehicles_btn)
        
        self._select_all_btn = QPushButton("Chọn Tất Cả")
        self._select_all_btn.clicked.connect(self._select_all_classes)
        quick_layout.addWidget(self._select_all_btn)
        
        self._select_none_btn = QPushButton("Bỏ Chọn")
        self._select_none_btn.clicked.connect(self._clear_all_classes)
        quick_layout.addWidget(self._select_none_btn)
        
        quick_layout.addStretch()
        class_layout.addLayout(quick_layout)
        
        layout.addWidget(class_group)
        
        layout.addStretch()
        
        return widget
        
    def _create_tracking_tab(self) -> QWidget:
        """Tạo tab cấu hình tracking"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Tracker settings
        tracker_group = QGroupBox("ByteTrack Settings")
        tracker_layout = QGridLayout(tracker_group)
        
        # Max age
        tracker_layout.addWidget(QLabel("Max Age (frames):"), 0, 0)
        self._max_age_spin = QSpinBox()
        self._max_age_spin.setRange(10, 300)
        self._max_age_spin.setValue(90)
        self._max_age_spin.setMinimumHeight(35)
        self._max_age_spin.setToolTip("Số frame tối đa giữ track khi không có detection")
        tracker_layout.addWidget(self._max_age_spin, 0, 1)
        
        # Trace length
        tracker_layout.addWidget(QLabel("Trace Length:"), 1, 0)
        self._trace_length_spin = QSpinBox()
        self._trace_length_spin.setRange(10, 200)
        self._trace_length_spin.setValue(50)
        self._trace_length_spin.setMinimumHeight(35)
        self._trace_length_spin.setToolTip("Độ dài của vệt theo dõi hiển thị")
        tracker_layout.addWidget(self._trace_length_spin, 1, 1)
        
        tracker_layout.setColumnStretch(2, 1)
        
        layout.addWidget(tracker_group)
        
        # Info box
        info_label = QLabel(
            "💡 <b>Gợi ý:</b><br>"
            "• <b>Max Age</b>: Tăng nếu xe bị che khuất lâu<br>"
            "• <b>Trace Length</b>: Tăng để thấy quỹ đạo dài hơn"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("""
            QLabel {
                background-color: #e3f2fd;
                padding: 15px;
                border-radius: 5px;
            }
        """)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        return widget
        
    def _create_visualization_tab(self) -> QWidget:
        """Tạo tab cấu hình hiển thị"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # Display options
        display_group = QGroupBox("Tùy Chọn Hiển Thị")
        display_layout = QVBoxLayout(display_group)
        
        self._show_boxes_check = QCheckBox("Hiển thị bounding boxes")
        self._show_boxes_check.setChecked(True)
        display_layout.addWidget(self._show_boxes_check)
        
        self._show_labels_check = QCheckBox("Hiển thị nhãn (class, ID)")
        self._show_labels_check.setChecked(True)
        display_layout.addWidget(self._show_labels_check)
        
        self._show_traces_check = QCheckBox("Hiển thị vệt di chuyển")
        self._show_traces_check.setChecked(True)
        display_layout.addWidget(self._show_traces_check)
        
        layout.addWidget(display_group)
        
        # BEV options
        bev_group = QGroupBox("Bird's Eye View (BEV)")
        bev_layout = QGridLayout(bev_group)
        
        self._enable_bev_check = QCheckBox("Bật Bird's Eye View")
        self._enable_bev_check.setChecked(True)
        bev_layout.addWidget(self._enable_bev_check, 0, 0, 1, 2)
        
        bev_layout.addWidget(QLabel("Phương pháp:"), 1, 0)
        self._bev_method_combo = QComboBox()
        self._bev_method_combo.addItem("IPM (Inverse Perspective Mapping)", "ipm")
        self._bev_method_combo.addItem("Homography", "homography")
        self._bev_method_combo.setMinimumHeight(35)
        bev_layout.addWidget(self._bev_method_combo, 1, 1)
        
        bev_layout.addWidget(QLabel("Chiều rộng BEV:"), 2, 0)
        self._bev_width_spin = QSpinBox()
        self._bev_width_spin.setRange(200, 800)
        self._bev_width_spin.setValue(400)
        self._bev_width_spin.setMinimumHeight(35)
        bev_layout.addWidget(self._bev_width_spin, 2, 1)
        
        bev_layout.addWidget(QLabel("Chiều cao BEV:"), 3, 0)
        self._bev_height_spin = QSpinBox()
        self._bev_height_spin.setRange(300, 1000)
        self._bev_height_spin.setValue(600)
        self._bev_height_spin.setMinimumHeight(35)
        bev_layout.addWidget(self._bev_height_spin, 3, 1)
        
        bev_layout.addWidget(QLabel("Chiều cao camera (m):"), 4, 0)
        self._camera_height_spin = QDoubleSpinBox()
        self._camera_height_spin.setRange(0.5, 10.0)
        self._camera_height_spin.setValue(1.5)
        self._camera_height_spin.setSingleStep(0.1)
        self._camera_height_spin.setMinimumHeight(35)
        self._camera_height_spin.setToolTip("Chiều cao camera so với mặt đường (cho IPM)")
        bev_layout.addWidget(self._camera_height_spin, 4, 1)
        
        layout.addWidget(bev_group)
        
        layout.addStretch()
        
        return widget
        
    def _connect_signals(self):
        """Kết nối các signals"""
        # Model
        self._browse_model_btn.clicked.connect(self._browse_model_file)
        self._model_preset_combo.currentIndexChanged.connect(self._on_model_preset_changed)
        self._detect_gpu_btn.clicked.connect(self._detect_gpu)
        self._load_model_btn.clicked.connect(self._load_model)
        
        # Auto reset model status when model path changes (user typing manually)
        self._model_path_edit.textChanged.connect(self._on_model_path_changed)
        # Load model when user press Enter
        self._model_path_edit.returnPressed.connect(self._load_model)
        # Auto reload when device changes
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        
        # Threshold sliders and spinboxes
        self._conf_slider.valueChanged.connect(lambda v: self._conf_spin.setValue(v / 100))
        self._conf_spin.valueChanged.connect(lambda v: self._conf_slider.setValue(int(v * 100)))
        
        self._iou_slider.valueChanged.connect(lambda v: self._iou_spin.setValue(v / 100))
        self._iou_spin.valueChanged.connect(lambda v: self._iou_slider.setValue(int(v * 100)))
        
        # Buttons
        self._reset_btn.clicked.connect(self._reset_to_defaults)
        self._confirm_btn.clicked.connect(self._on_confirm)
    
    def _on_model_path_changed(self):
        """Reset model status khi người dùng thay đổi path thủ công"""
        # Chỉ reset, không auto load (người dùng đang gõ)
        self._model_ready = False
        self._model_handler = None
        
        model_name = self._model_path_edit.text().strip()
        if model_name:
            if is_ultralytics_model(model_name):
                self._model_status_label.setText("⚪ Nhấn Enter để tải và load model")
            else:
                self._model_status_label.setText("⚪ Nhấn Enter để load model")
        else:
            self._model_status_label.setText("⚪ Chọn preset hoặc nhập tên model")
        
        self._model_status_label.setStyleSheet("color: #999999;")
        self._load_model_btn.setEnabled(True)
        self.model_status_changed.emit(False, "Chưa load model")
    
    def _on_device_changed(self):
        """Tự động reload model khi thay đổi device"""
        model_path = self._model_path_edit.text().strip()
        if model_path and self._model_handler is not None:
            # Đã có model trước đó, tự động reload với device mới
            self._load_model()
    
    def _load_model(self):
        """Bắt đầu load model trong background"""
        model_path = self._model_path_edit.text().strip()
        device = self._device_combo.currentData() or "cpu"
        
        if not model_path:
            self._model_status_label.setText("❌ Chưa chọn model")
            self._model_status_label.setStyleSheet("color: #f44336;")
            return
        
        # Nếu đang load thì không làm gì
        if self._model_loader_thread is not None and self._model_loader_thread.isRunning():
            return
        
        # Update UI
        self._model_status_label.setText("⏳ Đang load model...")
        self._model_status_label.setStyleSheet("color: #FF9800;")
        self._load_model_btn.setEnabled(False)
        self._load_model_btn.setText("⏳ Đang load...")
        
        # Create and start loader thread
        self._model_loader_thread = ModelLoaderThread(model_path, device, self)
        self._model_loader_thread.load_finished.connect(self._on_model_load_finished)
        self._model_loader_thread.status_updated.connect(self._on_model_status_updated)
        self._model_loader_thread.start()
    
    def _on_model_status_updated(self, status: str):
        """Callback khi có cập nhật trạng thái trong quá trình load"""
        self._model_status_label.setText(status)
        self._model_status_label.setStyleSheet("color: #FF9800;")
    
    def _on_model_load_finished(self, success: bool, message: str, model_handler):
        """Callback khi load model xong"""
        self._load_model_btn.setEnabled(True)
        self._load_model_btn.setText("🔄 Reload")
        
        if success:
            self._model_ready = True
            self._model_handler = model_handler
            self._model_status_label.setText(f"✅ {message}")
            self._model_status_label.setStyleSheet("color: #4CAF50;")
            self.model_status_changed.emit(True, message)
        else:
            self._model_ready = False
            self._model_handler = None
            self._model_status_label.setText(f"❌ {message}")
            self._model_status_label.setStyleSheet("color: #f44336;")
            self.model_status_changed.emit(False, message)
    
    def get_model_handler(self):
        """Lấy model handler đã load sẵn"""
        return self._model_handler
    
    def is_model_ready(self) -> bool:
        """Kiểm tra model đã sẵn sàng chưa"""
        return self._model_ready
        
    def _browse_model_file(self):
        """Mở dialog chọn model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn Model File",
            "",
            "Model Files (*.pt *.pth *.onnx);;All Files (*.*)"
        )
        if file_path:
            # Tạm ngắt signal để tránh trigger _on_model_path_changed
            self._model_preset_combo.blockSignals(True)
            self._model_path_edit.blockSignals(True)
            
            self._model_preset_combo.setCurrentIndex(
                self._model_preset_combo.count() - 1  # Select "Custom Model"
            )
            self._model_path_edit.setText(file_path)
            
            self._model_preset_combo.blockSignals(False)
            self._model_path_edit.blockSignals(False)
            
            # Tự động load model sau khi chọn file
            self._load_model()
            
    def _on_model_preset_changed(self, index: int):
        """Xử lý khi chọn preset model"""
        model_path = self._model_preset_combo.currentData()
        if model_path:
            # Block signal để tránh trigger _on_model_path_changed
            self._model_path_edit.blockSignals(True)
            self._model_path_edit.setText(model_path)
            self._model_path_edit.blockSignals(False)
            # Tự động load model khi chọn preset
            self._load_model()
            
    def _detect_gpu(self):
        try:
            import torch

            if torch.cuda.is_available():
                prop = torch.cuda.get_device_properties(0)
                name = prop.name
                mem = prop.total_memory / 1e9
                text = f"✅ {name} ({mem:.1f}GB)"
                color = "#4CAF50"
            else:
                text = "⚠️ Đang chạy bằng CPU"
                color = "#ff9800"

        except Exception:
            text = "❌ Không thể kiểm tra GPU"
            color = "#f44336"

        self._gpu_info_label.setText(text)
        self._gpu_info_label.setStyleSheet(f"color: {color};")
            
    def _select_vehicle_classes(self):
        """Chọn các class xe cộ"""
        vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        for class_id, checkbox in self._class_checkboxes.items():
            checkbox.setChecked(class_id in vehicle_ids)
            
    def _select_all_classes(self):
        """Chọn tất cả classes"""
        for checkbox in self._class_checkboxes.values():
            checkbox.setChecked(True)
            
    def _clear_all_classes(self):
        """Bỏ chọn tất cả classes"""
        for checkbox in self._class_checkboxes.values():
            checkbox.setChecked(False)
            
    def _reset_to_defaults(self):
        """Reset về giá trị mặc định"""
        reply = QMessageBox.question(
            self,
            "Xác Nhận Reset",
            "Bạn có chắc muốn reset tất cả cấu hình về mặc định?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._config = ProcessingConfig()
            self._load_config_to_ui()
            
    def _load_config_to_ui(self):
        """Load config vào UI"""
        # Model
        self._model_path_edit.setText(self._config.model_path)
        
        # Device
        device_index = self._device_combo.findData(self._config.device)
        if device_index >= 0:
            self._device_combo.setCurrentIndex(device_index)
            
        # Image size
        size_index = self._img_size_combo.findData(self._config.img_size)
        if size_index >= 0:
            self._img_size_combo.setCurrentIndex(size_index)
            
        # Thresholds
        self._conf_spin.setValue(self._config.conf_threshold)
        self._iou_spin.setValue(self._config.iou_threshold)
        
        # Classes
        for class_id, checkbox in self._class_checkboxes.items():
            checkbox.setChecked(class_id in self._config.classes)
            
        # Tracker
        self._max_age_spin.setValue(self._config.max_age)
        self._trace_length_spin.setValue(self._config.trace_length)
        
        # Visualization
        self._show_boxes_check.setChecked(self._config.show_boxes)
        self._show_labels_check.setChecked(self._config.show_labels)
        self._show_traces_check.setChecked(self._config.show_traces)
        
        # BEV
        self._enable_bev_check.setChecked(self._config.enable_bev)
        self._bev_width_spin.setValue(self._config.bev_width)
        self._bev_height_spin.setValue(self._config.bev_height)
        self._camera_height_spin.setValue(self._config.camera_height)
        
        bev_method_index = self._bev_method_combo.findData(self._config.bev_method)
        if bev_method_index >= 0:
            self._bev_method_combo.setCurrentIndex(bev_method_index)
            
    def _save_ui_to_config(self):
        """Lưu UI vào config"""
        self._config.model_path = self._model_path_edit.text().strip() or "yolov8n.pt"
        self._config.device = self._device_combo.currentData() or "cpu"
        self._config.img_size = self._img_size_combo.currentData() or 640
        
        self._config.conf_threshold = self._conf_spin.value()
        self._config.iou_threshold = self._iou_spin.value()
        
        self._config.classes = [
            class_id for class_id, checkbox in self._class_checkboxes.items()
            if checkbox.isChecked()
        ]
        
        self._config.max_age = self._max_age_spin.value()
        self._config.trace_length = self._trace_length_spin.value()
        
        self._config.show_boxes = self._show_boxes_check.isChecked()
        self._config.show_labels = self._show_labels_check.isChecked()
        self._config.show_traces = self._show_traces_check.isChecked()
        
        self._config.enable_bev = self._enable_bev_check.isChecked()
        self._config.bev_width = self._bev_width_spin.value()
        self._config.bev_height = self._bev_height_spin.value()
        self._config.bev_method = self._bev_method_combo.currentData() or "ipm"
        self._config.camera_height = self._camera_height_spin.value()
        
        self._config.save_video = bool(self._config.output_path)
        self._config.output_format = self._config.output_format or "mp4"
        output_path = self._config.output_path.strip() if self._config.output_path else f"output.{self._config.output_format}"
        root, ext = os.path.splitext(output_path)
        if not root:
            root = "output"
        current_ext = ext.lower().lstrip('.')
        if current_ext != self._config.output_format:
            output_path = f"{root}.{self._config.output_format}"
        self._config.output_path = output_path
        
    def _on_confirm(self):
        """Xử lý khi nhấn xác nhận"""
        self._save_ui_to_config()
        
        # Kiểm tra model đã được load chưa
        if not self._model_ready:
            reply = QMessageBox.question(
                self,
                "Model chưa được load",
                "Model chưa được load. Bạn có muốn load model trước khi tiếp tục?\n\n"
                "- Nhấn 'Yes' để load model trước\n"
                "- Nhấn 'No' để tiếp tục (model sẽ được load khi xử lý)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self._load_model()
                return
        
        self.config_confirmed.emit(self._config)
        
    def get_config(self) -> ProcessingConfig:
        """Lấy cấu hình hiện tại"""
        self._save_ui_to_config()
        return self._config
        
    def set_config(self, config: ProcessingConfig):
        """Đặt cấu hình"""
        self._config = config
        self._load_config_to_ui()

    def set_output_preferences(self, output_format: str, source_path: str = "", output_path: str = ""):
        """Đồng bộ output từ bước chọn video (không phụ thuộc UI bước 2)."""
        if not output_format:
            output_format = "mp4"
        self._config.output_format = output_format.lower()

        if output_path:
            root, ext = os.path.splitext(output_path)
            if not root:
                root = "output"
            current_ext = ext.lower().lstrip('.')
            if current_ext != self._config.output_format:
                output_path = f"{root}.{self._config.output_format}"
            self._config.output_path = output_path
            return

        if source_path:
            source_root = os.path.splitext(os.path.basename(source_path))[0] or "output"
            self._config.output_path = f"{source_root}_output.{self._config.output_format}"
        else:
            self._config.output_path = f"output.{self._config.output_format}"
