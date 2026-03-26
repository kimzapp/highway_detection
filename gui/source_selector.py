"""
Source Selector Widget
Widget cho phép người dùng chọn nguồn dữ liệu đầu vào
"""

import os

from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QComboBox,
    QFileDialog, QRadioButton, QButtonGroup,
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DEV-only flags: bật/tắt default local path nhanh trong quá trình phát triển.
DEV_USE_LOCAL_DEFAULT_PATHS = os.environ.get("HD_DEV_LOCAL_DEFAULT_PATHS", "1") == "1"
DEV_DEFAULT_VIDEO_PATH = os.environ.get(
    "HD_DEV_DEFAULT_VIDEO_PATH",
    r'E:\data_highway\highway_quality.mp4',
)


class SourceType(Enum):
    """Enum định nghĩa các loại nguồn đầu vào"""
    VIDEO = "video"
    CAMERA = "webcam"
    RTSP = "rtsp"
    IMAGES = "images"


@dataclass
class SourceConfig:
    """Cấu hình nguồn dữ liệu"""
    source_type: SourceType
    path: str = ""
    camera_id: int = 0
    rtsp_url: str = ""
    output_format: str = "mp4"
    output_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_type': self.source_type.value,
            'path': self.path,
            'camera_id': self.camera_id,
            'rtsp_url': self.rtsp_url,
            'output_format': self.output_format,
            'output_path': self.output_path,
        }


class SourceSelector(QWidget):
    """
    Widget để chọn nguồn dữ liệu đầu vào
    
    Signals:
        source_selected: Phát ra khi nguồn được chọn hợp lệ
        source_changed: Phát ra khi loại nguồn thay đổi
    """
    
    source_selected = pyqtSignal(object)  # SourceConfig
    source_changed = pyqtSignal(str)  # source type name
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self._set_default_output_path()
        self._apply_dev_default_video_path()

    def _apply_dev_default_video_path(self):
        """Áp dụng video path mặc định cho môi trường dev nếu bật cờ."""
        if not DEV_USE_LOCAL_DEFAULT_PATHS:
            return

        self._video_path_edit.setText(DEV_DEFAULT_VIDEO_PATH)
        self._set_default_output_path(video_path=DEV_DEFAULT_VIDEO_PATH)

        if os.path.isfile(DEV_DEFAULT_VIDEO_PATH):
            self._update_video_info(DEV_DEFAULT_VIDEO_PATH)
        else:
            self._video_info_label.setText("⚠️ DEV default video path không tồn tại")
            self._video_info_label.setStyleSheet("color: #ff9800; font-style: italic;")

        self._validate_source()
        
    def _setup_ui(self):
        """Thiết lập giao diện"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Chọn Nguồn Dữ Liệu")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Chọn loại nguồn và đường dẫn đến dữ liệu video")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #666666;")
        main_layout.addWidget(subtitle_label)
        
        main_layout.addSpacing(10)
        
        # Source type selection group
        source_group = QGroupBox("Loại Nguồn")
        source_layout = QHBoxLayout(source_group)
        source_layout.setSpacing(20)
        
        self._source_button_group = QButtonGroup(self)
        
        # Video file option
        self._video_radio = QRadioButton("📹 Video File")
        self._video_radio.setChecked(True)
        self._source_button_group.addButton(self._video_radio)
        source_layout.addWidget(self._video_radio)
        
        # Camera option (disabled for now but extensible)
        self._camera_radio = QRadioButton("📷 Camera")
        self._camera_radio.setEnabled(False)
        self._camera_radio.setToolTip("Sẽ được hỗ trợ trong phiên bản sau")
        self._source_button_group.addButton(self._camera_radio)
        source_layout.addWidget(self._camera_radio)
        
        # RTSP option (disabled for now but extensible)
        self._rtsp_radio = QRadioButton("🌐 RTSP Stream")
        self._rtsp_radio.setEnabled(False)
        self._rtsp_radio.setToolTip("Sẽ được hỗ trợ trong phiên bản sau")
        self._source_button_group.addButton(self._rtsp_radio)
        source_layout.addWidget(self._rtsp_radio)
        
        # Images folder option (disabled for now)
        self._images_radio = QRadioButton("🖼️ Image Folder")
        self._images_radio.setEnabled(False)
        self._images_radio.setToolTip("Sẽ được hỗ trợ trong phiên bản sau")
        self._source_button_group.addButton(self._images_radio)
        source_layout.addWidget(self._images_radio)
        
        main_layout.addWidget(source_group)
        
        # Stack widget for different source options
        self._create_video_input_section(main_layout)
        self._create_camera_input_section(main_layout)
        self._create_rtsp_input_section(main_layout)
        self._create_images_input_section(main_layout)
        
        # Initially show video section
        self._show_source_section(SourceType.VIDEO)
        
        # Spacer
        main_layout.addStretch(1)
        
        # Confirm button
        self._confirm_btn = QPushButton("Tiếp Tục →")
        self._confirm_btn.setMinimumHeight(45)
        self._confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self._confirm_btn.setEnabled(False)
        main_layout.addWidget(self._confirm_btn)
        
    def _create_video_input_section(self, parent_layout: QVBoxLayout):
        """Tạo section nhập video file"""
        self._video_section = QGroupBox("Video File")
        layout = QVBoxLayout(self._video_section)
        
        # File path input
        path_layout = QHBoxLayout()
        
        path_label = QLabel("Đường dẫn:")
        path_label.setMinimumWidth(80)
        path_layout.addWidget(path_label)
        
        self._video_path_edit = QLineEdit()
        self._video_path_edit.setPlaceholderText("Chọn file video (.mp4, .avi, .mkv...)")
        self._video_path_edit.setMinimumHeight(35)
        path_layout.addWidget(self._video_path_edit)
        
        self._browse_video_btn = QPushButton("Duyệt...")
        self._browse_video_btn.setMinimumHeight(35)
        self._browse_video_btn.setMinimumWidth(80)
        path_layout.addWidget(self._browse_video_btn)
        
        layout.addLayout(path_layout)
        
        # Video info (will be populated after selection)
        self._video_info_label = QLabel("")
        self._video_info_label.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(self._video_info_label)

        # Output format selection (step 1)
        output_format_layout = QHBoxLayout()

        output_format_label = QLabel("Định dạng output:")
        output_format_label.setMinimumWidth(110)
        output_format_layout.addWidget(output_format_label)

        self._output_format_combo = QComboBox()
        self._output_format_combo.setMinimumHeight(35)
        self._output_format_combo.addItem("MP4 (.mp4)", "mp4")
        self._output_format_combo.addItem("AVI (.avi)", "avi")
        self._output_format_combo.addItem("MOV (.mov)", "mov")
        self._output_format_combo.addItem("MKV (.mkv)", "mkv")
        output_format_layout.addWidget(self._output_format_combo, 1)

        format_hint = QLabel("Mặc định: mp4")
        format_hint.setStyleSheet("color: #666666;")
        output_format_layout.addWidget(format_hint)

        layout.addLayout(output_format_layout)

        # Output file path selection (step 1)
        output_path_layout = QHBoxLayout()

        output_path_label = QLabel("Lưu output:")
        output_path_label.setMinimumWidth(110)
        output_path_layout.addWidget(output_path_label)

        self._output_path_edit = QLineEdit()
        self._output_path_edit.setPlaceholderText("Chọn đường dẫn lưu video output")
        self._output_path_edit.setMinimumHeight(35)
        output_path_layout.addWidget(self._output_path_edit, 1)

        self._browse_output_btn = QPushButton("Duyệt...")
        self._browse_output_btn.setMinimumHeight(35)
        self._browse_output_btn.setMinimumWidth(80)
        output_path_layout.addWidget(self._browse_output_btn)

        layout.addLayout(output_path_layout)
        
        parent_layout.addWidget(self._video_section)
        
    def _create_camera_input_section(self, parent_layout: QVBoxLayout):
        """Tạo section nhập camera (placeholder cho tương lai)"""
        self._camera_section = QGroupBox("Camera")
        layout = QVBoxLayout(self._camera_section)
        
        # Camera selection
        camera_layout = QHBoxLayout()
        
        camera_label = QLabel("Camera ID:")
        camera_label.setMinimumWidth(80)
        camera_layout.addWidget(camera_label)
        
        self._camera_combo = QComboBox()
        self._camera_combo.addItem("Camera 0 (Default)", 0)
        self._camera_combo.addItem("Camera 1", 1)
        self._camera_combo.addItem("Camera 2", 2)
        self._camera_combo.setMinimumHeight(35)
        camera_layout.addWidget(self._camera_combo)
        
        self._detect_cameras_btn = QPushButton("Phát hiện")
        self._detect_cameras_btn.setMinimumHeight(35)
        self._detect_cameras_btn.setEnabled(False)
        camera_layout.addWidget(self._detect_cameras_btn)
        
        layout.addLayout(camera_layout)
        
        # Placeholder label
        placeholder = QLabel("⚠️ Tính năng camera sẽ được hỗ trợ trong phiên bản sau")
        placeholder.setStyleSheet("color: #ff9800;")
        layout.addWidget(placeholder)
        
        self._camera_section.hide()
        parent_layout.addWidget(self._camera_section)
        
    def _create_rtsp_input_section(self, parent_layout: QVBoxLayout):
        """Tạo section nhập RTSP stream (placeholder cho tương lai)"""
        self._rtsp_section = QGroupBox("RTSP Stream")
        layout = QVBoxLayout(self._rtsp_section)
        
        # RTSP URL input
        url_layout = QHBoxLayout()
        
        url_label = QLabel("RTSP URL:")
        url_label.setMinimumWidth(80)
        url_layout.addWidget(url_label)
        
        self._rtsp_url_edit = QLineEdit()
        self._rtsp_url_edit.setPlaceholderText("rtsp://username:password@ip:port/stream")
        self._rtsp_url_edit.setMinimumHeight(35)
        url_layout.addWidget(self._rtsp_url_edit)
        
        self._test_rtsp_btn = QPushButton("Test")
        self._test_rtsp_btn.setMinimumHeight(35)
        self._test_rtsp_btn.setEnabled(False)
        url_layout.addWidget(self._test_rtsp_btn)
        
        layout.addLayout(url_layout)
        
        # Placeholder label
        placeholder = QLabel("⚠️ Tính năng RTSP sẽ được hỗ trợ trong phiên bản sau")
        placeholder.setStyleSheet("color: #ff9800;")
        layout.addWidget(placeholder)
        
        self._rtsp_section.hide()
        parent_layout.addWidget(self._rtsp_section)
        
    def _create_images_input_section(self, parent_layout: QVBoxLayout):
        """Tạo section nhập folder ảnh (placeholder cho tương lai)"""
        self._images_section = QGroupBox("Image Folder")
        layout = QVBoxLayout(self._images_section)
        
        # Folder path input
        path_layout = QHBoxLayout()
        
        path_label = QLabel("Thư mục:")
        path_label.setMinimumWidth(80)
        path_layout.addWidget(path_label)
        
        self._images_path_edit = QLineEdit()
        self._images_path_edit.setPlaceholderText("Chọn thư mục chứa ảnh")
        self._images_path_edit.setMinimumHeight(35)
        path_layout.addWidget(self._images_path_edit)
        
        self._browse_images_btn = QPushButton("Duyệt...")
        self._browse_images_btn.setMinimumHeight(35)
        self._browse_images_btn.setEnabled(False)
        path_layout.addWidget(self._browse_images_btn)
        
        layout.addLayout(path_layout)
        
        # Placeholder label
        placeholder = QLabel("⚠️ Tính năng xử lý ảnh sẽ được hỗ trợ trong phiên bản sau")
        placeholder.setStyleSheet("color: #ff9800;")
        layout.addWidget(placeholder)
        
        self._images_section.hide()
        parent_layout.addWidget(self._images_section)
        
    def _connect_signals(self):
        """Kết nối các signals"""
        # Source type selection
        self._video_radio.toggled.connect(lambda checked: 
            self._on_source_type_changed(SourceType.VIDEO) if checked else None)
        self._camera_radio.toggled.connect(lambda checked: 
            self._on_source_type_changed(SourceType.CAMERA) if checked else None)
        self._rtsp_radio.toggled.connect(lambda checked: 
            self._on_source_type_changed(SourceType.RTSP) if checked else None)
        self._images_radio.toggled.connect(lambda checked: 
            self._on_source_type_changed(SourceType.IMAGES) if checked else None)
        
        # Video file browse
        self._browse_video_btn.clicked.connect(self._browse_video_file)
        self._browse_output_btn.clicked.connect(self._browse_output_file)
        self._output_format_combo.currentIndexChanged.connect(self._on_output_format_changed)
        self._video_path_edit.textChanged.connect(self._validate_source)
        
        # Confirm button
        self._confirm_btn.clicked.connect(self._on_confirm)
        
    def _on_source_type_changed(self, source_type: SourceType):
        """Xử lý khi loại nguồn thay đổi"""
        self._show_source_section(source_type)
        self.source_changed.emit(source_type.value)
        self._validate_source()
        
    def _show_source_section(self, source_type: SourceType):
        """Hiển thị section tương ứng với loại nguồn"""
        self._video_section.hide()
        self._camera_section.hide()
        self._rtsp_section.hide()
        self._images_section.hide()
        
        if source_type == SourceType.VIDEO:
            self._video_section.show()
        elif source_type == SourceType.CAMERA:
            self._camera_section.show()
        elif source_type == SourceType.RTSP:
            self._rtsp_section.show()
        elif source_type == SourceType.IMAGES:
            self._images_section.show()
            
    def _browse_video_file(self):
        """Mở dialog chọn file video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn File Video",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;All Files (*.*)"
        )
        if file_path:
            self._video_path_edit.setText(file_path)
            self._update_video_info(file_path)

            # Gợi ý output file theo tên input nếu đang dùng giá trị mặc định
            current_output = self._output_path_edit.text().strip()
            if not current_output or os.path.basename(current_output).lower().startswith("output"):
                self._set_default_output_path(video_path=file_path)

    def _browse_output_file(self):
        """Mở dialog chọn đường dẫn lưu output ở bước 1."""
        selected_ext = self._output_format_combo.currentData() or "mp4"
        current_path = self._output_path_edit.text().strip()

        default_path = current_path if current_path else self._build_default_output_path(ext=selected_ext)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Chọn Đường Dẫn Output",
            default_path,
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if file_path:
            self._output_path_edit.setText(file_path)

    def _on_output_format_changed(self, _index: int):
        """Đồng bộ extension của output path khi đổi định dạng."""
        current_path = self._output_path_edit.text().strip()
        selected_ext = self._output_format_combo.currentData() or "mp4"

        if not current_path:
            self._set_default_output_path()
            return

        root, _ = os.path.splitext(current_path)
        if not root:
            root = os.path.join(self._get_default_output_dir(), "output")
        self._output_path_edit.setText(f"{root}.{selected_ext}")

    def _get_project_root(self) -> str:
        """Lấy đường dẫn thư mục gốc dự án từ vị trí file GUI."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _get_default_output_dir(self) -> str:
        """Lấy thư mục output mặc định của dự án và đảm bảo thư mục tồn tại."""
        output_dir = os.path.join(self._get_project_root(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _build_default_output_path(self, ext: str = "mp4", video_path: str = "") -> str:
        """Tạo output path mặc định trong thư mục outputs của dự án."""
        output_dir = self._get_default_output_dir()
        if video_path:
            base_name = os.path.splitext(os.path.basename(video_path))[0] or "output"
            file_name = f"{base_name}_output.{ext}"
        else:
            file_name = f"output.{ext}"
        return os.path.join(output_dir, file_name)

    def _set_default_output_path(self, video_path: str = ""):
        """Đặt output path mặc định theo thư mục outputs của dự án."""
        ext = self._output_format_combo.currentData() or "mp4"
        self._output_path_edit.setText(self._build_default_output_path(ext=ext, video_path=video_path))
            
    def _update_video_info(self, file_path: str):
        """Cập nhật thông tin video"""
        import cv2
        import os
        
        if not os.path.exists(file_path):
            self._video_info_label.setText("❌ File không tồn tại")
            return
            
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frames / fps if fps > 0 else 0
                
                info_text = f"✅ {width}x{height} | {fps:.1f} FPS | {duration:.1f}s | {frames} frames"
                self._video_info_label.setText(info_text)
                self._video_info_label.setStyleSheet("color: #4CAF50; font-style: italic;")
                cap.release()
            else:
                self._video_info_label.setText("❌ Không thể mở video")
                self._video_info_label.setStyleSheet("color: #f44336; font-style: italic;")
        except Exception as e:
            self._video_info_label.setText(f"❌ Lỗi: {str(e)}")
            self._video_info_label.setStyleSheet("color: #f44336; font-style: italic;")
            
    def _validate_source(self):
        """Kiểm tra tính hợp lệ của nguồn đã chọn"""
        import os
        
        is_valid = False
        
        if self._video_radio.isChecked():
            video_path = self._video_path_edit.text().strip()
            is_valid = os.path.exists(video_path) and os.path.isfile(video_path)
        elif self._camera_radio.isChecked():
            is_valid = False  # Not implemented yet
        elif self._rtsp_radio.isChecked():
            is_valid = False  # Not implemented yet
        elif self._images_radio.isChecked():
            is_valid = False  # Not implemented yet
            
        self._confirm_btn.setEnabled(is_valid)
        return is_valid
        
    def _on_confirm(self):
        """Xử lý khi nhấn nút xác nhận"""
        config = self.get_source_config()
        if config:
            self.source_selected.emit(config)
            
    def get_source_config(self) -> Optional[SourceConfig]:
        """Lấy cấu hình nguồn hiện tại"""
        if self._video_radio.isChecked():
            video_path = self._video_path_edit.text().strip()
            if video_path:
                return SourceConfig(
                    source_type=SourceType.VIDEO,
                    path=video_path,
                    output_format=self._output_format_combo.currentData() or "mp4",
                    output_path=self._output_path_edit.text().strip(),
                )
        elif self._camera_radio.isChecked():
            return SourceConfig(
                source_type=SourceType.CAMERA,
                camera_id=self._camera_combo.currentData(),
                output_format="mp4",
                output_path=self._build_default_output_path(ext="mp4"),
            )
        elif self._rtsp_radio.isChecked():
            rtsp_url = self._rtsp_url_edit.text().strip()
            if rtsp_url:
                return SourceConfig(
                    source_type=SourceType.RTSP,
                    rtsp_url=rtsp_url,
                    output_format="mp4",
                    output_path=self._build_default_output_path(ext="mp4"),
                )
        elif self._images_radio.isChecked():
            images_path = self._images_path_edit.text().strip()
            if images_path:
                return SourceConfig(
                    source_type=SourceType.IMAGES,
                    path=images_path,
                    output_format="mp4",
                    output_path=self._build_default_output_path(ext="mp4"),
                )
        return None
    
    def get_current_source_type(self) -> SourceType:
        """Lấy loại nguồn hiện tại"""
        if self._video_radio.isChecked():
            return SourceType.VIDEO
        elif self._camera_radio.isChecked():
            return SourceType.CAMERA
        elif self._rtsp_radio.isChecked():
            return SourceType.RTSP
        elif self._images_radio.isChecked():
            return SourceType.IMAGES
        return SourceType.VIDEO
