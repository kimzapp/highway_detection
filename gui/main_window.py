"""
Main Window
Cửa sổ chính của ứng dụng Highway Detection GUI
"""

import os
import sys
import time
import cv2
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum, auto

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLabel, QPushButton, QProgressBar,
    QStatusBar, QMessageBox, QApplication, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from .source_selector import SourceSelector, SourceConfig
from .config_panel import ConfigPanel, ProcessingConfig
from .zone_selector_widget import ZoneSelectorWidget
from .styles import apply_stylesheet
from app_version import get_display_version


class AppState(Enum):
    """Trạng thái của ứng dụng"""
    SOURCE_SELECTION = auto()
    CONFIG = auto()
    ZONE_SELECTION = auto()
    PROCESSING = auto()
    COMPLETED = auto()


@dataclass
class AppConfig:
    """Cấu hình tổng hợp của ứng dụng"""
    source_config: Optional[SourceConfig] = None
    processing_config: Optional[ProcessingConfig] = None
    zones: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.zones is None:
            self.zones = []


class ProcessingThread(QThread):
    """Thread xử lý video"""
    
    progress_updated = pyqtSignal(int, int)  # current_frame, total_frames
    frame_processed = pyqtSignal(object)  # processed frame (using object to avoid numpy memory issues)
    processing_completed = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: AppConfig, model_handler=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._model_handler = model_handler  # Pre-loaded model handler
        self._is_running = True
        self._processor = None
        self._preview_max_fps = 12.0
        self._preview_interval_s = 1.0 / self._preview_max_fps
        
    def run(self):
        """Chạy xử lý video"""
        try:
            from process.video import VideoProcessor, AsyncVideoWriter
            
            # Create args-like object
            args = self._create_args()
            
            # Create processor with pre-loaded model if available
            self._processor = VideoProcessor(args, model_handler=self._model_handler)
            
            # Set BEV options
            self._processor.enable_bev = args.enable_bev
            self._processor.bev_width = args.bev_width
            self._processor.bev_height = args.bev_height
            self._processor.bev_method = args.bev_method
            self._processor.camera_height = args.camera_height
            
            # Process video with GUI display (not OpenCV window)
            output_path = args.output if args.save_video else None
            
            self._process_video_for_gui(
                video_path=args.input,
                output_path=output_path,
                preset_zones=self._config.zones if self._config.zones else None,
                writer_class=AsyncVideoWriter,
            )
            
            self.processing_completed.emit()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))
    
    def _process_video_for_gui(self, video_path: str, output_path: Optional[str], preset_zones, writer_class=None):
        """Xử lý video và emit frame về GUI thay vì dùng cv2.imshow"""
        from lane_mapping.road_zone import MultiRoadZoneOverlay
        from lane_mapping.bird_eye_view import create_combined_view
        from violations import ViolationDetector, ViolationVisualizer, ViolationType
        
        processor = self._processor
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize tracker
        processor._init_tracker(fps)
        print(f"Tracker initialized: box_annotator={processor.tracker.box_annotator is not None}, "
              f"trace_annotator={processor.tracker.trace_annotator is not None}")
        
        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Handle preset zones
        first_frame = None
        zone_polygons = None
        
        if preset_zones is not None and len(preset_zones) > 0:
            zone_polygons = preset_zones
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Cannot read first frame from video")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            print(f"Using {len(zone_polygons)} preset zone(s)")
            
            # Initialize road zone overlay
            processor.road_zone_overlay = MultiRoadZoneOverlay(
                zone_polygons=zone_polygons,
                alpha=0.2,
            )
            
            # Initialize Violation Detector
            processor.violation_detector = ViolationDetector(
                min_violation_frames=processor.min_violation_frames,
                min_normal_frames=3,
                enabled_violations={ViolationType.WRONG_LANE}
            )
            np_zone_polygons = [np.array(z) for z in zone_polygons]
            processor.violation_detector.set_valid_zones(np_zone_polygons)
            
            # Initialize Violation Visualizer
            processor.violation_visualizer = ViolationVisualizer(
                detector=processor.violation_detector,
                show_violation_box=True,
                show_violation_label=True,
                show_stats_panel=True
            )
            print("Violation Detector initialized")
            
            # Initialize BEV
            if processor.enable_bev:
                bev_polygon = processor.road_zone_overlay.get_combined_polygon()
                if bev_polygon is None:
                    bev_polygon = zone_polygons[0]
                
                processor._init_bev_transformer(
                    first_frame=first_frame,
                    zone_polygon=bev_polygon,
                    zone_polygons=zone_polygons,
                    show_progress=True
                )
                
                if processor.bev_transformer is not None:
                    processor.violation_detector.set_bev_transformer(processor.bev_transformer)
            
            total_points = sum(len(z) for z in zone_polygons)
            print(f"Road zones defined: {len(zone_polygons)} zone(s), {total_points} total points")
        
        # Video writer setup
        output_width = width
        output_height = height
        if processor.bev_visualizer is not None:
            bev_scale = height / processor.bev_transformer.bev_height
            bev_display_width = int(processor.bev_transformer.bev_width * bev_scale)
            output_width = width + bev_display_width
        
        async_writer = None
        if output_path:
            ext = output_path.lower().split('.')[-1]
            fourcc_code = 'mp4v'

            # Match codec to selected container while keeping the chosen extension.
            if ext == 'avi':
                fourcc_code = 'XVID'
            else:
                fourcc_code = 'mp4v'

            if writer_class is None:
                raise ValueError("writer_class is required for GUI async writing")

            try:
                async_writer = writer_class(
                    output_path=output_path,
                    fourcc_code=fourcc_code,
                    fps=fps,
                    frame_size=(output_width, output_height),
                )
                async_writer.start()
            except Exception:
                if ext != 'avi':
                    # Fallback codec for systems where mp4v cannot be opened.
                    async_writer = writer_class(
                        output_path=output_path,
                        fourcc_code='XVID',
                        fps=fps,
                        frame_size=(output_width, output_height),
                    )
                    async_writer.start()
                else:
                    raise

            print(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        inferred_frame_count = 0
        benchmark_every = 120
        timing_totals = {
            "inference": 0.0,
            "tracking": 0.0,
            "violations": 0.0,
            "bev": 0.0,
            "ui_emit": 0.0,
            "write": 0.0,
            "total": 0.0,
        }
        cached_detections = None
        last_preview_emit_ts = 0.0
        print("Processing...")
        print(f"  Tracker visualization: boxes={processor.show_boxes}, labels={processor.show_labels}, traces={processor.show_traces}")
        print(f"  Class filter: {processor.classes if processor.classes else 'None (detect all)'}")
        
        try:
            while self._is_running:
                loop_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break
                
                should_process_frame = (
                    processor.skip_frames <= 0
                    or frame_count % (processor.skip_frames + 1) == 0
                    or frame_count == 0
                )

                if should_process_frame:
                    inferred_frame_count += 1
                    infer_start = time.perf_counter()
                    cached_detections = processor.infer_detections(frame)
                    timing_totals["inference"] += time.perf_counter() - infer_start
                else:
                    if cached_detections is None:
                        from supervision import Detections
                        cached_detections = Detections.empty()

                track_start = time.perf_counter()
                annotated_frame, tracked_detections = processor.track_with_detections(
                    frame,
                    cached_detections,
                )
                timing_totals["tracking"] += time.perf_counter() - track_start

                stage_start = time.perf_counter()
                if processor.violation_detector is not None and len(tracked_detections) > 0:
                    processor._current_violations = processor.violation_detector.update(
                        detections=tracked_detections,
                        class_names=processor.model_names,
                        frame_number=frame_count
                    )
                else:
                    processor._current_violations = {}

                if processor.violation_visualizer is not None and len(tracked_detections) > 0:
                    annotated_frame = processor.violation_visualizer.draw_violations(
                        frame=annotated_frame,
                        detections=tracked_detections,
                        current_violations=processor._current_violations,
                        frame_number=frame_count,
                        copy_frame=False,
                    )
                timing_totals["violations"] += time.perf_counter() - stage_start
                
                # Add frame info overlay with FPS
                proc_fps = processor.fps_counter.avg_fps
                info_text = f"Frame: {frame_count}/{total_frames} | Tracks: {len(tracked_detections)} | FPS: {proc_fps:.1f}"
                cv2.putText(annotated_frame, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Create Bird's Eye View if enabled
                stage_start = time.perf_counter()
                display_frame = annotated_frame
                if processor.bev_visualizer is not None:
                    bev_frame = processor.bev_visualizer.draw(
                        detections=tracked_detections,
                        class_names=processor.model_names,
                        show_ids=True,
                        show_labels=True,
                    )
                    display_frame = create_combined_view(
                        camera_frame=annotated_frame,
                        bev_frame=bev_frame,
                        layout="horizontal"
                    )
                timing_totals["bev"] += time.perf_counter() - stage_start
                
                # Throttle preview updates so heavy UI conversion does not block processing throughput.
                stage_start = time.perf_counter()
                now = time.perf_counter()
                if frame_count == 0 or (now - last_preview_emit_ts) >= self._preview_interval_s:
                    self.frame_processed.emit(display_frame)
                    last_preview_emit_ts = now

                if frame_count % 2 == 0 or (frame_count + 1) >= total_frames:
                    self.progress_updated.emit(frame_count + 1, total_frames)
                timing_totals["ui_emit"] += time.perf_counter() - stage_start
                
                # Save frame
                stage_start = time.perf_counter()
                if async_writer:
                    async_writer.write(display_frame)
                timing_totals["write"] += time.perf_counter() - stage_start
                
                frame_count += 1
                
                # Update FPS counter
                processor.fps_counter.tick()

                timing_totals["total"] += time.perf_counter() - loop_start

                if frame_count % benchmark_every == 0:
                    avg_total = timing_totals["total"] / frame_count * 1000.0
                    avg_infer = timing_totals["inference"] / frame_count * 1000.0
                    avg_track = timing_totals["tracking"] / frame_count * 1000.0
                    avg_viol = timing_totals["violations"] / frame_count * 1000.0
                    avg_bev = timing_totals["bev"] / frame_count * 1000.0
                    avg_ui = timing_totals["ui_emit"] / frame_count * 1000.0
                    avg_write = timing_totals["write"] / frame_count * 1000.0
                    print(
                        "[Perf] "
                        f"frames={frame_count} total={avg_total:.2f}ms "
                        f"infer={avg_infer:.2f}ms track={avg_track:.2f}ms violations={avg_viol:.2f}ms "
                        f"bev={avg_bev:.2f}ms ui={avg_ui:.2f}ms write={avg_write:.2f}ms "
                        f"inferred={inferred_frame_count}"
                    )
                
        finally:
            cap.release()
            if async_writer:
                async_writer.close()
            print(f"Processed {frame_count} frames")
            print(f"Inference ran on {inferred_frame_count} frames (skip={processor.skip_frames})")
            if async_writer and async_writer.dropped_frames > 0:
                print(f"Dropped frames while writing: {async_writer.dropped_frames}")
            if frame_count > 0:
                print(
                    "[Perf] Final averages: "
                    f"total={timing_totals['total'] / frame_count * 1000.0:.2f}ms, "
                    f"infer={timing_totals['inference'] / frame_count * 1000.0:.2f}ms, "
                    f"track={timing_totals['tracking'] / frame_count * 1000.0:.2f}ms, "
                    f"violations={timing_totals['violations'] / frame_count * 1000.0:.2f}ms, "
                    f"bev={timing_totals['bev'] / frame_count * 1000.0:.2f}ms, "
                    f"ui={timing_totals['ui_emit'] / frame_count * 1000.0:.2f}ms, "
                    f"write={timing_totals['write'] / frame_count * 1000.0:.2f}ms"
                )
            
    def _create_args(self):
        """Tạo object args từ config"""
        
        class Args:
            pass
            
        args = Args()
        
        # Source
        if self._config.source_config:
            args.source = self._config.source_config.source_type.value
            args.input = self._config.source_config.path
        else:
            args.source = "video"
            args.input = "input.mp4"
            
        # Processing config
        pc = self._config.processing_config or ProcessingConfig()
        
        args.model = pc.model_path
        args.device = pc.device
        args.img_size = pc.img_size
        
        args.conf_thres = pc.conf_threshold
        args.iou_thres = pc.iou_threshold
        args.classes = pc.classes if pc.classes else None
        
        args.max_age = pc.max_age
        args.trace_length = pc.trace_length
        args.skip_frames = pc.skip_frames
        args.min_violation_frames = pc.min_violation_frames
        
        args.show_boxes = pc.show_boxes
        args.show_labels = pc.show_labels
        args.show_traces = pc.show_traces
        
        args.enable_bev = pc.enable_bev
        args.bev_width = pc.bev_width
        args.bev_height = pc.bev_height
        args.bev_method = pc.bev_method
        args.camera_height = pc.camera_height
        
        args.output = pc.output_path
        args.save_video = pc.save_video
        args.display = True
        
        args.select_zone = False
        
        return args
        
    def stop(self):
        """Dừng xử lý"""
        self._is_running = False


class MainWindow(QMainWindow):
    """
    Cửa sổ chính của ứng dụng
    
    Flow:
    1. Source Selection - Chọn nguồn video
    2. Config - Cấu hình tham số
    3. Zone Selection - Chọn valid zones
    4. Processing - Xử lý video
    """
    
    def __init__(self):
        super().__init__()
        
        self._app_config = AppConfig()
        self._current_state = AppState.SOURCE_SELECTION
        self._processing_thread: Optional[ProcessingThread] = None
        self._logo_original_pixmap: Optional[QPixmap] = None
        self._logo_label: Optional[QLabel] = None
        
        self._setup_ui()
        self._connect_signals()
        self._update_state(AppState.SOURCE_SELECTION)
        
    def _setup_ui(self):
        """Thiết lập giao diện chính"""
        self.setWindowTitle("Highway Detection System")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = self._create_header()
        main_layout.addWidget(header)
        
        # Progress indicator
        self._progress_widget = self._create_progress_indicator()
        main_layout.addWidget(self._progress_widget)
        
        # Stacked widget for different pages
        self._stacked_widget = QStackedWidget()
        
        # Page 1: Source Selection
        self._source_selector = SourceSelector()
        self._stacked_widget.addWidget(self._source_selector)
        
        # Page 2: Configuration
        self._config_panel = ConfigPanel()
        self._stacked_widget.addWidget(self._config_panel)
        
        # Page 3: Zone Selection
        self._zone_selector = ZoneSelectorWidget()
        self._stacked_widget.addWidget(self._zone_selector)
        
        # Page 4: Processing
        self._processing_page = self._create_processing_page()
        self._stacked_widget.addWidget(self._processing_page)
        
        main_layout.addWidget(self._stacked_widget, 1)
        
        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Sẵn sàng")
        
    def _create_header(self) -> QWidget:
        """Tạo header"""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #2196F3;
                padding: 12px;
            }
        """)
        header.setFixedHeight(90)
        
        layout = QHBoxLayout(header)
        
        # Logo
        self._logo_label = QLabel()
        self._logo_label.setFixedSize(180, 64)
        self._logo_label.setAlignment(Qt.AlignCenter)
        self._logo_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.8);
                border-radius: 10px;
                padding: 4px;
            }
        """)
        layout.addWidget(self._logo_label)

        logo_path = self._resolve_logo_path()
        if logo_path:
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                self._logo_original_pixmap = pixmap
                self._update_logo_pixmap()

        layout.addSpacing(14)

        # Title
        title = QLabel("Highway Detection System")
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 20px;
                font-weight: bold;
            }
        """)
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Version
        version = QLabel(get_display_version())
        version.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        layout.addWidget(version)
        
        return header

    def _resolve_logo_path(self) -> Optional[str]:
        """Tìm đường dẫn logo phù hợp cho cả dev mode và frozen mode."""
        candidates = []

        if getattr(sys, 'frozen', False):
            exe_dir = os.path.dirname(sys.executable)
            candidates.append(os.path.join(exe_dir, 'assets', 'logo.png'))

            meipass = getattr(sys, '_MEIPASS', None)
            if meipass:
                candidates.append(os.path.join(meipass, 'assets', 'logo.png'))

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.join(project_root, 'assets', 'logo.png'))

        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def _update_logo_pixmap(self):
        """Scale logo theo kích thước khung hiển thị, luôn giữ tỉ lệ."""
        if not self._logo_label or not self._logo_original_pixmap:
            return

        target_size = self._logo_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        scaled = self._logo_original_pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self._logo_label.setPixmap(scaled)
        
    def _create_progress_indicator(self) -> QWidget:
        """Tạo progress indicator hiển thị các bước"""
        widget = QFrame()
        widget.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border-bottom: 1px solid #ddd;
            }
        """)
        widget.setFixedHeight(50)
        
        layout = QHBoxLayout(widget)
        layout.setSpacing(0)
        
        self._step_labels = []
        steps = [
            ("1. Chọn Nguồn", AppState.SOURCE_SELECTION),
            ("2. Cấu Hình", AppState.CONFIG),
            ("3. Chọn Zone", AppState.ZONE_SELECTION),
            ("4. Xử Lý", AppState.PROCESSING)
        ]
        
        for i, (text, state) in enumerate(steps):
            step_widget = QWidget()
            step_layout = QHBoxLayout(step_widget)
            step_layout.setContentsMargins(10, 0, 10, 0)
            
            # Step label
            label = QLabel(text)
            label.setAlignment(Qt.AlignCenter)
            step_layout.addWidget(label)
            
            # Arrow (except last)
            if i < len(steps) - 1:
                arrow = QLabel("→")
                arrow.setStyleSheet("color: #999;")
                step_layout.addWidget(arrow)
                
            layout.addWidget(step_widget)
            self._step_labels.append((label, state))
            
        return widget
        
    def _create_processing_page(self) -> QWidget:
        """Tạo trang xử lý"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Processing info
        self._processing_label = QLabel("Đang xử lý video...")
        self._processing_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self._processing_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._processing_label)
        
        # Video display area - main content
        self._video_display_label = QLabel("Đang tải...")
        self._video_display_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 2px solid #333333;
                border-radius: 8px;
            }
        """)
        self._video_display_label.setAlignment(Qt.AlignCenter)
        self._video_display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_display_label.setMinimumSize(640, 400)
        layout.addWidget(self._video_display_label, 1)
        
        # Bottom controls
        controls_layout = QHBoxLayout()
        
        # Progress bar
        progress_container = QVBoxLayout()
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimumHeight(20)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)
        progress_container.addWidget(self._progress_bar)
        
        # Progress text
        self._progress_text = QLabel("0 / 0 frames")
        self._progress_text.setStyleSheet("color: #888; font-size: 12px;")
        self._progress_text.setAlignment(Qt.AlignCenter)
        progress_container.addWidget(self._progress_text)
        
        controls_layout.addLayout(progress_container, 1)
        
        controls_layout.addSpacing(20)
        
        # Stop button
        self._stop_btn = QPushButton("⏹️ Dừng Xử Lý")
        self._stop_btn.setMinimumHeight(45)
        self._stop_btn.setMinimumWidth(150)
        self._stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        controls_layout.addWidget(self._stop_btn)
        
        layout.addLayout(controls_layout)
        
        return widget
        
    def _connect_signals(self):
        """Kết nối signals"""
        # Source selector
        self._source_selector.source_selected.connect(self._on_source_selected)
        
        # Config panel
        self._config_panel.config_confirmed.connect(self._on_config_confirmed)
        self._config_panel._back_btn.clicked.connect(
            lambda: self._update_state(AppState.SOURCE_SELECTION)
        )
        
        # Zone selector
        self._zone_selector.zones_confirmed.connect(self._on_zones_confirmed)
        self._zone_selector.back_requested.connect(
            lambda: self._update_state(AppState.CONFIG)
        )
        
        # Processing
        self._stop_btn.clicked.connect(self._stop_processing)
        
    def _update_state(self, new_state: AppState):
        """Cập nhật trạng thái ứng dụng"""
        self._current_state = new_state
        
        # Update stacked widget
        state_to_page = {
            AppState.SOURCE_SELECTION: 0,
            AppState.CONFIG: 1,
            AppState.ZONE_SELECTION: 2,
            AppState.PROCESSING: 3,
        }
        
        if new_state in state_to_page:
            self._stacked_widget.setCurrentIndex(state_to_page[new_state])
            
        # Update progress indicator
        self._update_progress_indicator()
        
        # Update status bar
        state_messages = {
            AppState.SOURCE_SELECTION: "Chọn nguồn video đầu vào",
            AppState.CONFIG: "Cấu hình các tham số xử lý",
            AppState.ZONE_SELECTION: "Chọn vùng đường hợp lệ",
            AppState.PROCESSING: "Đang xử lý video...",
            AppState.COMPLETED: "Xử lý hoàn tất!"
        }
        self._status_bar.showMessage(state_messages.get(new_state, ""))
        
    def _update_progress_indicator(self):
        """Cập nhật progress indicator"""
        current_state = self._current_state
        
        for label, state in self._step_labels:
            if state.value < current_state.value:
                # Completed step
                label.setStyleSheet("""
                    QLabel {
                        color: #4CAF50;
                        font-weight: bold;
                    }
                """)
            elif state == current_state:
                # Current step
                label.setStyleSheet("""
                    QLabel {
                        color: #2196F3;
                        font-weight: bold;
                    }
                """)
            else:
                # Future step
                label.setStyleSheet("""
                    QLabel {
                        color: #999;
                    }
                """)
                
    def _on_source_selected(self, source_config: SourceConfig):
        """Xử lý khi nguồn được chọn"""
        self._app_config.source_config = source_config
        self._config_panel.set_output_preferences(
            output_format=source_config.output_format,
            source_path=source_config.path,
            output_path=source_config.output_path,
        )
        self._status_bar.showMessage(f"Đã chọn: {source_config.path}")
        
        # Move to config
        self._update_state(AppState.CONFIG)
        
    def _on_config_confirmed(self, processing_config: ProcessingConfig):
        """Xử lý khi cấu hình được xác nhận"""
        self._app_config.processing_config = processing_config

        self._zone_selector.set_bev_preview_config(
            bev_method=processing_config.bev_method,
            camera_height=processing_config.camera_height,
            bev_width=processing_config.bev_width,
            bev_height=processing_config.bev_height,
        )
        
        # Load first frame for zone selection
        if self._app_config.source_config:
            video_path = self._app_config.source_config.path
            cap = cv2.VideoCapture(video_path)
            
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    self._zone_selector.set_frame(frame)
                    self._update_state(AppState.ZONE_SELECTION)
                else:
                    QMessageBox.warning(
                        self,
                        "Lỗi",
                        "Không thể đọc frame đầu tiên từ video!"
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Lỗi",
                    "Không thể mở video!"
                )
                
    def _on_zones_confirmed(self, zones: List[np.ndarray]):
        """Xử lý khi zones được xác nhận"""
        self._app_config.zones = zones
        
        # Start processing
        self._start_processing()
        
    def _start_processing(self):
        """Bắt đầu xử lý video"""
        self._update_state(AppState.PROCESSING)
        
        # Reset video display
        self._video_display_label.setText("Đang khởi tạo...")
        
        # Get pre-loaded model handler from config panel
        model_handler = self._config_panel.get_model_handler()
        
        # Create and start processing thread
        self._processing_thread = ProcessingThread(self._app_config, model_handler, self)
        self._processing_thread.progress_updated.connect(self._on_progress_updated)
        self._processing_thread.frame_processed.connect(self._on_frame_processed)
        self._processing_thread.processing_completed.connect(self._on_processing_completed)
        self._processing_thread.error_occurred.connect(self._on_processing_error)
        self._processing_thread.start()
        
    def _stop_processing(self):
        """Dừng xử lý"""
        if self._processing_thread:
            self._processing_thread.stop()
            self._processing_thread.wait()
            self._processing_thread = None
        
        # Hiển thị thông báo
        QMessageBox.information(
            self,
            "Đã Dừng",
            "Quá trình xử lý video đã được dừng lại."
        )
        
        # Reset config và redirect về màn hình chính
        self._app_config = AppConfig()
        self._status_bar.showMessage("Đã dừng xử lý - Quay về màn hình chính")
        self._update_state(AppState.SOURCE_SELECTION)
        
    def _on_progress_updated(self, current: int, total: int):
        """Cập nhật tiến trình"""
        if total > 0:
            self._progress_bar.setMaximum(total)
            self._progress_bar.setValue(current)
            self._progress_text.setText(f"{current} / {total} frames")
            percent = int(100 * current / total)
            self._processing_label.setText(f"Đang xử lý video... {percent}%")
    
    def _on_frame_processed(self, frame: np.ndarray):
        """Hiển thị frame đã xử lý lên GUI"""
        if frame is None:
            return
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Scale to fit label while maintaining aspect ratio
            label_size = self._video_display_label.size()
            h, w = rgb_frame.shape[:2]
            
            scale = min(label_size.width() / w, label_size.height() / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if new_w > 0 and new_h > 0:
                if new_w == w and new_h == h:
                    scaled_frame = rgb_frame
                else:
                    scaled_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Create QImage and QPixmap
                # Note: .copy() is required because QImage doesn't copy data from numpy array
                # Without copy(), data may be garbage collected before display
                h, w, ch = scaled_frame.shape
                bytes_per_line = ch * w
                q_image = QImage(scaled_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(q_image)
                
                self._video_display_label.setPixmap(pixmap)
        except Exception as e:
            print(f"Error displaying frame: {e}")
            
    def _on_processing_completed(self):
        """Xử lý khi hoàn thành"""
        self._status_bar.showMessage("Xử lý hoàn tất!")
        self._processing_label.setText("✅ Xử lý hoàn tất!")
        
        # Hiển thị thông báo hoàn thành
        QMessageBox.information(
            self,
            "Hoàn Thành",
            "Xử lý video đã hoàn tất!\nBạn có thể chọn video khác để xử lý."
        )
        
        # Reset config và tự động redirect về màn hình chính
        self._app_config = AppConfig()
        self._status_bar.showMessage("Xử lý hoàn tất - Quay về màn hình chính")
        self._update_state(AppState.SOURCE_SELECTION)
            
    def _on_processing_error(self, error_msg: str):
        """Xử lý khi có lỗi"""
        QMessageBox.critical(
            self,
            "Lỗi Xử Lý",
            f"Đã xảy ra lỗi:\n{error_msg}\n\nQuay về màn hình chính."
        )
        
        # Reset config và redirect về màn hình chính
        self._app_config = AppConfig()
        self._status_bar.showMessage("Lỗi xử lý - Quay về màn hình chính")
        self._update_state(AppState.SOURCE_SELECTION)
        
    def closeEvent(self, event):
        """Xử lý khi đóng cửa sổ"""
        if self._processing_thread and self._processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Xác Nhận Thoát",
                "Video đang được xử lý. Bạn có chắc muốn thoát?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self._processing_thread.stop()
                self._processing_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def resizeEvent(self, event):
        """Đảm bảo logo vẫn nét và vừa vặn khi thay đổi kích thước cửa sổ."""
        super().resizeEvent(event)
        self._update_logo_pixmap()


def run_gui():
    """Chạy ứng dụng GUI"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Apply custom stylesheet
    apply_stylesheet(app)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    return app.exec_()


if __name__ == "__main__":
    sys.exit(run_gui())
