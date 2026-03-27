"""
Zone Selector Widget
Widget cho phép người dùng chọn valid zone trên video frame
Với tính năng gợi ý đường dựa trên lane detection
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QListWidget, QListWidgetItem,
    QSplitter, QMessageBox, QSizePolicy, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import (
    QImage, QPixmap, QMouseEvent, QKeyEvent
)

# Import LaneLineSuggestion từ road_zone module
from lane_mapping.road_zone import LaneLineSuggestion


@dataclass
class Zone:
    """Đại diện cho một zone"""
    points: List[Tuple[int, int]] = field(default_factory=list)
    name: str = ""
    color: Tuple[int, int, int] = (0, 255, 0)
    
    def is_valid(self) -> bool:
        """Kiểm tra zone có hợp lệ không (ít nhất 3 điểm)"""
        return len(self.points) >= 3
    
    def to_numpy(self) -> np.ndarray:
        """Chuyển đổi thành numpy array"""
        return np.array(self.points, dtype=np.int32)


class ZoneCanvas(QLabel):
    """
    Canvas để vẽ và chỉnh sửa zones
    """
    
    point_added = pyqtSignal(tuple)  # (x, y)
    point_removed = pyqtSignal()
    zone_completed = pyqtSignal()
    suggestion_added = pyqtSignal()  # Signal khi thêm điểm gợi ý
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._original_frame: Optional[np.ndarray] = None
        self._display_scale: float = 1.0
        self._offset_x: int = 0
        self._offset_y: int = 0
        
        # Zone data
        self._zones: List[Zone] = []
        self._current_points: List[Tuple[int, int]] = []
        self._active_zone_index: int = -1
        
        # Mouse tracking
        self._mouse_pos: Tuple[int, int] = (0, 0)
        
        # Lane suggestion feature
        self._lane_suggester: Optional[LaneLineSuggestion] = None
        self._current_suggestion: List[Tuple[int, int]] = []
        self._enable_suggestion: bool = True
        self._show_lane_detection: bool = False
        self._suggestion_color = (255, 0, 255)  # Magenta
        
        # Colors for multiple zones
        self._zone_colors = [
            (0, 255, 0),    # Green
            (255, 165, 0),  # Orange
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 0, 0),    # Blue
            (0, 165, 255),  # Orange variant
        ]
        
        # Settings
        self._point_radius = 6
        self._line_thickness = 2
        self._zone_alpha = 0.3
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard input
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #333333;
            }
        """)
        
    def set_frame(self, frame: np.ndarray):
        """Đặt frame để hiển thị và khởi tạo lane detection"""
        self._original_frame = frame.copy()
        self._lane_suggester = None

        # Khởi tạo lane suggester và detect lanes ngay để tránh giật khi tương tác.
        if self._enable_suggestion:
            self._lane_suggester = LaneLineSuggestion(
                canny_low=50,
                canny_high=150,
                hough_threshold=50,
                min_line_length=50,
                max_line_gap=30,
                suggestion_distance=30
            )
            self._lane_suggester.detect_lanes(frame)
        
        self._update_display()
        
    def _update_display(self):
        """Cập nhật hiển thị"""
        if self._original_frame is None:
            return
            
        # Create display frame with zones drawn
        display_frame = self._draw_zones_on_frame(self._original_frame.copy())
        
        # Scale to fit widget
        widget_size = self.size()
        h, w = display_frame.shape[:2]
        
        self._display_scale = min(widget_size.width() / w, widget_size.height() / h)
        new_w = int(w * self._display_scale)
        new_h = int(h * self._display_scale)
        
        # Calculate offset for centering
        self._offset_x = (widget_size.width() - new_w) // 2
        self._offset_y = (widget_size.height() - new_h) // 2
        
        if new_w > 0 and new_h > 0:
            scaled_frame = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
            
            # Create pixmap
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            self.setPixmap(pixmap)
            
    def _draw_zones_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ tất cả zones lên frame"""
        # Vẽ lane detection debug nếu bật
        if self._show_lane_detection and self._lane_suggester:
            frame = self._lane_suggester.draw_detected_lanes(frame, 
                                                             color=(80, 80, 90), 
                                                             thickness=1)
        
        # Draw completed zones
        for i, zone in enumerate(self._zones):
            if zone.is_valid():
                color = self._zone_colors[i % len(self._zone_colors)]
                self._draw_zone(frame, zone.points, color, filled=True)
        
        # Draw suggestion path nếu có
        if self._current_suggestion and len(self._current_suggestion) >= 2:
            suggestion_pts = np.array(self._current_suggestion, dtype=np.int32)
            # Glow effect
            cv2.polylines(frame, [suggestion_pts], isClosed=False, 
                         color=(180, 80, 220), thickness=4, lineType=cv2.LINE_AA)
            cv2.polylines(frame, [suggestion_pts], isClosed=False, 
                         color=self._suggestion_color, thickness=2, lineType=cv2.LINE_AA)
            # Điểm gợi ý nhỏ
            for pt in self._current_suggestion[::4]:
                cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
                
        elif self._current_suggestion and len(self._current_suggestion) == 1:
            # Vẽ điểm gợi ý đơn (snap point)
            pt = self._current_suggestion[0]
            cv2.circle(frame, pt, self._point_radius + 6, self._suggestion_color, 1, cv2.LINE_AA)
            cv2.circle(frame, pt, self._point_radius + 3, self._suggestion_color, 1, cv2.LINE_AA)
            cv2.circle(frame, pt, self._point_radius, self._suggestion_color, -1, cv2.LINE_AA)
                
        # Draw current zone being edited
        if self._current_points:
            color = self._zone_colors[len(self._zones) % len(self._zone_colors)]
            self._draw_zone(frame, self._current_points, color, filled=False)
            
            # Draw preview line to mouse position
            if len(self._current_points) > 0:
                last_point = self._current_points[-1]
                cv2.line(frame, last_point, self._mouse_pos, 
                        (255, 255, 0), 1, cv2.LINE_AA)
        
        # Vẽ hướng dẫn phím tắt
        frame = self._draw_keyboard_hints(frame)
                
        return frame
    
    def _draw_keyboard_hints(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ hướng dẫn phím tắt lên frame"""
        h, w = frame.shape[:2]
        
        hints = [
            ("S", "Add suggested path", (130, 230, 230)),
            ("L", f"Show lanes: {'ON' if self._show_lane_detection else 'OFF'}", 
             (130, 230, 130) if self._show_lane_detection else (200, 200, 200)),
        ]
        
        # Panel background
        panel_x, panel_y = 10, h - 80
        panel_w, panel_h = 200, 65
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 35), -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        
        # Title
        cv2.putText(frame, "Keyboard Shortcuts", (panel_x + 10, panel_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 185), 1, cv2.LINE_AA)
        
        y_pos = panel_y + 38
        for key, desc, color in hints:
            # Key badge
            cv2.rectangle(frame, (panel_x + 10, y_pos - 12), (panel_x + 30, y_pos + 4), (45, 45, 50), -1)
            cv2.putText(frame, key, (panel_x + 15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 215), 1, cv2.LINE_AA)
            # Description
            cv2.putText(frame, desc, (panel_x + 38, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            y_pos += 22
        
        return frame
        
    def _draw_zone(self, frame: np.ndarray, points: List[Tuple[int, int]], 
                   color: Tuple[int, int, int], filled: bool = True):
        """Vẽ một zone lên frame"""
        if not points:
            return
            
        pts = np.array(points, dtype=np.int32)
        
        if filled and len(points) >= 3:
            # Draw filled polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, self._zone_alpha, frame, 1 - self._zone_alpha, 0, frame)
            
        # Draw polygon outline
        if len(points) >= 2:
            cv2.polylines(frame, [pts], isClosed=(len(points) >= 3 and filled), 
                         color=color, thickness=self._line_thickness)
            
        # Draw points
        for i, point in enumerate(points):
            # Outer circle
            cv2.circle(frame, point, self._point_radius + 2, (0, 0, 0), -1)
            # Inner circle
            cv2.circle(frame, point, self._point_radius, color, -1)
            # Point number
            cv2.putText(frame, str(i + 1), 
                       (point[0] + self._point_radius + 2, point[1] - self._point_radius),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                       
    def _widget_to_frame_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Chuyển đổi tọa độ widget sang tọa độ frame"""
        if self._display_scale == 0:
            return (0, 0)
            
        # Remove offset
        x -= self._offset_x
        y -= self._offset_y
        
        # Scale back to original coordinates.
        # Dùng round để giảm sai số tích lũy do truncate khi vẽ/resize nhiều lần.
        frame_x = int(round(x / self._display_scale))
        frame_y = int(round(y / self._display_scale))
        
        # Clamp to frame bounds
        if self._original_frame is not None:
            h, w = self._original_frame.shape[:2]
            frame_x = max(0, min(frame_x, w - 1))
            frame_y = max(0, min(frame_y, h - 1))
            
        return (frame_x, frame_y)
        
    def mousePressEvent(self, event: QMouseEvent):
        """Xử lý click chuột"""
        if self._original_frame is None:
            return
            
        frame_coords = self._widget_to_frame_coords(event.x(), event.y())
        
        if event.button() == Qt.LeftButton:
            # Add point
            self._current_points.append(frame_coords)
            self.point_added.emit(frame_coords)
            self._update_display()
        elif event.button() == Qt.RightButton:
            # Remove last point
            if self._current_points:
                self._current_points.pop()
                self.point_removed.emit()
                self._update_display()
                
    def mouseMoveEvent(self, event: QMouseEvent):
        """Xử lý di chuột"""
        if self._original_frame is None:
            return
            
        self._mouse_pos = self._widget_to_frame_coords(event.x(), event.y())
        
        # Cập nhật đường gợi ý
        if self._enable_suggestion:
            self._update_suggestion(self._mouse_pos)
        
        self._update_display()
    
    def _update_suggestion(self, mouse_pos: Tuple[int, int]):
        """Cập nhật đường gợi ý dựa trên vị trí chuột"""
        if not self._enable_suggestion or self._lane_suggester is None:
            self._current_suggestion = []
            return
        
        if len(self._current_points) >= 1:
            # Có điểm trước đó - gợi ý theo hướng đang vẽ
            last_point = self._current_points[-1]
            self._current_suggestion = self._lane_suggester.get_extended_suggestion(
                last_point, mouse_pos, num_points=40
            )
        else:
            # Chưa có điểm - gợi ý điểm gần nhất (snap point)
            nearest = self._lane_suggester.find_nearest_edge_point(mouse_pos)
            if nearest:
                self._current_suggestion = [nearest]
            else:
                self._current_suggestion = []
    
    def _add_suggestion_points(self):
        """Thêm các điểm gợi ý vào polygon"""
        if not self._current_suggestion:
            return False
            
        # Lọc các điểm để tránh trùng lặp và quá gần nhau
        min_distance = 15  # Khoảng cách tối thiểu giữa các điểm
        added_count = 0
        
        for pt in self._current_suggestion:
            if not self._current_points:
                self._current_points.append(pt)
                added_count += 1
            else:
                last_pt = self._current_points[-1]
                dist = np.sqrt((pt[0] - last_pt[0])**2 + (pt[1] - last_pt[1])**2)
                if dist >= min_distance:
                    self._current_points.append(pt)
                    added_count += 1
        
        self._current_suggestion = []
        
        if added_count > 0:
            self.suggestion_added.emit()
            self._update_display()
            return True
        return False
    
    def keyPressEvent(self, event: QKeyEvent):
        """Xử lý phím bấm"""
        key = event.key()
        
        if key == Qt.Key_S:
            # Thêm các điểm gợi ý
            self._add_suggestion_points()
        elif key == Qt.Key_L:
            # Bật/tắt hiển thị lane detection
            self._show_lane_detection = not self._show_lane_detection
            self._update_display()
        else:
            super().keyPressEvent(event)
        
    def resizeEvent(self, event):
        """Xử lý resize"""
        super().resizeEvent(event)
        self._update_display()
        
    def complete_current_zone(self) -> bool:
        """Hoàn thành zone hiện tại"""
        if len(self._current_points) < 3:
            return False
            
        zone = Zone(
            points=self._current_points.copy(),
            name=f"Zone {len(self._zones) + 1}",
            color=self._zone_colors[len(self._zones) % len(self._zone_colors)]
        )
        self._zones.append(zone)
        self._current_points = []
        self.zone_completed.emit()
        self._update_display()
        return True
        
    def remove_zone(self, index: int):
        """Xóa zone theo index"""
        if 0 <= index < len(self._zones):
            self._zones.pop(index)
            self._update_display()
            
    def clear_current(self):
        """Xóa zone đang vẽ"""
        self._current_points = []
        self._update_display()
        
    def clear_all(self):
        """Xóa tất cả zones"""
        self._zones = []
        self._current_points = []
        self._update_display()
        
    def get_zones(self) -> List[Zone]:
        """Lấy danh sách zones"""
        return self._zones.copy()
        
    def get_zones_as_numpy(self) -> List[np.ndarray]:
        """Lấy zones dưới dạng numpy arrays"""
        return [zone.to_numpy() for zone in self._zones if zone.is_valid()]
    
    def get_current_points(self) -> List[Tuple[int, int]]:
        """Lấy các điểm đang vẽ hiện tại"""
        return self._current_points.copy()
        
    def get_current_points_count(self) -> int:
        """Lấy số điểm đang vẽ"""
        return len(self._current_points)


class ZoneSelectorWidget(QWidget):
    """
    Widget hoàn chỉnh cho việc chọn zones
    
    Signals:
        zones_confirmed: Phát ra khi zones được xác nhận
        cancelled: Phát ra khi hủy bỏ
    """
    
    zones_confirmed = pyqtSignal(list)  # List of numpy arrays
    cancelled = pyqtSignal()
    back_requested = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._preview_bev_method: str = "ipm"
        self._preview_camera_height: float = 1.5
        self._preview_bev_width: int = 200
        self._preview_bev_height: int = 300
        self._bev_preview_base_image: Optional[np.ndarray] = None
        self._bev_preview_transformer = None
        self._bev_preview_method_used: str = ""
        self._bev_preview_cache_key: Optional[Tuple] = None
        self._setup_ui()
        self._connect_signals()

    def set_bev_preview_config(
        self,
        bev_method: str = "ipm",
        camera_height: float = 1.5,
        bev_width: int = 400,
        bev_height: int = 600,
    ):
        """Đồng bộ cấu hình BEV preview với cấu hình runtime."""
        method = (bev_method or "ipm").lower()
        self._preview_bev_method = method if method in {"ipm", "homography"} else "ipm"
        self._preview_camera_height = max(0.1, float(camera_height))
        self._preview_bev_width = max(120, int(bev_width))
        self._preview_bev_height = max(180, int(bev_height))
        self._reset_bev_preview_cache()
        self._preload_bev_preview_background()
        self._update_bev_preview()
        
    def _setup_ui(self):
        """Thiết lập giao diện"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Title  
        title_label = QLabel("Chọn Vùng Đường Hợp Lệ (Valid Zone)")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Instructions
        instructions = QLabel(
            "🖱️ <b>Chuột trái:</b> Thêm điểm | "
            "🖱️ <b>Chuột phải:</b> Xóa điểm cuối | "
            "⌨️ <b>S:</b> Thêm gợi ý | "
            "⌨️ <b>L:</b> Bật/tắt lane detection"
        )
        instructions.setStyleSheet("color: #666666; padding: 5px;")
        instructions.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(instructions)
        
        # Splitter for canvas and controls
        splitter = QSplitter(Qt.Horizontal)
        
        # Canvas
        self._canvas = ZoneCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setMinimumSize(500, 400)
        splitter.addWidget(self._canvas)
        
        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 5, 0)
        
        # Zone list
        zone_group = QGroupBox("Danh Sách Zones")
        zone_layout = QVBoxLayout(zone_group)
        
        self._zone_list = QListWidget()
        self._zone_list.setMaximumHeight(150)
        zone_layout.addWidget(self._zone_list)
        
        # Zone buttons
        zone_btn_layout = QHBoxLayout()
        
        self._complete_zone_btn = QPushButton("✓ Hoàn Thành Zone")
        self._complete_zone_btn.setEnabled(False)
        self._complete_zone_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        zone_btn_layout.addWidget(self._complete_zone_btn)
        
        zone_layout.addLayout(zone_btn_layout)
        
        zone_btn_layout2 = QHBoxLayout()
        
        self._delete_zone_btn = QPushButton("🗑️ Xóa Zone")
        self._delete_zone_btn.setEnabled(False)
        zone_btn_layout2.addWidget(self._delete_zone_btn)
        
        self._clear_current_btn = QPushButton("↩️ Xóa Đang Vẽ")
        zone_btn_layout2.addWidget(self._clear_current_btn)
        
        zone_layout.addLayout(zone_btn_layout2)
        
        right_layout.addWidget(zone_group)
        
        # Bird's Eye View Preview
        bev_group = QGroupBox("🔍 Bird's Eye View Preview")
        bev_layout = QVBoxLayout(bev_group)
        
        self._bev_preview = QLabel()
        self._bev_preview.setMinimumSize(180, 250)
        self._bev_preview.setMaximumSize(220, 320)
        self._bev_preview.setAlignment(Qt.AlignCenter)
        self._bev_preview.setStyleSheet("""
            QLabel {
                background-color: #282828;
                border: 1px solid #444444;
                border-radius: 4px;
            }
        """)
        self._bev_preview.setText("Vẽ ít nhất 4 điểm\nđể xem preview")
        bev_layout.addWidget(self._bev_preview)
        
        # Checkbox to enable/disable BEV preview
        self._bev_preview_checkbox = QCheckBox("Tự động cập nhật BEV")
        self._bev_preview_checkbox.setChecked(True)
        self._bev_preview_checkbox.setToolTip("Bật/tắt cập nhật Bird's Eye View khi vẽ zone")
        bev_layout.addWidget(self._bev_preview_checkbox)
        
        right_layout.addWidget(bev_group)
        
        # Current drawing info
        info_group = QGroupBox("Thông Tin")
        info_layout = QVBoxLayout(info_group)
        
        self._points_label = QLabel("Điểm hiện tại: 0")
        info_layout.addWidget(self._points_label)
        
        self._zones_count_label = QLabel("Tổng số zones: 0")
        info_layout.addWidget(self._zones_count_label)
        
        right_layout.addWidget(info_group)
        
        # Tips
        tips_group = QGroupBox("💡 Mẹo")
        tips_layout = QVBoxLayout(tips_group)
        
        tips_text = QLabel(
            "• Vẽ theo chiều kim đồng hồ\n"
            "• Bắt đầu từ góc trên bên trái\n"
            "• Có thể tạo nhiều zones\n"
            "• Nhấn S để thêm gợi ý lane\n"
            "• Nhấn L để xem lane detection\n"
            "• Mỗi zone cần ít nhất 3 điểm"
        )
        tips_text.setStyleSheet("color: #666666; font-size: 11px;")
        tips_layout.addWidget(tips_text)
        
        right_layout.addWidget(tips_group)
        
        right_layout.addStretch()
        
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 250])
        
        main_layout.addWidget(splitter, 1)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self._back_btn = QPushButton("← Quay Lại")
        self._back_btn.setMinimumHeight(40)
        button_layout.addWidget(self._back_btn)
        
        button_layout.addStretch()
        
        self._clear_all_btn = QPushButton("🗑️ Xóa Tất Cả")
        self._clear_all_btn.setMinimumHeight(40)
        button_layout.addWidget(self._clear_all_btn)
        
        self._skip_btn = QPushButton("⏭️ Bỏ Qua Zone")
        self._skip_btn.setMinimumHeight(40)
        self._skip_btn.setToolTip("Tiếp tục mà không cần zone")
        button_layout.addWidget(self._skip_btn)
        
        self._confirm_btn = QPushButton("✓ Xác Nhận & Tiếp Tục")
        self._confirm_btn.setMinimumHeight(40)
        self._confirm_btn.setEnabled(False)
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
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self._confirm_btn)
        
        main_layout.addLayout(button_layout)
        
    def _connect_signals(self):
        """Kết nối signals"""
        # Canvas signals
        self._canvas.point_added.connect(self._on_point_added)
        self._canvas.point_removed.connect(self._on_point_removed)
        self._canvas.zone_completed.connect(self._update_zone_list)
        self._canvas.suggestion_added.connect(self._on_suggestion_added)
        
        # Button signals
        self._complete_zone_btn.clicked.connect(self._complete_current_zone)
        self._delete_zone_btn.clicked.connect(self._delete_selected_zone)
        self._clear_current_btn.clicked.connect(self._clear_current)
        self._clear_all_btn.clicked.connect(self._clear_all)
        self._back_btn.clicked.connect(self.back_requested.emit)
        self._skip_btn.clicked.connect(self._on_skip)
        self._confirm_btn.clicked.connect(self._on_confirm)
        
        # List signals
        self._zone_list.itemSelectionChanged.connect(self._on_zone_selection_changed)
        
        # BEV preview checkbox
        self._bev_preview_checkbox.stateChanged.connect(self._on_bev_checkbox_changed)
    
    def _on_bev_checkbox_changed(self, state):
        """Xử lý khi checkbox BEV preview thay đổi"""
        if state == Qt.Checked:
            self._update_bev_preview()
        else:
            self._bev_preview.setText("Preview tắt\n(Tick checkbox để bật)")
            self._bev_preview.setPixmap(QPixmap())
        
    def set_frame(self, frame: np.ndarray):
        """Đặt frame để chọn zone"""
        self._canvas.set_frame(frame)
        self._reset_bev_preview_cache()
        self._preload_bev_preview_background()
        self._update_bev_preview()

    def _reset_bev_preview_cache(self):
        """Xóa cache preview để buộc preload lại khi cấu hình/frame đổi."""
        self._bev_preview_base_image = None
        self._bev_preview_transformer = None
        self._bev_preview_method_used = ""
        self._bev_preview_cache_key = None

    def _build_default_preview_polygon(self, width: int, height: int) -> np.ndarray:
        """Tạo polygon mặc định để preload BEV trước khi user chọn zone."""
        top_y = int(height * 0.35)
        bottom_y = int(height * 0.92)
        top_left_x = int(width * 0.38)
        top_right_x = int(width * 0.62)
        bottom_left_x = int(width * 0.12)
        bottom_right_x = int(width * 0.88)

        return np.array(
            [
                [top_left_x, top_y],
                [top_right_x, top_y],
                [bottom_right_x, bottom_y],
                [bottom_left_x, bottom_y],
            ],
            dtype=np.int32,
        )

    def _preload_bev_preview_background(self) -> bool:
        """Preload nền BEV để tránh giật ở lần user chọn điểm đầu tiên."""
        frame = self._canvas._original_frame
        if frame is None:
            return False

        h, w = frame.shape[:2]
        cache_key = (
            h,
            w,
            self._preview_bev_method,
            self._preview_camera_height,
            self._preview_bev_width,
            self._preview_bev_height,
        )

        if self._bev_preview_cache_key == cache_key and self._bev_preview_base_image is not None:
            return True

        polygon = self._build_default_preview_polygon(w, h)
        transformer = None
        visualizer = None
        method_used = ""

        from lane_mapping.bird_eye_view import (
            BirdEyeViewTransformer,
            BirdEyeViewVisualizer,
            IPMBirdEyeViewTransformer,
            IPMBirdEyeViewVisualizer,
        )

        if self._preview_bev_method == "ipm":
            try:
                ipm_transformer = IPMBirdEyeViewTransformer(
                    frame_width=w,
                    frame_height=h,
                    camera_height=self._preview_camera_height,
                    bev_width=self._preview_bev_width,
                    bev_height=self._preview_bev_height,
                    roi_polygon=polygon,
                    auto_calibrate=True,
                )
                ipm_transformer.calibrate_from_frame(frame)
                transformer = ipm_transformer
                visualizer = IPMBirdEyeViewVisualizer(
                    transformer=transformer,
                    bg_color=(30, 30, 35),
                    show_grid=True,
                    show_distance_markers=True,
                    valid_zone_polygons=[],
                    show_zones=False,
                )
                method_used = "IPM"
            except Exception:
                transformer = None
                visualizer = None

        if transformer is None:
            try:
                transformer = BirdEyeViewTransformer(
                    source_polygon=polygon,
                    bev_width=self._preview_bev_width,
                    bev_height=self._preview_bev_height,
                    margin=20,
                )
                visualizer = BirdEyeViewVisualizer(
                    transformer=transformer,
                    bg_color=(40, 40, 40),
                    lane_color=(80, 80, 80),
                    lane_border_color=(255, 255, 0),
                    show_zones=False,
                )
                method_used = "Homography"
            except Exception:
                return False

        self._bev_preview_transformer = transformer
        self._bev_preview_base_image = visualizer.base_image.copy()
        self._bev_preview_method_used = method_used
        self._bev_preview_cache_key = cache_key
        return True

    def _draw_polygon_preview_on_bev(
        self,
        bev_frame: np.ndarray,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        fill_polygon: bool,
    ):
        """Vẽ polygon zone từ image space lên BEV preview bằng transformer đã preload."""
        if self._bev_preview_transformer is None or len(points) < 2:
            return

        transformed_pts: List[Tuple[int, int]] = []
        h, w = bev_frame.shape[:2]

        for pt in points:
            bev_pt = self._bev_preview_transformer.transform_point((int(pt[0]), int(pt[1])))
            if bev_pt == (-1, -1):
                continue

            x = max(0, min(int(bev_pt[0]), w - 1))
            y = max(0, min(int(bev_pt[1]), h - 1))
            transformed_pts.append((x, y))

        if len(transformed_pts) < 2:
            return

        pts_np = np.array(transformed_pts, dtype=np.int32)

        if fill_polygon and len(transformed_pts) >= 3:
            overlay = bev_frame.copy()
            cv2.fillPoly(overlay, [pts_np], color)
            cv2.addWeighted(overlay, 0.35, bev_frame, 0.65, 0, bev_frame)

        cv2.polylines(
            bev_frame,
            [pts_np],
            isClosed=(len(transformed_pts) >= 3),
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        for idx, (x, y) in enumerate(transformed_pts):
            cv2.circle(bev_frame, (x, y), 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(bev_frame, (x, y), 3, color, -1, cv2.LINE_AA)
            cv2.putText(
                bev_frame,
                str(idx + 1),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        
    def _on_point_added(self, point: tuple):
        """Xử lý khi thêm điểm"""
        self._update_info()
        self._update_bev_preview()
        
    def _on_point_removed(self):
        """Xử lý khi xóa điểm"""
        self._update_info()
        self._update_bev_preview()
    
    def _on_suggestion_added(self):
        """Xử lý khi thêm điểm từ suggestion"""
        self._update_info()
        self._update_bev_preview()
        
    def _update_info(self):
        """Cập nhật thông tin hiển thị"""
        points_count = self._canvas.get_current_points_count()
        zones_count = len(self._canvas.get_zones())
        
        self._points_label.setText(f"Điểm hiện tại: {points_count}")
        self._zones_count_label.setText(f"Tổng số zones: {zones_count}")
        
        # Enable/disable buttons
        self._complete_zone_btn.setEnabled(points_count >= 3)
        self._confirm_btn.setEnabled(zones_count > 0 or points_count >= 3)
    
    def _update_bev_preview(self):
        """
        Cập nhật Bird's Eye View preview dựa trên zone hiện tại
        Sử dụng cùng cơ chế như VideoProcessor trong main.py:
        - Thử IPM transformer trước (với auto calibrate)
        - Fallback về Homography nếu IPM thất bại
        """
        # Kiểm tra checkbox
        if not self._bev_preview_checkbox.isChecked():
            return
            
        if self._canvas._original_frame is None:
            self._bev_preview.setText("Chưa có frame để preview")
            self._bev_preview.setPixmap(QPixmap())
            return

        try:
            if not self._preload_bev_preview_background():
                self._bev_preview.setText("Không thể preload BEV preview")
                self._bev_preview.setPixmap(QPixmap())
                return

            bev_frame = self._bev_preview_base_image.copy()
            zones = self._canvas.get_zones()
            current_points = self._canvas.get_current_points()

            for zone in zones:
                if zone.is_valid():
                    self._draw_polygon_preview_on_bev(
                        bev_frame,
                        list(zone.points),
                        zone.color,
                        fill_polygon=True,
                    )

            if current_points:
                current_color = self._canvas._zone_colors[len(zones) % len(self._canvas._zone_colors)]
                self._draw_polygon_preview_on_bev(
                    bev_frame,
                    current_points,
                    current_color,
                    fill_polygon=(len(current_points) >= 3),
                )

            info_text = (
                f"Preview ({self._bev_preview_method_used}) | "
                f"Current: {len(current_points)} pts | Zones: {len(zones)}"
            )
            cv2.putText(
                bev_frame,
                info_text,
                (8, self._preview_bev_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (170, 170, 175),
                1,
                cv2.LINE_AA,
            )

            rgb_frame = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2RGB)
            h_bev, w_bev, ch = rgb_frame.shape
            bytes_per_line = ch * w_bev
            q_image = QImage(rgb_frame.data, w_bev, h_bev, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            preview_size = self._bev_preview.size()
            scaled_pixmap = pixmap.scaled(
                preview_size.width() - 10,
                preview_size.height() - 10,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            self._bev_preview.setPixmap(scaled_pixmap)
        except Exception as e:
            self._bev_preview.setText(f"Lỗi tạo BEV:\n{str(e)[:80]}")
        
    def _update_zone_list(self):
        """Cập nhật danh sách zones"""
        self._zone_list.clear()
        
        for i, zone in enumerate(self._canvas.get_zones()):
            item = QListWidgetItem(f"Zone {i + 1} ({len(zone.points)} điểm)")
            self._zone_list.addItem(item)
            
        self._update_info()
        self._update_bev_preview()
        
    def _complete_current_zone(self):
        """Hoàn thành zone hiện tại"""
        if self._canvas.complete_current_zone():
            self._update_zone_list()
        else:
            QMessageBox.warning(
                self, 
                "Không Thể Hoàn Thành",
                "Zone cần ít nhất 3 điểm!"
            )
            
    def _delete_selected_zone(self):
        """Xóa zone được chọn"""
        selected_items = self._zone_list.selectedItems()
        if selected_items:
            index = self._zone_list.row(selected_items[0])
            self._canvas.remove_zone(index)
            self._update_zone_list()
            
    def _on_zone_selection_changed(self):
        """Xử lý khi chọn zone trong list"""
        has_selection = len(self._zone_list.selectedItems()) > 0
        self._delete_zone_btn.setEnabled(has_selection)
        
    def _clear_current(self):
        """Xóa zone đang vẽ"""
        self._canvas.clear_current()
        self._update_info()
        self._update_bev_preview()
        
    def _clear_all(self):
        """Xóa tất cả zones"""
        reply = QMessageBox.question(
            self,
            "Xác Nhận Xóa",
            "Bạn có chắc muốn xóa tất cả zones?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._canvas.clear_all()
            self._update_zone_list()
            
    def _on_skip(self):
        """Bỏ qua việc chọn zone"""
        reply = QMessageBox.question(
            self,
            "Xác Nhận Bỏ Qua",
            "Bạn có chắc muốn tiếp tục mà không chọn zone?\n"
            "Tất cả xe sẽ được theo dõi trên toàn bộ frame.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.zones_confirmed.emit([])
            
    def _on_confirm(self):
        """Xác nhận zones"""
        # Complete current zone if has enough points
        if self._canvas.get_current_points_count() >= 3:
            self._canvas.complete_current_zone()
            
        zones = self._canvas.get_zones_as_numpy()
        
        if not zones:
            reply = QMessageBox.question(
                self,
                "Không Có Zone",
                "Chưa có zone nào được tạo. Tiếp tục?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
                
        self.zones_confirmed.emit(zones)
        
    def get_zones(self) -> List[np.ndarray]:
        """Lấy danh sách zones"""
        # Auto-complete current zone if valid
        if self._canvas.get_current_points_count() >= 3:
            self._canvas.complete_current_zone()
        return self._canvas.get_zones_as_numpy()
