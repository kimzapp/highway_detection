"""
Road Zone Selection Module
Cho phép người dùng xác định thủ công vùng đường hợp lệ
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class LaneLineSuggestion:
    """
    Phát hiện và gợi ý vạch kẻ đường để hỗ trợ người dùng chọn vùng
    """
    
    def __init__(self, canny_low: int = 50, canny_high: int = 150,
                 hough_threshold: int = 50, min_line_length: int = 50,
                 max_line_gap: int = 30, suggestion_distance: int = 30):
        """
        Khởi tạo LaneLineSuggestion
        
        Args:
            canny_low: Ngưỡng thấp cho Canny edge detection
            canny_high: Ngưỡng cao cho Canny edge detection
            hough_threshold: Ngưỡng cho Hough Line Transform
            min_line_length: Độ dài tối thiểu của line
            max_line_gap: Khoảng cách tối đa giữa các điểm trên line
            suggestion_distance: Khoảng cách để kích hoạt gợi ý (pixels)
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.suggestion_distance = suggestion_distance
        
        self.detected_lines: List[np.ndarray] = []
        self.edge_points: List[np.ndarray] = []  # Contour points từ edges
        
    def detect_lanes(self, frame: np.ndarray) -> None:
        """
        Phát hiện vạch kẻ đường trong frame
        
        Args:
            frame: Frame BGR để phân tích
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Lưu edge points để gợi ý theo contour
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.edge_points = []
        for contour in contours:
            if cv2.arcLength(contour, False) > self.min_line_length:
                self.edge_points.append(contour.reshape(-1, 2))
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold,
                                minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)
        
        self.detected_lines = []
        if lines is not None:
            for line in lines:
                self.detected_lines.append(line[0])  # [x1, y1, x2, y2]
    
    def find_nearest_edge_point(self, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Tìm điểm edge gần nhất với vị trí chuột
        
        Args:
            point: Tọa độ chuột (x, y)
            
        Returns:
            Điểm edge gần nhất hoặc None
        """
        if not self.edge_points:
            return None
        
        min_dist = float('inf')
        nearest_point = None
        
        for contour_points in self.edge_points:
            for edge_pt in contour_points:
                dist = np.sqrt((edge_pt[0] - point[0])**2 + (edge_pt[1] - point[1])**2)
                if dist < min_dist and dist < self.suggestion_distance:
                    min_dist = dist
                    nearest_point = (int(edge_pt[0]), int(edge_pt[1]))
        
        return nearest_point
    
    def get_suggestion_path(self, point: Tuple[int, int], 
                           direction: str = "forward",
                           num_points: int = 20) -> List[Tuple[int, int]]:
        """
        Lấy đường gợi ý dọc theo vạch kẻ đường từ một điểm
        
        Args:
            point: Điểm bắt đầu (x, y)
            direction: Hướng tìm ("forward" hoặc "backward" theo y)
            num_points: Số điểm gợi ý
            
        Returns:
            Danh sách các điểm gợi ý
        """
        suggestion = []
        
        # Tìm contour chứa hoặc gần điểm nhất
        best_contour = None
        best_idx = -1
        min_dist = float('inf')
        
        for contour_points in self.edge_points:
            for idx, edge_pt in enumerate(contour_points):
                dist = np.sqrt((edge_pt[0] - point[0])**2 + (edge_pt[1] - point[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_contour = contour_points
                    best_idx = idx
        
        if best_contour is None or min_dist > self.suggestion_distance:
            return suggestion
        
        # Lấy các điểm dọc theo contour
        if direction == "forward":
            # Đi theo hướng tăng index (thường là xuống dưới)
            end_idx = min(best_idx + num_points, len(best_contour))
            for i in range(best_idx, end_idx):
                suggestion.append((int(best_contour[i][0]), int(best_contour[i][1])))
        else:
            # Đi theo hướng giảm index (thường là lên trên)
            start_idx = max(best_idx - num_points, 0)
            for i in range(best_idx, start_idx, -1):
                suggestion.append((int(best_contour[i][0]), int(best_contour[i][1])))
        
        return suggestion
    
    def get_extended_suggestion(self, last_point: Tuple[int, int], 
                                current_point: Tuple[int, int],
                                num_points: int = 30) -> List[Tuple[int, int]]:
        """
        Lấy gợi ý mở rộng dựa trên hướng đang vẽ
        
        Args:
            last_point: Điểm trước đó đã chọn
            current_point: Vị trí chuột hiện tại
            num_points: Số điểm gợi ý
            
        Returns:
            Danh sách điểm gợi ý theo hướng đang vẽ
        """
        # Xác định hướng dựa trên vector từ last_point đến current_point
        dx = current_point[0] - last_point[0]
        dy = current_point[1] - last_point[1]
        
        suggestion = []
        
        # Tìm contour gần current_point nhất
        best_contour = None
        best_idx = -1
        min_dist = float('inf')
        
        for contour_points in self.edge_points:
            for idx, edge_pt in enumerate(contour_points):
                dist = np.sqrt((edge_pt[0] - current_point[0])**2 + 
                             (edge_pt[1] - current_point[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_contour = contour_points
                    best_idx = idx
        
        if best_contour is None or min_dist > self.suggestion_distance * 2:
            return suggestion
        
        # Tìm hướng đi trên contour phù hợp với hướng vẽ
        # Thử cả 2 hướng và chọn hướng có vector tương tự
        forward_pts = []
        backward_pts = []
        
        # Forward direction
        for i in range(best_idx, min(best_idx + num_points, len(best_contour))):
            forward_pts.append((int(best_contour[i][0]), int(best_contour[i][1])))
        
        # Backward direction
        for i in range(best_idx, max(best_idx - num_points, -1), -1):
            backward_pts.append((int(best_contour[i][0]), int(best_contour[i][1])))
        
        # Chọn hướng có vector tương tự với hướng vẽ
        if len(forward_pts) >= 2:
            fwd_dx = forward_pts[-1][0] - forward_pts[0][0]
            fwd_dy = forward_pts[-1][1] - forward_pts[0][1]
            fwd_dot = dx * fwd_dx + dy * fwd_dy
        else:
            fwd_dot = -float('inf')
        
        if len(backward_pts) >= 2:
            bwd_dx = backward_pts[-1][0] - backward_pts[0][0]
            bwd_dy = backward_pts[-1][1] - backward_pts[0][1]
            bwd_dot = dx * bwd_dx + dy * bwd_dy
        else:
            bwd_dot = -float('inf')
        
        if fwd_dot > bwd_dot:
            suggestion = forward_pts
        else:
            suggestion = backward_pts
        
        return suggestion

    def draw_detected_lanes(self, frame: np.ndarray, 
                           color: Tuple[int, int, int] = (100, 100, 100),
                           thickness: int = 1) -> np.ndarray:
        """
        Vẽ các vạch kẻ đường đã phát hiện (dùng để debug)
        
        Args:
            frame: Frame để vẽ
            color: Màu vẽ
            thickness: Độ dày
            
        Returns:
            Frame đã vẽ
        """
        result = frame.copy()
        for line in self.detected_lines:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)
        return result


class RoadZoneSelector:
    """
    Công cụ chọn vùng đường hợp lệ bằng cách click chuột để tạo polygon
    Hỗ trợ chọn nhiều polygon cho nhiều vùng đường hợp lệ
    
    Hướng dẫn sử dụng:
    - Click chuột trái: Thêm điểm vào polygon hiện tại
    - Click chuột phải: Xóa điểm cuối cùng
    - Di chuột gần vạch kẻ đường: Hiển thị gợi ý
    - Nhấn 'S': Thêm điểm gợi ý vào polygon
    - Nhấn 'N': Lưu zone hiện tại và bắt đầu zone mới
    - Nhấn 'D': Xóa zone hiện tại
    - Nhấn 'Tab': Chuyển đổi giữa các zone
    - Nhấn 'Enter': Xác nhận tất cả zone và tiếp tục
    - Nhấn 'r': Reset zone hiện tại
    - Nhấn 'L': Bật/tắt hiển thị lane detection
    - Nhấn 'Esc': Hủy và thoát
    """
    
    WINDOW_NAME = "Select Road Zones - Click to add points, N for new zone, Enter to confirm"
    
    def __init__(self, zone_color: Tuple[int, int, int] = (0, 255, 0), 
                 zone_alpha: float = 0.3,
                 point_color: Tuple[int, int, int] = (0, 0, 255),
                 line_color: Tuple[int, int, int] = (255, 255, 0),
                 suggestion_color: Tuple[int, int, int] = (255, 0, 255),
                 enable_suggestion: bool = True):
        """
        Khởi tạo RoadZoneSelector
        
        Args:
            zone_color: Màu fill của vùng (BGR)
            zone_alpha: Độ trong suốt của vùng (0-1)
            point_color: Màu các điểm đã chọn
            line_color: Màu đường nối các điểm
            suggestion_color: Màu đường gợi ý (BGR)
            enable_suggestion: Bật/tắt tính năng gợi ý
        """
        self.zone_color = zone_color
        self.zone_alpha = zone_alpha
        self.point_color = point_color
        self.line_color = line_color
        self.suggestion_color = suggestion_color
        self.enable_suggestion = enable_suggestion
        
        # Multi-zone support
        self.zones: List[List[Tuple[int, int]]] = []  # List of completed zones
        self.points: List[Tuple[int, int]] = []  # Current zone being edited
        self._current_zone_index: int = 0  # Index of current zone (for editing completed zones)
        self._editing_completed_zone: bool = False  # True if editing a completed zone
        
        # Zone colors for visualization (different colors for each zone)
        self._zone_colors = [
            (0, 255, 0),    # Green
            (255, 165, 0),  # Orange (BGR)
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 0, 0),    # Blue
            (0, 165, 255),  # Orange variant
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
        ]
        
        self._current_frame: Optional[np.ndarray] = None
        self._selection_done = False
        self._cancelled = False
        
        # Lane suggestion
        self._lane_suggester: Optional[LaneLineSuggestion] = None
        self._current_mouse_pos: Tuple[int, int] = (0, 0)
        self._current_suggestion: List[Tuple[int, int]] = []
        self._show_lane_detection = False  # Hiển thị debug lane detection
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback xử lý sự kiện chuột"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Click trái - thêm điểm
            self.points.append((x, y))
            self._update_suggestion((x, y))
            self._draw_preview()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Click phải - xóa điểm cuối
            if self.points:
                self.points.pop()
                self._draw_preview()
        elif event == cv2.EVENT_MOUSEMOVE:
            # Di chuột - cập nhật gợi ý
            self._current_mouse_pos = (x, y)
            if self.enable_suggestion:
                self._update_suggestion((x, y))
            self._draw_preview()
    
    def _update_suggestion(self, mouse_pos: Tuple[int, int]):
        """Cập nhật đường gợi ý dựa trên vị trí chuột"""
        if not self.enable_suggestion or self._lane_suggester is None:
            self._current_suggestion = []
            return
        
        if len(self.points) >= 1:
            # Có điểm trước đó - gợi ý theo hướng đang vẽ
            last_point = self.points[-1]
            self._current_suggestion = self._lane_suggester.get_extended_suggestion(
                last_point, mouse_pos, num_points=40
            )
        else:
            # Chưa có điểm - gợi ý điểm gần nhất
            nearest = self._lane_suggester.find_nearest_edge_point(mouse_pos)
            if nearest:
                self._current_suggestion = [nearest]
            else:
                self._current_suggestion = []
    
    def _add_suggestion_points(self):
        """Thêm các điểm gợi ý vào polygon"""
        if self._current_suggestion:
            # Lọc các điểm để tránh trùng lặp và quá gần nhau
            min_distance = 15  # Khoảng cách tối thiểu giữa các điểm
            for pt in self._current_suggestion:
                if not self.points:
                    self.points.append(pt)
                else:
                    last_pt = self.points[-1]
                    dist = np.sqrt((pt[0] - last_pt[0])**2 + (pt[1] - last_pt[1])**2)
                    if dist >= min_distance:
                        self.points.append(pt)
            self._current_suggestion = []
            self._draw_preview()
    
    def _save_current_zone(self):
        """Lưu zone hiện tại vào danh sách zones và bắt đầu zone mới"""
        if len(self.points) >= 3:
            if self._editing_completed_zone:
                # Đang edit zone đã hoàn thành - cập nhật
                self.zones[self._current_zone_index] = list(self.points)
                self._editing_completed_zone = False
            else:
                # Thêm zone mới
                self.zones.append(list(self.points))
            self.points = []
            self._current_zone_index = len(self.zones)
            self._current_suggestion = []
            return True
        return False
    
    def _delete_current_zone(self):
        """Xóa zone hiện tại"""
        if self._editing_completed_zone and self.zones:
            # Xóa zone đã hoàn thành đang edit
            if 0 <= self._current_zone_index < len(self.zones):
                self.zones.pop(self._current_zone_index)
            self._editing_completed_zone = False
            self._current_zone_index = len(self.zones)
            self.points = []
        else:
            # Reset zone đang vẽ
            self.points = []
        self._current_suggestion = []
        self._draw_preview()
    
    def _switch_zone(self):
        """Chuyển đổi giữa các zone (Tab key)"""
        total_zones = len(self.zones)
        if total_zones == 0:
            return
        
        # Lưu zone hiện tại nếu có đủ điểm
        if len(self.points) >= 3 and not self._editing_completed_zone:
            self.zones.append(list(self.points))
            total_zones = len(self.zones)
        
        # Chuyển sang zone tiếp theo
        self._current_zone_index = (self._current_zone_index + 1) % (total_zones + 1)
        
        if self._current_zone_index < total_zones:
            # Edit zone đã hoàn thành
            self.points = list(self.zones[self._current_zone_index])
            self._editing_completed_zone = True
        else:
            # Tạo zone mới
            self.points = []
            self._editing_completed_zone = False
        
        self._current_suggestion = []
        self._draw_preview()
    
    def _get_zone_color(self, index: int) -> Tuple[int, int, int]:
        """Lấy màu cho zone theo index"""
        return self._zone_colors[index % len(self._zone_colors)]
    
    def _get_ui_scale(self, frame_width: int, frame_height: int) -> dict:
        """Tính toán scale cho UI dựa trên kích thước frame"""
        # Base resolution: 1280x720
        base_w, base_h = 1280, 720
        scale_w = frame_width / base_w
        scale_h = frame_height / base_h
        scale = min(scale_w, scale_h)  # Dùng scale nhỏ hơn để không bị tràn
        scale = max(0.5, min(scale, 2.0))  # Giới hạn scale từ 0.5 đến 2.0
        
        return {
            'scale': scale,
            'font_scale': scale * 0.5,
            'font_scale_header': scale * 0.55,
            'font_scale_small': scale * 0.4,
            'thickness': max(1, int(scale)),
            'panel_width': int(200 * scale),
            'panel_padding': int(12 * scale),
            'line_height': int(26 * scale),
            'point_radius': max(4, int(7 * scale)),
            'cross_size': max(10, int(14 * scale)),
            'radius': max(6, int(10 * scale)),
        }
    
    def _draw_rounded_rect(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                           color: Tuple[int, int, int], radius: int = 10, 
                           thickness: int = -1, alpha: float = 1.0) -> np.ndarray:
        """Vẽ hình chữ nhật bo góc với độ trong suốt"""
        overlay = img.copy()
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Ensure radius doesn't exceed half of width or height
        radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        
        # Vẽ các phần của rounded rectangle
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Vẽ các góc bo tròn với LINE_AA
        cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    def _draw_panel(self, img: np.ndarray, x: int, y: int, width: int, height: int,
                    bg_color: Tuple[int, int, int] = (20, 20, 25), 
                    alpha: float = 0.88, radius: int = 8) -> np.ndarray:
        """Vẽ panel với background mờ và viền nhẹ"""
        result = self._draw_rounded_rect(img, (x, y), (x + width, y + height), 
                                         bg_color, radius=radius, alpha=alpha)
        # Thêm border nhẹ
        result = self._draw_rounded_rect(result, (x, y), (x + width, y + height), 
                                         (60, 60, 65), radius=radius, thickness=1, alpha=0.5)
        return result
    
    def _draw_preview(self):
        """Vẽ preview polygon lên frame với giao diện hiện đại"""
        if self._current_frame is None:
            return
        
        h, w = self._current_frame.shape[:2]
        preview = self._current_frame.copy()
        
        # Tính UI scale
        ui = self._get_ui_scale(w, h)
        scale = ui['scale']
        
        # Font settings
        FONT = cv2.FONT_HERSHEY_DUPLEX  # Font hiện đại hơn
        FONT_LIGHT = cv2.FONT_HERSHEY_SIMPLEX
        
        # Vẽ lane detection debug nếu bật
        if self._show_lane_detection and self._lane_suggester:
            preview = self._lane_suggester.draw_detected_lanes(preview, 
                                                               color=(80, 80, 90), 
                                                               thickness=1)
        
        # ===== VẼ ĐƯỜNG GỢI Ý =====
        if self._current_suggestion and len(self._current_suggestion) >= 2:
            suggestion_pts = np.array(self._current_suggestion, dtype=np.int32)
            # Glow effect mềm mại
            cv2.polylines(preview, [suggestion_pts], isClosed=False, 
                         color=(180, 80, 220), thickness=max(4, int(5*scale)), lineType=cv2.LINE_AA)
            cv2.polylines(preview, [suggestion_pts], isClosed=False, 
                         color=self.suggestion_color, thickness=max(2, int(2*scale)), lineType=cv2.LINE_AA)
            
            # Điểm gợi ý nhỏ
            for i, pt in enumerate(self._current_suggestion[::4]):
                cv2.circle(preview, pt, max(2, int(3*scale)), (255, 255, 255), -1, cv2.LINE_AA)
                
        elif self._current_suggestion and len(self._current_suggestion) == 1:
            pt = self._current_suggestion[0]
            r = ui['point_radius']
            cv2.circle(preview, pt, r + 6, self.suggestion_color, 1, cv2.LINE_AA)
            cv2.circle(preview, pt, r + 3, self.suggestion_color, 1, cv2.LINE_AA)
            cv2.circle(preview, pt, r, self.suggestion_color, -1, cv2.LINE_AA)
        
        # ===== VẼ PREVIEW LINE TỪ ĐIỂM CUỐI ĐẾN CHUỘT =====
        if self.points and self._current_mouse_pos:
            x1, y1 = self.points[-1]
            x2, y2 = self._current_mouse_pos
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist > 0:
                # Vẽ đường nét đứt mượt mà
                dash_len = max(8, int(10 * scale))
                gap_len = max(6, int(8 * scale))
                dx, dy = (x2-x1)/dist, (y2-y1)/dist
                i = 0
                while i < dist:
                    start = (int(x1 + dx * i), int(y1 + dy * i))
                    end_i = min(i + dash_len, dist)
                    end = (int(x1 + dx * end_i), int(y1 + dy * end_i))
                    cv2.line(preview, start, end, (160, 160, 170), max(1, int(1.5*scale)), cv2.LINE_AA)
                    i += dash_len + gap_len
        
        # ===== VẼ CÁC ZONE ĐÃ HOÀN THÀNH =====
        for zone_idx, zone_points in enumerate(self.zones):
            if len(zone_points) >= 3:
                zone_color = self._get_zone_color(zone_idx)
                pts = np.array(zone_points, dtype=np.int32)
                
                # Fill zone
                overlay = preview.copy()
                cv2.fillPoly(overlay, [pts], zone_color, cv2.LINE_AA)
                alpha = 0.4 if zone_idx == self._current_zone_index and self._editing_completed_zone else 0.25
                preview = cv2.addWeighted(overlay, alpha, preview, 1 - alpha, 0)
                
                # Border
                border_thickness = 3 if zone_idx == self._current_zone_index and self._editing_completed_zone else 2
                cv2.polylines(preview, [pts], isClosed=True, color=zone_color, 
                             thickness=border_thickness, lineType=cv2.LINE_AA)
                
                # Zone label
                centroid = np.mean(pts, axis=0).astype(int)
                label = f"Zone {zone_idx + 1}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, 1)
                cv2.rectangle(preview, 
                             (centroid[0] - tw//2 - 5, centroid[1] - th//2 - 5),
                             (centroid[0] + tw//2 + 5, centroid[1] + th//2 + 5),
                             (30, 30, 35), -1)
                cv2.putText(preview, label, (centroid[0] - tw//2, centroid[1] + th//4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, zone_color, 1, cv2.LINE_AA)
        
        # ===== VẼ POLYGON HIỆN TẠI ĐANG EDIT =====
        current_zone_color = self._get_zone_color(self._current_zone_index if self._editing_completed_zone else len(self.zones))
        
        if len(self.points) >= 3:
            overlay = preview.copy()
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], current_zone_color, cv2.LINE_AA)
            preview = cv2.addWeighted(overlay, self.zone_alpha, preview, 1 - self.zone_alpha, 0)
        
        # ===== VẼ ĐƯỜNG VIỀN POLYGON =====
        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            # Subtle glow
            cv2.polylines(preview, [pts], isClosed=True, color=(80, 180, 80), 
                         thickness=max(3, int(4*scale)), lineType=cv2.LINE_AA)
            cv2.polylines(preview, [pts], isClosed=True, color=self.line_color, 
                         thickness=max(1, int(2*scale)), lineType=cv2.LINE_AA)
        
        # ===== VẼ CÁC ĐIỂM ĐÃ CHỌN =====
        r = ui['point_radius']
        for i, pt in enumerate(self.points):
            # Outer ring
            cv2.circle(preview, pt, r + 4, (70, 70, 75), 1, cv2.LINE_AA)
            # Main circle with gradient effect
            cv2.circle(preview, pt, r + 1, self.line_color, -1, cv2.LINE_AA)
            cv2.circle(preview, pt, r + 1, (255, 255, 255), 1, cv2.LINE_AA)
            # Inner highlight
            cv2.circle(preview, pt, max(2, r - 2), self.point_color, -1, cv2.LINE_AA)
            
            # Label với style mới
            label = str(i + 1)
            font_s = ui['font_scale_small']
            (tw, th), baseline = cv2.getTextSize(label, FONT, font_s, 1)
            lx, ly = pt[0] + r + 6, pt[1] - r - 2
            # Background pill
            cv2.rectangle(preview, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (30, 30, 35), -1)
            cv2.rectangle(preview, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (80, 80, 85), 1)
            cv2.putText(preview, label, (lx, ly), FONT, font_s, (230, 230, 235), 1, cv2.LINE_AA)
        
        # ===== VẼ CROSSHAIR =====
        if self._current_mouse_pos:
            mx, my = self._current_mouse_pos
            cs = ui['cross_size']
            gap = max(4, int(5 * scale))
            # Crosshair mỏng, tinh tế
            cross_color = (180, 180, 185)
            cv2.line(preview, (mx - cs, my), (mx - gap, my), cross_color, 1, cv2.LINE_AA)
            cv2.line(preview, (mx + gap, my), (mx + cs, my), cross_color, 1, cv2.LINE_AA)
            cv2.line(preview, (mx, my - cs), (mx, my - gap), cross_color, 1, cv2.LINE_AA)
            cv2.line(preview, (mx, my + gap), (mx, my + cs), cross_color, 1, cv2.LINE_AA)
        
        # ===== PANEL HƯỚNG DẪN =====
        panel_w = ui['panel_width']
        panel_pad = ui['panel_padding']
        line_h = ui['line_height']
        panel_x, panel_y = int(12 * scale), int(12 * scale)
        
        # Tính chiều cao panel
        num_items = 10  # Updated for new instructions
        panel_h = int(panel_pad * 2 + line_h * num_items + 90 * scale)
        
        preview = self._draw_panel(preview, panel_x, panel_y, panel_w, panel_h, 
                                   radius=ui['radius'])
        
        # Header
        header_y = panel_y + int(24 * scale)
        cv2.putText(preview, "ZONE SELECTOR", (panel_x + panel_pad, header_y),
                   FONT, ui['font_scale_header'], (140, 200, 255), 1, cv2.LINE_AA)
        # Divider line
        div_y = header_y + int(10 * scale)
        cv2.line(preview, (panel_x + panel_pad, div_y), 
                (panel_x + panel_w - panel_pad, div_y), (55, 55, 60), 1, cv2.LINE_AA)
        
        # Instructions - Updated with multi-zone support
        instructions = [
            ("L-Click", "Add point", (130, 230, 130)),
            ("R-Click", "Undo", (130, 160, 230)),
            ("S", "Add path", (230, 130, 230)),
            ("N", "New zone", (130, 230, 230)),
            ("Tab", "Switch zone", (230, 200, 130)),
            ("D", "Delete zone", (230, 130, 130)),
            ("L", "Show lanes", (230, 200, 130)),
            ("Enter", "Confirm all", (130, 230, 130)),
            ("R", "Reset current", (230, 170, 130)),
            ("Esc", "Cancel", (160, 160, 165)),
        ]
        
        y_pos = div_y + int(22 * scale)
        key_box_w = int(52 * scale)
        
        for key, desc, color in instructions:
            # Key badge
            (key_tw, _), _ = cv2.getTextSize(key, FONT_LIGHT, ui['font_scale_small'], 1)
            kx = panel_x + panel_pad
            cv2.rectangle(preview, (kx, y_pos - int(14*scale)), 
                         (kx + key_box_w, y_pos + int(4*scale)), (45, 45, 50), -1)
            cv2.rectangle(preview, (kx, y_pos - int(14*scale)), 
                         (kx + key_box_w, y_pos + int(4*scale)), (75, 75, 80), 1)
            # Center key text
            key_x = kx + (key_box_w - key_tw) // 2
            cv2.putText(preview, key, (key_x, y_pos),
                       FONT_LIGHT, ui['font_scale_small'], (210, 210, 215), 1, cv2.LINE_AA)
            # Description
            cv2.putText(preview, desc, (kx + key_box_w + int(8*scale), y_pos),
                       FONT_LIGHT, ui['font_scale_small'], color, 1, cv2.LINE_AA)
            y_pos += line_h
        
        # ===== STATUS BAR PHÍA DƯỚI PANEL =====
        status_y = y_pos + int(8 * scale)
        cv2.line(preview, (panel_x + panel_pad, status_y - int(5*scale)), 
                (panel_x + panel_w - panel_pad, status_y - int(5*scale)), (55, 55, 60), 1, cv2.LINE_AA)
        
        # Zone count indicator
        total_zones = len(self.zones) + (1 if len(self.points) >= 3 else 0)
        zone_text = f"Zones: {total_zones}"
        cv2.putText(preview, zone_text, (panel_x + panel_pad, status_y + int(12*scale)),
                   FONT_LIGHT, ui['font_scale_small'], (140, 200, 255), 1, cv2.LINE_AA)
        
        # Current zone points indicator
        pts_count = len(self.points)
        pts_color = (130, 230, 130) if pts_count >= 3 else (230, 200, 130)
        pts_text = f"Pts: {pts_count}"
        cv2.putText(preview, pts_text, (panel_x + panel_pad + int(70*scale), status_y + int(12*scale)),
                   FONT_LIGHT, ui['font_scale_small'], pts_color, 1, cv2.LINE_AA)
        
        # Suggestion indicator
        if self.enable_suggestion:
            ind_color = (130, 230, 130)
            ind_text = "AI"
        else:
            ind_color = (100, 100, 105)
            ind_text = "--"
        cv2.putText(preview, ind_text, (panel_x + panel_w - panel_pad - int(22*scale), status_y + int(12*scale)),
                   FONT_LIGHT, ui['font_scale_small'], ind_color, 1, cv2.LINE_AA)
        
        # ===== SUGGESTION TOAST =====
        if self._current_suggestion and len(self._current_suggestion) >= 2:
            msg = f"Press S to add {len(self._current_suggestion)} points"
            (tw, th), _ = cv2.getTextSize(msg, FONT, ui['font_scale'], 1)
            toast_w = tw + int(30 * scale)
            toast_h = th + int(16 * scale)
            toast_x = (w - toast_w) // 2
            toast_y = int(15 * scale)
            
            preview = self._draw_panel(preview, toast_x, toast_y, toast_w, toast_h,
                                       bg_color=(70, 35, 90), alpha=0.92, radius=ui['radius'])
            cv2.putText(preview, msg, (toast_x + int(15*scale), toast_y + th + int(5*scale)),
                       FONT, ui['font_scale'], (240, 200, 255), 1, cv2.LINE_AA)
        
        # ===== COORDINATES =====
        if self._current_mouse_pos:
            coord_text = f"{self._current_mouse_pos[0]}, {self._current_mouse_pos[1]}"
            (tw, th), _ = cv2.getTextSize(coord_text, FONT_LIGHT, ui['font_scale_small'], 1)
            cx = w - tw - int(20 * scale)
            cy = h - int(15 * scale)
            preview = self._draw_panel(preview, cx - int(8*scale), cy - th - int(6*scale), 
                                       tw + int(16*scale), th + int(12*scale),
                                       bg_color=(30, 30, 35), alpha=0.75, radius=max(4, int(6*scale)))
            cv2.putText(preview, coord_text, (cx, cy),
                       FONT_LIGHT, ui['font_scale_small'], (160, 160, 165), 1, cv2.LINE_AA)
        
        # ===== READY INDICATOR =====
        total_valid_zones = len(self.zones) + (1 if len(self.points) >= 3 else 0)
        if total_valid_zones > 0:
            if len(self.points) >= 3:
                ready_text = f"Ready - Press N for new zone or Enter to confirm ({total_valid_zones} zones)"
            else:
                ready_text = f"Ready - Press Enter to confirm ({total_valid_zones} zones)"
            (tw, th), _ = cv2.getTextSize(ready_text, FONT, ui['font_scale'], 1)
            rx = (w - tw) // 2 - int(15*scale)
            ry = h - int(25 * scale)
            preview = self._draw_panel(preview, rx - int(10*scale), ry - th - int(8*scale), 
                                       tw + int(30*scale), th + int(18*scale),
                                       bg_color=(30, 70, 35), alpha=0.9, radius=ui['radius'])
            cv2.putText(preview, ready_text, (rx, ry),
                       FONT, ui['font_scale'], (140, 240, 150), 1, cv2.LINE_AA)
        
        cv2.imshow(self.WINDOW_NAME, preview)
    
    def select_zone(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Hiển thị UI để người dùng chọn vùng đường
        
        Args:
            frame: Frame để hiển thị (BGR)
            
        Returns:
            Numpy array với các điểm polygon [(x1,y1), (x2,y2),...] hoặc None nếu hủy
        """
        self._current_frame = frame.copy()
        self.points = []
        self.zones = []  # Reset zones for multi-zone support
        self._editing_completed_zone = False
        self._current_zone_index = 0
        self._selection_done = False
        self._cancelled = False
        self._current_suggestion = []
        self._show_lane_detection = False
        
        # Khởi tạo lane suggester và phát hiện vạch kẻ đường
        if self.enable_suggestion:
            self._lane_suggester = LaneLineSuggestion()
            print("Đang phát hiện vạch kẻ đường...")
            self._lane_suggester.detect_lanes(frame)
            print(f"Đã phát hiện {len(self._lane_suggester.detected_lines)} đường thẳng và "
                  f"{len(self._lane_suggester.edge_points)} contour edges")
        
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
        
        self._draw_preview()
        
        while not self._selection_done and not self._cancelled:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 or key == 10:  # Enter - Confirm all zones
                # Lưu zone hiện tại nếu đủ điểm
                if len(self.points) >= 3:
                    if not self._editing_completed_zone:
                        self.zones.append(list(self.points))
                    else:
                        self.zones[self._current_zone_index] = list(self.points)
                
                if len(self.zones) > 0:
                    self._selection_done = True
                else:
                    print("Cần ít nhất 1 zone với 3 điểm!")
            elif key == ord('n') or key == ord('N'):  # New zone
                if self._save_current_zone():
                    print(f"Zone {len(self.zones)} saved. Starting new zone...")
                self._draw_preview()
            elif key == ord('d') or key == ord('D'):  # Delete zone
                self._delete_current_zone()
                print("Zone deleted.")
            elif key == 9:  # Tab - Switch zone
                self._switch_zone()
                if self._editing_completed_zone:
                    print(f"Editing Zone {self._current_zone_index + 1}")
                else:
                    print("Creating new zone")
            elif key == ord('r') or key == ord('R'):  # Reset current zone
                self.points = []
                self._current_suggestion = []
                self._draw_preview()
            elif key == ord('s') or key == ord('S'):  # Add suggestion points
                self._add_suggestion_points()
            elif key == ord('l') or key == ord('L'):  # Toggle lane detection view
                self._show_lane_detection = not self._show_lane_detection
                self._draw_preview()
            elif key == 27:  # Esc
                self._cancelled = True
        
        cv2.destroyWindow(self.WINDOW_NAME)
        
        if self._cancelled or len(self.zones) == 0:
            return None
        
        # Return first zone for backward compatibility
        return np.array(self.zones[0], dtype=np.int32)
    
    def select_zones(self, frame: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Hiển thị UI để người dùng chọn nhiều vùng đường
        
        Args:
            frame: Frame để hiển thị (BGR)
            
        Returns:
            List các numpy array polygon hoặc None nếu hủy
        """
        # Sử dụng select_zone và lấy tất cả zones
        result = self.select_zone(frame)
        
        if self._cancelled or len(self.zones) == 0:
            return None
        
        return [np.array(zone, dtype=np.int32) for zone in self.zones]
    
    def get_zone_polygon(self) -> Optional[np.ndarray]:
        """Lấy polygon đầu tiên đã chọn (backward compatibility)"""
        if len(self.zones) > 0:
            return np.array(self.zones[0], dtype=np.int32)
        if len(self.points) >= 3:
            return np.array(self.points, dtype=np.int32)
        return None
    
    def get_zone_polygons(self) -> List[np.ndarray]:
        """Lấy tất cả các polygon đã chọn"""
        result = [np.array(zone, dtype=np.int32) for zone in self.zones]
        if len(self.points) >= 3 and not self._editing_completed_zone:
            result.append(np.array(self.points, dtype=np.int32))
        return result


class RoadZoneOverlay:
    """
    Vẽ overlay vùng đường hợp lệ lên frame
    """
    
    def __init__(self, zone_polygon: np.ndarray,
                 fill_color: Tuple[int, int, int] = (0, 255, 0),
                 border_color: Tuple[int, int, int] = (255, 255, 0),
                 alpha: float = 0.2,
                 border_thickness: int = 2,
                 label: str = "Valid Lane"):
        """
        Khởi tạo RoadZoneOverlay
        
        Args:
            zone_polygon: Polygon định nghĩa vùng đường [(x1,y1), (x2,y2),...]
            fill_color: Màu fill (BGR)
            border_color: Màu viền (BGR)
            alpha: Độ trong suốt (0-1)
            border_thickness: Độ dày viền
            label: Nhãn hiển thị
        """
        self.zone_polygon = zone_polygon
        self.fill_color = fill_color
        self.border_color = border_color
        self.alpha = alpha
        self.border_thickness = border_thickness
        self.label = label
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Vẽ vùng đường lên frame
        
        Args:
            frame: Frame cần vẽ (BGR)
            
        Returns:
            Frame đã vẽ overlay
        """
        if self.zone_polygon is None or len(self.zone_polygon) < 3:
            return frame
        
        result = frame.copy()
        
        # Vẽ fill với transparency
        overlay = result.copy()
        cv2.fillPoly(overlay, [self.zone_polygon], self.fill_color)
        result = cv2.addWeighted(overlay, self.alpha, result, 1 - self.alpha, 0)
        
        # Vẽ viền
        cv2.polylines(result, [self.zone_polygon], isClosed=True, 
                     color=self.border_color, thickness=self.border_thickness)
        
        # Vẽ label ở góc trên của polygon
        if self.label:
            # Tìm điểm cao nhất (y nhỏ nhất)
            top_point = self.zone_polygon[np.argmin(self.zone_polygon[:, 1])]
            label_pos = (int(top_point[0]), int(top_point[1]) - 10)
            
            # Background cho text
            (text_w, text_h), _ = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, 
                         (label_pos[0] - 2, label_pos[1] - text_h - 5),
                         (label_pos[0] + text_w + 2, label_pos[1] + 5),
                         self.border_color, -1)
            cv2.putText(result, self.label, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return result
    
    def is_point_inside(self, point: Tuple[int, int]) -> bool:
        """
        Kiểm tra một điểm có nằm trong vùng đường không
        
        Args:
            point: Tọa độ điểm (x, y)
            
        Returns:
            True nếu điểm nằm trong vùng
        """
        if self.zone_polygon is None:
            return False
        result = cv2.pointPolygonTest(self.zone_polygon, point, False)
        return result >= 0
    
    def is_box_inside(self, box: Tuple[int, int, int, int], threshold: float = 0.5) -> bool:
        """
        Kiểm tra bounding box có nằm trong vùng đường không
        
        Args:
            box: Bounding box (x1, y1, x2, y2)
            threshold: Tỉ lệ diện tích overlap cần thiết (0-1)
            
        Returns:
            True nếu box overlap với vùng đường >= threshold
        """
        if self.zone_polygon is None:
            return False
        
        x1, y1, x2, y2 = box
        
        # Tạo mask cho box
        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= 0:
            return False
        
        # Kiểm tra center point của bottom edge (vị trí xe trên đường)
        center_bottom = (int((x1 + x2) / 2), int(y2))
        return self.is_point_inside(center_bottom)


class MultiRoadZoneOverlay:
    """
    Vẽ overlay nhiều vùng đường hợp lệ lên frame
    Hỗ trợ kiểm tra điểm/box nằm trong bất kỳ zone nào
    """
    
    # Palette màu cho các zone khác nhau
    DEFAULT_COLORS = [
        (0, 255, 0),    # Green
        (255, 165, 0),  # Orange (BGR)
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 0, 0),    # Blue
        (0, 165, 255),  # Orange variant
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
    ]
    
    def __init__(self, zone_polygons: List[np.ndarray],
                 fill_colors: Optional[List[Tuple[int, int, int]]] = None,
                 border_color: Tuple[int, int, int] = (255, 255, 0),
                 alpha: float = 0.2,
                 border_thickness: int = 2,
                 labels: Optional[List[str]] = None):
        """
        Khởi tạo MultiRoadZoneOverlay
        
        Args:
            zone_polygons: List các polygon định nghĩa vùng đường
            fill_colors: List màu fill cho mỗi zone (hoặc None để dùng màu mặc định)
            border_color: Màu viền (BGR)
            alpha: Độ trong suốt (0-1)
            border_thickness: Độ dày viền
            labels: List nhãn hiển thị cho mỗi zone (hoặc None để tự động đặt tên)
        """
        self.zone_polygons = zone_polygons
        self.border_color = border_color
        self.alpha = alpha
        self.border_thickness = border_thickness
        
        # Gán màu và label cho mỗi zone
        num_zones = len(zone_polygons)
        if fill_colors is None:
            self.fill_colors = [self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)] 
                               for i in range(num_zones)]
        else:
            self.fill_colors = fill_colors
            
        if labels is None:
            self.labels = [f"Zone {i+1}" for i in range(num_zones)]
        else:
            self.labels = labels
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Vẽ tất cả vùng đường lên frame
        
        Args:
            frame: Frame cần vẽ (BGR)
            
        Returns:
            Frame đã vẽ overlay
        """
        if not self.zone_polygons:
            return frame
        
        result = frame.copy()
        
        for i, zone_polygon in enumerate(self.zone_polygons):
            if zone_polygon is None or len(zone_polygon) < 3:
                continue
                
            fill_color = self.fill_colors[i] if i < len(self.fill_colors) else self.DEFAULT_COLORS[0]
            label = self.labels[i] if i < len(self.labels) else f"Zone {i+1}"
            
            # Vẽ fill với transparency
            overlay = result.copy()
            cv2.fillPoly(overlay, [zone_polygon], fill_color)
            result = cv2.addWeighted(overlay, self.alpha, result, 1 - self.alpha, 0)
            
            # Vẽ viền
            cv2.polylines(result, [zone_polygon], isClosed=True, 
                         color=fill_color, thickness=self.border_thickness)
            
            # Vẽ label ở góc trên của polygon
            if label:
                # Tìm điểm cao nhất (y nhỏ nhất)
                top_point = zone_polygon[np.argmin(zone_polygon[:, 1])]
                label_pos = (int(top_point[0]), int(top_point[1]) - 10)
                
                # Background cho text
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result, 
                             (label_pos[0] - 2, label_pos[1] - text_h - 5),
                             (label_pos[0] + text_w + 2, label_pos[1] + 5),
                             fill_color, -1)
                cv2.putText(result, label, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return result
    
    def is_point_inside(self, point: Tuple[int, int]) -> bool:
        """
        Kiểm tra một điểm có nằm trong bất kỳ vùng đường nào không
        
        Args:
            point: Tọa độ điểm (x, y)
            
        Returns:
            True nếu điểm nằm trong ít nhất một vùng
        """
        for zone_polygon in self.zone_polygons:
            if zone_polygon is not None and len(zone_polygon) >= 3:
                result = cv2.pointPolygonTest(zone_polygon, point, False)
                if result >= 0:
                    return True
        return False
    
    def get_zone_index(self, point: Tuple[int, int]) -> int:
        """
        Lấy index của zone chứa điểm
        
        Args:
            point: Tọa độ điểm (x, y)
            
        Returns:
            Index của zone chứa điểm, hoặc -1 nếu không nằm trong zone nào
        """
        for i, zone_polygon in enumerate(self.zone_polygons):
            if zone_polygon is not None and len(zone_polygon) >= 3:
                result = cv2.pointPolygonTest(zone_polygon, point, False)
                if result >= 0:
                    return i
        return -1
    
    def is_box_inside(self, box: Tuple[int, int, int, int], threshold: float = 0.5) -> bool:
        """
        Kiểm tra bounding box có nằm trong bất kỳ vùng đường nào không
        
        Args:
            box: Bounding box (x1, y1, x2, y2)
            threshold: Tỉ lệ diện tích overlap cần thiết (0-1)
            
        Returns:
            True nếu box overlap với ít nhất một vùng đường
        """
        x1, y1, x2, y2 = box
        
        # Tạo mask cho box
        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= 0:
            return False
        
        # Kiểm tra center point của bottom edge (vị trí xe trên đường)
        center_bottom = (int((x1 + x2) / 2), int(y2))
        return self.is_point_inside(center_bottom)
    
    def get_primary_polygon(self) -> Optional[np.ndarray]:
        """
        Lấy polygon đầu tiên (primary) cho các use-case cần một polygon duy nhất
        
        Returns:
            Polygon đầu tiên hoặc None nếu không có zone nào
        """
        if self.zone_polygons and len(self.zone_polygons) > 0:
            return self.zone_polygons[0]
        return None
    
    def get_combined_polygon(self) -> Optional[np.ndarray]:
        """
        Tạo một polygon bao quanh tất cả các zone (convex hull)
        Hữu ích cho BEV transformation
        
        Returns:
            Convex hull của tất cả các điểm từ tất cả zones
        """
        if not self.zone_polygons:
            return None
        
        # Gộp tất cả các điểm
        all_points = []
        for zone_polygon in self.zone_polygons:
            if zone_polygon is not None and len(zone_polygon) >= 3:
                all_points.extend(zone_polygon.tolist())
        
        if len(all_points) < 3:
            return None
        
        # Tính convex hull
        points_array = np.array(all_points, dtype=np.int32)
        hull = cv2.convexHull(points_array)
        return hull.reshape(-1, 2)
