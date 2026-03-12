"""
Bird's Eye View (BEV) Transformation Module
Chuyển đổi vùng làn đường sang góc nhìn từ trên xuống để giám sát chính xác hơn
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import supervision as sv


class BirdEyeViewTransformer:
    """
    Chuyển đổi perspective từ camera view sang bird's eye view
    
    Sử dụng 4 điểm từ polygon làn đường để tính ma trận transform.
    Điểm được sắp xếp theo thứ tự: top-left, top-right, bottom-right, bottom-left
    """
    
    def __init__(
        self, 
        source_polygon: np.ndarray,
        bev_width: int = 400,
        bev_height: int = 600,
        margin: int = 50
    ):
        """
        Khởi tạo BEV Transformer
        
        Args:
            source_polygon: Polygon vùng làn đường từ camera view
            bev_width: Chiều rộng của BEV output
            bev_height: Chiều cao của BEV output
            margin: Margin xung quanh BEV
        """
        self.source_polygon = source_polygon
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.margin = margin
        
        # Sắp xếp các điểm polygon theo thứ tự chuẩn
        self.source_points = self._order_points(source_polygon)
        
        # Chiều rộng và chiều cao khả dụng cho làn đường (trừ margin)
        available_width = bev_width - 2 * margin
        
        # Căn giữa theo chiều ngang
        center_x = bev_width / 2
        
        # Tạo điểm đích cho BEV (hình chữ nhật căn giữa)
        self.dest_points = np.array([
            [center_x - available_width / 2, margin],                    # top-left
            [center_x + available_width / 2, margin],                    # top-right  
            [center_x + available_width / 2, bev_height - margin],       # bottom-right
            [center_x - available_width / 2, bev_height - margin]        # bottom-left
        ], dtype=np.float32)
        
        # Tính ma trận perspective transform
        self.transform_matrix = cv2.getPerspectiveTransform(
            self.source_points.astype(np.float32), 
            self.dest_points
        )
        
        # Tính ma trận inverse để chuyển ngược từ BEV -> camera view
        self.inverse_matrix = cv2.getPerspectiveTransform(
            self.dest_points,
            self.source_points.astype(np.float32)
        )
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left
        
        Nếu polygon có nhiều hơn 4 điểm, chọn 4 điểm góc (bounding box corners)
        """
        if len(pts) == 4:
            # Sắp xếp 4 điểm
            rect = np.zeros((4, 2), dtype=np.float32)
            
            # Top-left có tổng x+y nhỏ nhất
            # Bottom-right có tổng x+y lớn nhất
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            # Top-right có hiệu y-x nhỏ nhất
            # Bottom-left có hiệu y-x lớn nhất
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            return rect
        else:
            # Với polygon nhiều hơn 4 điểm, tìm bounding quadrilateral
            # Lấy 4 điểm cực: trên cùng, dưới cùng, trái nhất, phải nhất
            hull = cv2.convexHull(pts)
            hull = hull.reshape(-1, 2)
            
            # Tìm 4 điểm góc dựa trên vị trí
            top_idx = np.argmin(hull[:, 1])
            bottom_idx = np.argmax(hull[:, 1])
            left_idx = np.argmin(hull[:, 0])
            right_idx = np.argmax(hull[:, 0])
            
            # Xác định 4 góc
            # Top side: điểm có y nhỏ nhất ở bên trái và bên phải
            top_points = hull[hull[:, 1] < np.median(hull[:, 1])]
            if len(top_points) >= 2:
                top_left = top_points[np.argmin(top_points[:, 0])]
                top_right = top_points[np.argmax(top_points[:, 0])]
            else:
                top_left = hull[top_idx]
                top_right = hull[top_idx]
            
            # Bottom side: điểm có y lớn nhất ở bên trái và bên phải  
            bottom_points = hull[hull[:, 1] >= np.median(hull[:, 1])]
            if len(bottom_points) >= 2:
                bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
                bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]
            else:
                bottom_left = hull[bottom_idx]
                bottom_right = hull[bottom_idx]
            
            return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    def transform_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """
        Chuyển đổi một điểm từ camera view sang BEV
        
        Args:
            point: Tọa độ (x, y) trong camera view
            
        Returns:
            Tọa độ (x, y) trong BEV
        """
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.transform_matrix)
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi nhiều điểm từ camera view sang BEV
        
        Args:
            points: Mảng điểm shape (N, 2)
            
        Returns:
            Mảng điểm đã transform shape (N, 2)
        """
        if len(points) == 0:
            return np.array([])
        
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self.transform_matrix)
        return transformed.reshape(-1, 2).astype(np.int32)
    
    def transform_box_to_point(self, box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Chuyển đổi bounding box thành điểm đại diện trên BEV
        Sử dụng điểm giữa cạnh dưới của box (vị trí xe trên mặt đường)
        
        Args:
            box: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Tọa độ điểm trên BEV
        """
        x1, y1, x2, y2 = box
        # Điểm giữa cạnh dưới
        center_bottom = (int((x1 + x2) / 2), int(y2))
        return self.transform_point(center_bottom)
    
    def inverse_transform_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """
        Chuyển đổi điểm từ BEV ngược về camera view
        """
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.inverse_matrix)
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))


class BirdEyeViewVisualizer:
    """
    Hiển thị Bird's Eye View với các xe được tracking
    """
    
    def __init__(
        self,
        transformer: BirdEyeViewTransformer,
        bg_color: Tuple[int, int, int] = (40, 40, 40),
        lane_color: Tuple[int, int, int] = (100, 100, 100),
        lane_border_color: Tuple[int, int, int] = (255, 255, 0),
        vehicle_colors: Optional[dict] = None,
        history_length: int = 10,
        show_zones: bool = True,
        valid_zone_color: Tuple[int, int, int] = (0, 100, 0),      # Xanh lá đậm
        invalid_zone_color: Tuple[int, int, int] = (0, 0, 100),    # Đỏ đậm
        zone_alpha: float = 0.6
    ):
        """
        Khởi tạo BEV Visualizer
        
        Args:
            transformer: BEV transformer đã được khởi tạo
            bg_color: Màu nền BEV
            lane_color: Màu của làn đường
            lane_border_color: Màu viền làn đường
            vehicle_colors: Dict mapping class_id -> color
            history_length: Số frame lưu lại để tính hướng di chuyển
            show_zones: Hiển thị vùng valid/invalid
            valid_zone_color: Màu vùng hợp lệ (làn đường) - xanh lá
            invalid_zone_color: Màu vùng không hợp lệ - đỏ
            zone_alpha: Độ trong suốt của vùng zone (0.0-1.0)
        """
        self.transformer = transformer
        self.bg_color = bg_color
        self.lane_color = lane_color
        self.lane_border_color = lane_border_color
        self.history_length = history_length
        self.show_zones = show_zones
        self.valid_zone_color = valid_zone_color
        self.invalid_zone_color = invalid_zone_color
        self.zone_alpha = zone_alpha
        
        # Lưu lịch sử vị trí của mỗi xe để tính hướng di chuyển
        # Dict: tracker_id -> list of (x, y) positions
        self.position_history: dict = {}
        
        # Màu mặc định cho các loại xe
        self.vehicle_colors = vehicle_colors or {
            0: (0, 255, 0),     # person - xanh lá
            1: (255, 165, 0),   # bicycle - cam
            2: (0, 0, 255),     # car - đỏ
            3: (255, 0, 255),   # motorcycle - tím
            5: (0, 255, 255),   # bus - vàng
            7: (255, 0, 0),     # truck - xanh dương
        }
        self.default_color = (255, 255, 255)  # Trắng cho các class khác
        
        # Tạo base BEV image với làn đường
        self._create_base_image()
    
    def _create_base_image(self):
        """Tạo ảnh nền BEV với làn đường căn giữa và vùng valid/invalid"""
        self.base_image = np.full(
            (self.transformer.bev_height, self.transformer.bev_width, 3),
            self.bg_color,
            dtype=np.uint8
        )
        
        lane_pts = self.transformer.dest_points.astype(np.int32)
        
        if self.show_zones:
            # Vẽ vùng Invalid Zone (toàn bộ nền) - màu đỏ
            invalid_overlay = np.full_like(self.base_image, self.invalid_zone_color, dtype=np.uint8)
            
            # Vẽ vùng Valid Zone (làn đường) - màu xanh lá
            valid_overlay = np.zeros_like(self.base_image, dtype=np.uint8)
            cv2.fillPoly(valid_overlay, [lane_pts], self.valid_zone_color)
            
            # Tạo mask cho valid zone
            valid_mask = np.zeros((self.transformer.bev_height, self.transformer.bev_width), dtype=np.uint8)
            cv2.fillPoly(valid_mask, [lane_pts], 255)
            
            # Blend các zone lên base image
            # Vùng invalid (ngoài làn đường)
            invalid_mask = cv2.bitwise_not(valid_mask)
            self.base_image = cv2.addWeighted(
                self.base_image, 1 - self.zone_alpha,
                invalid_overlay, self.zone_alpha,
                0
            )
            
            # Vùng valid (trong làn đường)
            valid_region = cv2.bitwise_and(valid_overlay, valid_overlay, mask=valid_mask)
            base_valid_region = cv2.bitwise_and(self.base_image, self.base_image, mask=valid_mask)
            base_invalid_region = cv2.bitwise_and(self.base_image, self.base_image, mask=invalid_mask)
            
            # Blend valid zone riêng
            blended_valid = cv2.addWeighted(
                base_valid_region, 0.4,
                valid_region, 0.6,
                0
            )
            
            # Kết hợp lại
            self.base_image = cv2.add(base_invalid_region, blended_valid)
            
            # Vẽ viền làn đường (đường phân cách valid/invalid)
            cv2.polylines(self.base_image, [lane_pts], True, (255, 255, 255), 2)
            
            # Thêm label cho các zone
            self._draw_zone_labels()
        else:
            # Vẽ làn đường thông thường (không có zone colors)
            cv2.fillPoly(self.base_image, [lane_pts], self.lane_color)
            cv2.polylines(self.base_image, [lane_pts], True, self.lane_border_color, 2)
        
        # Thêm label căn giữa ở phía trên
        label = "Bird's Eye View"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = (self.transformer.bev_width - text_w) // 2
        cv2.putText(self.base_image, label, (text_x, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _draw_zone_labels(self):
        """Vẽ nhãn cho các vùng valid/invalid"""
        # Label "VALID ZONE" ở giữa làn đường
        valid_label = "VALID ZONE"
        (vl_w, vl_h), _ = cv2.getTextSize(valid_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Tính vị trí giữa làn đường
        lane_center_x = int(self.transformer.bev_width / 2)
        lane_center_y = int((self.transformer.margin + self.transformer.bev_height - self.transformer.margin) / 2)
        
        # Vẽ text với background
        cv2.rectangle(self.base_image,
                     (lane_center_x - vl_w // 2 - 8, lane_center_y - vl_h // 2 - 8),
                     (lane_center_x + vl_w // 2 + 8, lane_center_y + vl_h // 2 + 8),
                     (0, 150, 0), -1)
        cv2.rectangle(self.base_image,
                     (lane_center_x - vl_w // 2 - 8, lane_center_y - vl_h // 2 - 8),
                     (lane_center_x + vl_w // 2 + 8, lane_center_y + vl_h // 2 + 8),
                     (255, 255, 255), 1)
        cv2.putText(self.base_image, valid_label,
                   (lane_center_x - vl_w // 2, lane_center_y + vl_h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Label "INVALID ZONE" ở bên trái
        invalid_label = "INVALID"
        (il_w, il_h), _ = cv2.getTextSize(invalid_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        left_x = self.transformer.margin // 2
        left_y = self.transformer.bev_height // 2
        
        cv2.rectangle(self.base_image,
                     (left_x - il_w // 2 - 5, left_y - il_h // 2 - 5),
                     (left_x + il_w // 2 + 5, left_y + il_h // 2 + 5),
                     (0, 0, 180), -1)
        cv2.rectangle(self.base_image,
                     (left_x - il_w // 2 - 5, left_y - il_h // 2 - 5),
                     (left_x + il_w // 2 + 5, left_y + il_h // 2 + 5),
                     (255, 255, 255), 1)
        cv2.putText(self.base_image, invalid_label,
                   (left_x - il_w // 2, left_y + il_h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Label "INVALID ZONE" ở bên phải
        right_x = self.transformer.bev_width - self.transformer.margin // 2
        cv2.rectangle(self.base_image,
                     (right_x - il_w // 2 - 5, left_y - il_h // 2 - 5),
                     (right_x + il_w // 2 + 5, left_y + il_h // 2 + 5),
                     (0, 0, 180), -1)
        cv2.rectangle(self.base_image,
                     (right_x - il_w // 2 - 5, left_y - il_h // 2 - 5),
                     (right_x + il_w // 2 + 5, left_y + il_h // 2 + 5),
                     (255, 255, 255), 1)
        cv2.putText(self.base_image, invalid_label,
                   (right_x - il_w // 2, left_y + il_h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_legend(self, bev_image: np.ndarray, valid_count: int, invalid_count: int, total_count: int):
        """
        Vẽ legend box với thông tin chi tiết về zone và thống kê
        
        Args:
            bev_image: Ảnh BEV để vẽ lên
            valid_count: Số xe trong vùng valid
            invalid_count: Số xe trong vùng invalid
            total_count: Tổng số xe
        """
        # Legend box ở góc trên bên phải
        legend_width = 120
        legend_height = 110
        legend_x = self.transformer.bev_width - legend_width - 10
        legend_y = 40
        
        # Vẽ background legend
        overlay = bev_image.copy()
        cv2.rectangle(overlay, 
                     (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height),
                     (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, bev_image, 0.2, 0, bev_image)
        
        # Viền legend
        cv2.rectangle(bev_image, 
                     (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height),
                     (255, 255, 255), 1)
        
        # Title
        cv2.putText(bev_image, "ZONE INFO", 
                   (legend_x + 15, legend_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Đường kẻ phân cách
        cv2.line(bev_image, (legend_x + 5, legend_y + 25), 
                (legend_x + legend_width - 5, legend_y + 25), (100, 100, 100), 1)
        
        # Valid zone indicator
        cv2.rectangle(bev_image, 
                     (legend_x + 8, legend_y + 32), 
                     (legend_x + 22, legend_y + 46), 
                     (0, 180, 0), -1)
        cv2.putText(bev_image, f"Valid: {valid_count}", 
                   (legend_x + 28, legend_y + 44),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Invalid zone indicator
        cv2.rectangle(bev_image, 
                     (legend_x + 8, legend_y + 52), 
                     (legend_x + 22, legend_y + 66), 
                     (0, 0, 180), -1)
        cv2.putText(bev_image, f"Invalid: {invalid_count}", 
                   (legend_x + 28, legend_y + 64),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Đường kẻ phân cách
        cv2.line(bev_image, (legend_x + 5, legend_y + 72), 
                (legend_x + legend_width - 5, legend_y + 72), (100, 100, 100), 1)
        
        # Total count
        cv2.putText(bev_image, f"Total: {total_count}", 
                   (legend_x + 22, legend_y + 88),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Warning nếu có xe invalid
        if invalid_count > 0:
            cv2.putText(bev_image, "! WARNING", 
                       (legend_x + 20, legend_y + 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def is_in_valid_zone(self, point: Tuple[int, int]) -> bool:
        """
        Kiểm tra một điểm có nằm trong vùng valid (làn đường) hay không
        
        Args:
            point: Tọa độ (x, y) trên BEV
            
        Returns:
            True nếu điểm nằm trong làn đường (valid zone), False nếu ngoài
        """
        lane_pts = self.transformer.dest_points.astype(np.int32)
        # pointPolygonTest trả về: > 0 nếu bên trong, = 0 nếu trên biên, < 0 nếu bên ngoài
        result = cv2.pointPolygonTest(lane_pts, (float(point[0]), float(point[1])), False)
        return result >= 0
    
    def get_zone_status(self, bev_point: Tuple[int, int]) -> str:
        """
        Lấy trạng thái vùng của một điểm trên BEV
        
        Args:
            bev_point: Tọa độ (x, y) trên BEV
            
        Returns:
            "valid" nếu trong làn đường, "invalid" nếu ngoài làn đường
        """
        return "valid" if self.is_in_valid_zone(bev_point) else "invalid"
    
    def _draw_dashed_line(self, img, pt1, pt2, color, thickness, dash_length, gap_length):
        """Vẽ đường đứt đoạn"""
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        if dist == 0:
            return
            
        dx = (pt2[0] - pt1[0]) / dist
        dy = (pt2[1] - pt1[1]) / dist
        
        current = 0
        while current < dist:
            start = (int(pt1[0] + dx * current), int(pt1[1] + dy * current))
            end_dist = min(current + dash_length, dist)
            end = (int(pt1[0] + dx * end_dist), int(pt1[1] + dy * end_dist))
            cv2.line(img, start, end, color, thickness)
            current += dash_length + gap_length
    
    def get_vehicle_color(self, class_id: int, tracker_id: int = None) -> Tuple[int, int, int]:
        """Lấy màu cho xe dựa trên class hoặc tracker ID"""
        if tracker_id is not None:
            # Tạo màu unique dựa trên tracker ID
            np.random.seed(tracker_id * 10)
            return tuple(np.random.randint(100, 255, 3).tolist())
        return self.vehicle_colors.get(class_id, self.default_color)
    
    def update_position_history(self, tracker_id: int, position: Tuple[int, int]):
        """
        Cập nhật lịch sử vị trí cho một xe
        
        Args:
            tracker_id: ID của xe
            position: Vị trí (x, y) hiện tại trên BEV
        """
        if tracker_id not in self.position_history:
            self.position_history[tracker_id] = []
        
        self.position_history[tracker_id].append(position)
        
        # Giới hạn số lượng lịch sử
        if len(self.position_history[tracker_id]) > self.history_length:
            self.position_history[tracker_id].pop(0)
    
    def get_movement_direction(self, tracker_id: int) -> Optional[Tuple[float, float]]:
        """
        Tính hướng di chuyển của xe dựa trên lịch sử vị trí
        
        Args:
            tracker_id: ID của xe
            
        Returns:
            Vector hướng (dx, dy) đã normalize, hoặc None nếu chưa đủ dữ liệu
        """
        if tracker_id not in self.position_history:
            return None
        
        history = self.position_history[tracker_id]
        if len(history) < 2:
            return None
        
        # Tính vector di chuyển từ vị trí cũ nhất đến vị trí mới nhất
        # Sử dụng nhiều điểm để có hướng ổn định hơn
        old_pos = history[0]
        new_pos = history[-1]
        
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]
        
        # Tính độ dài vector
        length = np.sqrt(dx * dx + dy * dy)
        
        # Nếu xe gần như đứng yên, không có hướng rõ ràng
        if length < 3:
            return None
        
        # Normalize vector
        return (dx / length, dy / length)
    
    def clean_old_tracks(self, current_tracker_ids: List[int]):
        """
        Xóa lịch sử của các xe không còn được track
        
        Args:
            current_tracker_ids: Danh sách ID của các xe đang được track
        """
        # Lấy danh sách các tracker_id cần xóa
        ids_to_remove = [tid for tid in self.position_history.keys() 
                        if tid not in current_tracker_ids]
        for tid in ids_to_remove:
            del self.position_history[tid]
    
    def draw(
        self,
        detections: sv.Detections,
        class_names: Optional[dict] = None,
        show_ids: bool = True,
        show_labels: bool = True,
        show_zone_stats: bool = True
    ) -> np.ndarray:
        """
        Vẽ BEV với các xe từ detections
        
        Args:
            detections: Supervision Detections object với tracker IDs
            class_names: Dict mapping class_id -> class name
            show_ids: Hiển thị tracker IDs
            show_labels: Hiển thị class labels
            show_zone_stats: Hiển thị thống kê số xe trong từng zone
            
        Returns:
            BEV image với các xe được vẽ
        """
        bev_image = self.base_image.copy()
        
        if detections is None or len(detections) == 0:
            return bev_image
        
        # Lấy thông tin detections
        boxes = detections.xyxy
        class_ids = detections.class_id
        tracker_ids = detections.tracker_id if detections.tracker_id is not None else [None] * len(boxes)
        confidences = detections.confidence if detections.confidence is not None else [1.0] * len(boxes)
        
        # Xóa lịch sử của các xe không còn được track
        valid_tracker_ids = [tid for tid in tracker_ids if tid is not None]
        self.clean_old_tracks(valid_tracker_ids)
        
        # Đếm số xe trong từng zone
        valid_count = 0
        invalid_count = 0
        
        # Xóa lịch sử của các xe không còn được track
        valid_tracker_ids = [tid for tid in tracker_ids if tid is not None]
        self.clean_old_tracks(valid_tracker_ids)
        
        # Vẽ từng xe lên BEV
        for i, (box, class_id, tracker_id, conf) in enumerate(zip(boxes, class_ids, tracker_ids, confidences)):
            # Transform điểm xe sang BEV
            try:
                bev_point = self.transformer.transform_box_to_point(box)
            except:
                continue
            
            # Kiểm tra điểm có nằm trong vùng BEV không
            if not (0 <= bev_point[0] < self.transformer.bev_width and 
                    0 <= bev_point[1] < self.transformer.bev_height):
                continue
            
            # Cập nhật lịch sử vị trí nếu có tracker_id
            if tracker_id is not None:
                self.update_position_history(tracker_id, bev_point)
            
            # Lấy màu
            color = self.get_vehicle_color(class_id, tracker_id)
            
            # Kiểm tra xe có nằm trong vùng valid hay không
            is_valid = self.is_in_valid_zone(bev_point)
            
            # Đếm số xe
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
            
            # Vẽ hình đại diện xe (hình chữ nhật nhỏ)
            vehicle_width = 20
            vehicle_height = 35
            x, y = bev_point
            
            # Vẽ xe như hình chữ nhật
            pt1 = (x - vehicle_width // 2, y - vehicle_height // 2)
            pt2 = (x + vehicle_width // 2, y + vehicle_height // 2)
            cv2.rectangle(bev_image, pt1, pt2, color, -1)
            
            # Viền xe: trắng nếu valid, đỏ nhấp nháy nếu invalid
            if is_valid:
                border_color = (255, 255, 255)  # Trắng
                cv2.rectangle(bev_image, pt1, pt2, border_color, 1)
            else:
                # Xe nằm ngoài làn đường - vẽ viền đỏ đậm và thêm cảnh báo
                border_color = (0, 0, 255)  # Đỏ
                cv2.rectangle(bev_image, pt1, pt2, border_color, 2)
                # Vẽ dấu X cảnh báo trên xe
                cv2.line(bev_image, pt1, pt2, (0, 0, 255), 2)
                cv2.line(bev_image, (pt1[0], pt2[1]), (pt2[0], pt1[1]), (0, 0, 255), 2)
            
            # Tính và vẽ mũi tên chỉ hướng di chuyển thực tế
            arrow_length = 15  # Độ dài mũi tên
            direction = None
            if tracker_id is not None:
                direction = self.get_movement_direction(tracker_id)
            
            if direction is not None:
                dx, dy = direction
                # Tính điểm bắt đầu và kết thúc của mũi tên trên xe
                arrow_start = (int(x - dx * arrow_length / 2), int(y - dy * arrow_length / 2))
                arrow_end = (int(x + dx * arrow_length / 2), int(y + dy * arrow_length / 2))
                cv2.arrowedLine(bev_image, arrow_start, arrow_end, (255, 255, 255), 2, tipLength=0.5)
            else:
                # Nếu chưa có hướng, vẽ một chấm tròn nhỏ ở giữa
                cv2.circle(bev_image, (x, y), 3, (255, 255, 255), -1)
            
            # Vẽ label
            label_parts = []
            if show_ids and tracker_id is not None:
                label_parts.append(f"#{tracker_id}")
            if show_labels and class_names and class_id in class_names:
                label_parts.append(class_names[class_id])
            
            if label_parts:
                label = " ".join(label_parts)
                label_pos = (x - vehicle_width // 2, y - vehicle_height // 2 - 5)
                
                # Background cho label
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(bev_image,
                             (label_pos[0] - 2, label_pos[1] - text_h - 2),
                             (label_pos[0] + text_w + 2, label_pos[1] + 2),
                             color, -1)
                cv2.putText(bev_image, label, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Thêm thống kê ở phía dưới
        vehicle_count = len(detections)
        
        if show_zone_stats and self.show_zones:
            # Vẽ legend box với thông tin chi tiết
            self._draw_legend(bev_image, valid_count, invalid_count, vehicle_count)
            
            # Hiển thị thống kê nhanh ở phía dưới
            stats_y = self.transformer.bev_height - 10
            
            # Valid count (màu xanh lá, bên trái)
            valid_text = f"In Lane: {valid_count}"
            cv2.putText(bev_image, valid_text, (10, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            
            # Invalid count (màu đỏ, bên phải)
            invalid_text = f"Out Lane: {invalid_count}"
            (inv_w, _), _ = cv2.getTextSize(invalid_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(bev_image, invalid_text, 
                       (self.transformer.bev_width - inv_w - 10, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        else:
            # Hiển thị thống kê đơn giản
            count_text = f"Vehicles: {vehicle_count}"
            (count_w, count_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            count_x = (self.transformer.bev_width - count_w) // 2
            cv2.putText(bev_image, count_text, 
                       (count_x, self.transformer.bev_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return bev_image


def create_combined_view(
    camera_frame: np.ndarray,
    bev_frame: np.ndarray,
    layout: str = "horizontal"
) -> np.ndarray:
    """
    Kết hợp camera view và BEV thành một frame
    
    Args:
        camera_frame: Frame từ camera
        bev_frame: Frame BEV
        layout: "horizontal" hoặc "vertical"
        
    Returns:
        Combined frame
    """
    cam_h, cam_w = camera_frame.shape[:2]
    bev_h, bev_w = bev_frame.shape[:2]
    
    if layout == "horizontal":
        # Scale BEV để có cùng chiều cao với camera frame
        scale = cam_h / bev_h
        new_bev_w = int(bev_w * scale)
        new_bev_h = cam_h
        bev_resized = cv2.resize(bev_frame, (new_bev_w, new_bev_h))
        
        # Tạo canvas cho BEV với chiều cao bằng camera frame
        # Đảm bảo BEV được căn giữa theo chiều dọc
        bev_canvas = np.full((cam_h, new_bev_w, 3), (40, 40, 40), dtype=np.uint8)
        
        # Tính offset để căn giữa BEV theo chiều dọc (nếu cần)
        y_offset = (cam_h - new_bev_h) // 2
        y_offset = max(0, y_offset)
        
        # Đặt BEV vào canvas căn giữa
        if y_offset > 0:
            bev_canvas[y_offset:y_offset + new_bev_h, :] = bev_resized
        else:
            bev_canvas = bev_resized
        
        # Vẽ đường phân cách giữa camera view và BEV
        cv2.line(bev_canvas, (0, 0), (0, cam_h), (255, 255, 255), 2)
        
        # Ghép ngang
        combined = np.hstack([camera_frame, bev_canvas])
    else:
        # Scale BEV để có cùng chiều rộng với camera frame
        scale = cam_w / bev_w
        new_bev_h = int(bev_h * scale)
        new_bev_w = cam_w
        bev_resized = cv2.resize(bev_frame, (new_bev_w, new_bev_h))
        
        # Tạo canvas cho BEV với chiều rộng bằng camera frame
        bev_canvas = np.full((new_bev_h, cam_w, 3), (40, 40, 40), dtype=np.uint8)
        
        # Tính offset để căn giữa BEV theo chiều ngang
        x_offset = (cam_w - new_bev_w) // 2
        x_offset = max(0, x_offset)
        
        if x_offset > 0:
            bev_canvas[:, x_offset:x_offset + new_bev_w] = bev_resized
        else:
            bev_canvas = bev_resized
        
        # Vẽ đường phân cách
        cv2.line(bev_canvas, (0, 0), (cam_w, 0), (255, 255, 255), 2)
        
        # Ghép dọc
        combined = np.vstack([camera_frame, bev_canvas])
    
    return combined


# =============================================================================
# INVERSE PERSPECTIVE MAPPING (IPM) - PHƯƠNG PHÁP NÂNG CAO
# =============================================================================

class VanishingPointDetector:
    """
    Phát hiện vanishing point để ước lượng camera parameters tự động
    
    Vanishing point là điểm hội tụ của các đường song song trên mặt đường,
    giúp xác định góc nghiêng camera và horizon line.
    """
    
    def __init__(self, canny_low: int = 50, canny_high: int = 150,
                 hough_threshold: int = 80, min_line_length: int = 50,
                 max_line_gap: int = 50):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
    
    def detect_lines(self, frame: np.ndarray) -> np.ndarray:
        """Phát hiện các đường thẳng trong frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Chỉ xét nửa dưới của ảnh (vùng đường)
        h = edges.shape[0]
        roi_mask = np.zeros_like(edges)
        roi_mask[h//3:, :] = 255
        edges = cv2.bitwise_and(edges, roi_mask)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold,
                                minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)
        return lines
    
    def filter_lane_lines(self, lines: np.ndarray, frame_width: int) -> Tuple[List, List]:
        """
        Lọc và phân loại các đường thành left lanes và right lanes
        Dựa trên độ nghiêng (slope)
        """
        if lines is None:
            return [], []
        
        left_lines = []
        right_lines = []
        center_x = frame_width // 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Lọc các đường gần nằm ngang
            if abs(slope) < 0.3:
                continue
            
            # Tính điểm giữa của đường
            mid_x = (x1 + x2) // 2
            
            # Phân loại dựa trên slope và vị trí
            if slope < 0 and mid_x < center_x:
                left_lines.append(line[0])
            elif slope > 0 and mid_x > center_x:
                right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def find_vanishing_point(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Tìm vanishing point từ các lane lines
        
        Returns:
            Tọa độ (x, y) của vanishing point hoặc None
        """
        h, w = frame.shape[:2]
        lines = self.detect_lines(frame)
        left_lines, right_lines = self.filter_lane_lines(lines, w)
        
        if len(left_lines) < 1 or len(right_lines) < 1:
            # Không đủ lines, trả về điểm mặc định
            return (w // 2, h // 3)
        
        # Tính các đường hồi quy cho left và right lanes
        def fit_line(lines_list):
            points = []
            for x1, y1, x2, y2 in lines_list:
                points.extend([(x1, y1), (x2, y2)])
            if len(points) < 2:
                return None
            points = np.array(points)
            # Fit đường thẳng y = mx + b
            vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            return (vx, vy, x0, y0)
        
        left_fit = fit_line(left_lines)
        right_fit = fit_line(right_lines)
        
        if left_fit is None or right_fit is None:
            return (w // 2, h // 3)
        
        # Tìm điểm giao của 2 đường
        vx1, vy1, x01, y01 = left_fit
        vx2, vy2, x02, y02 = right_fit
        
        # Chuyển về dạng ax + by + c = 0
        # Line 1: vy1 * (x - x01) = vx1 * (y - y01)
        # => vy1 * x - vx1 * y = vy1 * x01 - vx1 * y01
        a1, b1, c1 = vy1, -vx1, vy1 * x01 - vx1 * y01
        a2, b2, c2 = vy2, -vx2, vy2 * x02 - vx2 * y02
        
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return (w // 2, h // 3)
        
        vp_x = (c1 * b2 - c2 * b1) / det
        vp_y = (a1 * c2 - a2 * c1) / det
        
        # Validate vanishing point
        if 0 <= vp_x < w and 0 <= vp_y < h * 0.6:
            return (int(vp_x), int(vp_y))
        
        return (w // 2, h // 3)


class IPMBirdEyeViewTransformer:
    """
    Inverse Perspective Mapping (IPM) Bird's Eye View Transformer
    
    Phương pháp này sử dụng mô hình hình học camera thực để transform,
    chính xác hơn homography vì:
    1. Dựa trên thông số camera thực (height, pitch angle, focal length)
    2. Tự động estimate parameters từ vanishing point
    3. Scale đúng theo khoảng cách thực
    4. Xử lý tốt hơn với các camera có góc nghiêng khác nhau
    
    Công thức IPM:
    - Y_world = camera_height / tan(pitch_angle + atan((v - cy) / focal_length))
    - X_world = Y_world * (u - cx) / focal_length
    """
    
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        camera_height: float = 1.5,           # Chiều cao camera (meters)
        pitch_angle: float = None,             # Góc nghiêng (radians), None = auto
        focal_length: float = None,            # Focal length (pixels), None = auto
        bev_width: int = 400,
        bev_height: int = 600,
        bev_scale: float = 30.0,              # pixels per meter trong BEV
        roi_polygon: np.ndarray = None,        # Vùng quan tâm (optional)
        auto_calibrate: bool = True            # Tự động calibrate từ frame
    ):
        """
        Khởi tạo IPM Transformer
        
        Args:
            frame_width, frame_height: Kích thước frame gốc
            camera_height: Chiều cao camera so với mặt đường (meters)
            pitch_angle: Góc nghiêng camera (radians). None = auto estimate
            focal_length: Tiêu cự camera (pixels). None = estimate từ frame size
            bev_width, bev_height: Kích thước BEV output
            bev_scale: Tỷ lệ pixels/meter trong BEV
            roi_polygon: Polygon giới hạn vùng transform
            auto_calibrate: Tự động calibrate từ vanishing point
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_height = camera_height
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.bev_scale = bev_scale
        self.roi_polygon = roi_polygon
        
        # Camera center (principal point)
        self.cx = frame_width / 2
        self.cy = frame_height / 2
        
        # Estimate focal length nếu không cung cấp
        # Giả sử FOV khoảng 60-70 độ cho camera dashcam/surveillance
        if focal_length is None:
            fov_horizontal = np.radians(65)  # 65 degrees FOV
            self.focal_length = frame_width / (2 * np.tan(fov_horizontal / 2))
        else:
            self.focal_length = focal_length
        
        # Estimate pitch angle nếu không cung cấp
        if pitch_angle is None:
            # Mặc định: giả sử horizon ở khoảng 1/3 từ trên xuống
            horizon_y = frame_height / 3
            self.pitch_angle = np.arctan((self.cy - horizon_y) / self.focal_length)
        else:
            self.pitch_angle = pitch_angle
        
        self._vanishing_point = None
        self._is_calibrated = False
        self.auto_calibrate = auto_calibrate
        
        # Precompute các giá trị cần thiết
        self._update_transform_params()
    
    def _update_transform_params(self):
        """Cập nhật các tham số transform sau khi calibrate"""
        self.sin_pitch = np.sin(self.pitch_angle)
        self.cos_pitch = np.cos(self.pitch_angle)
        
        # Tính toán vùng BEV
        # BEV center offset (camera position trong BEV)
        self.bev_center_x = self.bev_width / 2
        self.bev_origin_y = self.bev_height - 50  # Vị trí camera trong BEV
    
    def calibrate_from_frame(self, frame: np.ndarray) -> bool:
        """
        Tự động calibrate camera parameters từ vanishing point
        
        Args:
            frame: Frame BGR để phân tích
            
        Returns:
            True nếu calibrate thành công
        """
        detector = VanishingPointDetector()
        vp = detector.find_vanishing_point(frame)
        
        if vp is None:
            return False
        
        self._vanishing_point = vp
        vp_x, vp_y = vp
        
        # Ước lượng pitch angle từ vanishing point
        # Vanishing point y cho biết horizon line
        self.pitch_angle = np.arctan((self.cy - vp_y) / self.focal_length)
        
        # Điều chỉnh camera center x nếu vanishing point lệch
        # (cho trường hợp camera không hoàn toàn thẳng)
        self.cx = vp_x
        
        self._update_transform_params()
        self._is_calibrated = True
        
        return True
    
    def image_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        Chuyển điểm từ image coordinates sang world coordinates (ground plane)
        
        Args:
            u, v: Tọa độ pixel trong ảnh
            
        Returns:
            X, Y: Tọa độ world (meters) trên mặt đất
        """
        # Tránh division by zero và các điểm trên horizon
        # Điểm trên horizon sẽ có Y = infinity
        
        # Góc từ camera tới điểm (u, v)
        alpha = np.arctan((v - self.cy) / self.focal_length)
        theta = self.pitch_angle + alpha
        
        # Nếu theta <= 0, điểm nằm trên hoặc trên horizon
        if theta <= 0.01:
            return (float('inf'), float('inf'))
        
        # Khoảng cách Y trên mặt đất
        Y = self.camera_height / np.tan(theta)
        
        # Khoảng cách X trên mặt đất
        X = Y * (u - self.cx) / self.focal_length
        
        return (X, Y)
    
    def world_to_bev(self, X: float, Y: float) -> Tuple[int, int]:
        """
        Chuyển từ world coordinates sang BEV pixel coordinates
        
        Args:
            X, Y: World coordinates (meters)
            
        Returns:
            bev_x, bev_y: BEV pixel coordinates
        """
        # X_world -> bev_x (trái phải)
        bev_x = self.bev_center_x + X * self.bev_scale
        
        # Y_world -> bev_y (xa gần, Y lớn = xa = ở trên BEV)
        bev_y = self.bev_origin_y - Y * self.bev_scale
        
        return (int(bev_x), int(bev_y))
    
    def transform_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """
        Chuyển đổi một điểm từ camera view sang BEV
        
        Args:
            point: Tọa độ (x, y) trong camera view
            
        Returns:
            Tọa độ (x, y) trong BEV
        """
        u, v = point
        X, Y = self.image_to_world(u, v)
        
        # Kiểm tra điểm hợp lệ
        if X == float('inf') or Y == float('inf'):
            return (-1, -1)
        if Y < 0 or Y > 100:  # Giới hạn khoảng cách hợp lý (100m)
            return (-1, -1)
        
        return self.world_to_bev(X, Y)
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi nhiều điểm từ camera view sang BEV
        
        Args:
            points: Mảng điểm shape (N, 2)
            
        Returns:
            Mảng điểm đã transform shape (N, 2)
        """
        if len(points) == 0:
            return np.array([])
        
        result = []
        for pt in points:
            bev_pt = self.transform_point((pt[0], pt[1]))
            result.append(bev_pt)
        
        return np.array(result, dtype=np.int32)
    
    def transform_box_to_point(self, box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Chuyển đổi bounding box thành điểm đại diện trên BEV
        Sử dụng điểm giữa cạnh dưới của box (vị trí xe trên mặt đường)
        
        Args:
            box: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Tọa độ điểm trên BEV
        """
        x1, y1, x2, y2 = box
        center_bottom = (int((x1 + x2) / 2), int(y2))
        return self.transform_point(center_bottom)
    
    def is_point_in_roi(self, point: Tuple[int, int]) -> bool:
        """Kiểm tra điểm có nằm trong ROI không"""
        if self.roi_polygon is None:
            return True
        result = cv2.pointPolygonTest(self.roi_polygon, point, False)
        return result >= 0
    
    def get_calibration_info(self) -> dict:
        """Trả về thông tin calibration hiện tại"""
        return {
            'camera_height': self.camera_height,
            'pitch_angle_deg': np.degrees(self.pitch_angle),
            'focal_length': self.focal_length,
            'principal_point': (self.cx, self.cy),
            'vanishing_point': self._vanishing_point,
            'is_calibrated': self._is_calibrated
        }
    
    def draw_calibration_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Vẽ overlay hiển thị thông tin calibration
        
        Args:
            frame: Frame để vẽ
            
        Returns:
            Frame với overlay
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Vẽ horizon line
        horizon_y = int(self.cy - self.focal_length * np.tan(self.pitch_angle))
        if 0 < horizon_y < h:
            cv2.line(result, (0, horizon_y), (w, horizon_y), (0, 255, 255), 2)
            cv2.putText(result, "Horizon", (10, horizon_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Vẽ vanishing point
        if self._vanishing_point:
            vp = self._vanishing_point
            cv2.circle(result, vp, 10, (0, 0, 255), 2)
            cv2.circle(result, vp, 3, (0, 0, 255), -1)
            cv2.putText(result, "VP", (vp[0] + 15, vp[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Vẽ grid trên mặt đất để kiểm tra
        grid_color = (100, 100, 100)
        for distance in [5, 10, 20, 30, 50]:  # meters
            # Tìm y trong ảnh tương ứng với distance này
            theta = np.arctan(self.camera_height / distance)
            img_y = self.cy + self.focal_length * np.tan(self.pitch_angle - theta + np.pi/2)
            img_y = int(img_y)
            
            if 0 < img_y < h:
                cv2.line(result, (0, img_y), (w, img_y), grid_color, 1)
                cv2.putText(result, f"{distance}m", (w - 50, img_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)
        
        # Hiển thị thông tin calibration
        info = [
            f"Pitch: {np.degrees(self.pitch_angle):.1f} deg",
            f"Focal: {self.focal_length:.0f} px",
            f"Height: {self.camera_height:.1f} m"
        ]
        y_pos = 30
        for text in info:
            cv2.putText(result, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
        
        return result


class IPMBirdEyeViewVisualizer:
    """
    Visualizer cho IPM Bird's Eye View
    Tương tự BirdEyeViewVisualizer nhưng sử dụng IPM transformer
    Hỗ trợ hiển thị valid/invalid zone để phát hiện xe đi sai làn
    """
    
    # Palette màu mặc định cho các valid zones
    DEFAULT_ZONE_COLORS = [
        (0, 150, 0),    # Green
        (150, 100, 0),  # Teal/Blue-green (BGR)
        (0, 150, 150),  # Yellow-ish
        (150, 0, 150),  # Purple
        (150, 0, 0),    # Blue
        (0, 100, 150),  # Orange
        (100, 0, 100),  # Dark purple
        (100, 100, 0),  # Dark teal
    ]
    
    def __init__(
        self,
        transformer: IPMBirdEyeViewTransformer,
        bg_color: Tuple[int, int, int] = (30, 30, 35),
        grid_color: Tuple[int, int, int] = (50, 50, 55),
        lane_color: Tuple[int, int, int] = (80, 80, 85),
        vehicle_colors: Optional[dict] = None,
        history_length: int = 15,
        show_grid: bool = True,
        show_distance_markers: bool = True,
        # Zone visualization parameters - support both single and multiple zones
        valid_zone_polygon: np.ndarray = None,  # Single zone (backward compatibility)
        valid_zone_polygons: List[np.ndarray] = None,  # Multiple zones
        show_zones: bool = True,
        valid_zone_color: Tuple[int, int, int] = (0, 120, 0),      # Xanh lá đậm
        valid_zone_colors: List[Tuple[int, int, int]] = None,      # Colors for multiple zones
        invalid_zone_color: Tuple[int, int, int] = (0, 0, 120),    # Đỏ đậm
        zone_alpha: float = 0.5
    ):
        """
        Khởi tạo IPM BEV Visualizer
        
        Args:
            transformer: IPM transformer
            bg_color: Màu nền
            grid_color: Màu grid
            lane_color: Màu làn đường
            vehicle_colors: Dict mapping class_id -> color
            history_length: Số frame lưu lịch sử
            show_grid: Hiển thị grid
            show_distance_markers: Hiển thị markers khoảng cách
            valid_zone_polygon: Single polygon vùng đường hợp lệ (backward compatibility)
            valid_zone_polygons: List các polygons vùng đường hợp lệ (từ camera view)
            show_zones: Hiển thị vùng valid/invalid
            valid_zone_color: Màu vùng hợp lệ mặc định (xanh lá)
            valid_zone_colors: List màu cho từng zone (None = dùng default palette)
            invalid_zone_color: Màu vùng không hợp lệ (đỏ)
            zone_alpha: Độ trong suốt của vùng zone (0.0-1.0)
        """
        self.transformer = transformer
        self.bg_color = bg_color
        self.grid_color = grid_color
        self.lane_color = lane_color
        self.show_grid = show_grid
        self.show_distance_markers = show_distance_markers
        self.history_length = history_length
        
        # Zone visualization - support multiple zones
        # Convert single polygon to list for unified handling
        if valid_zone_polygons is not None:
            self.valid_zone_polygons = valid_zone_polygons  # Camera view coordinates
        elif valid_zone_polygon is not None:
            self.valid_zone_polygons = [valid_zone_polygon]  # Wrap single polygon
        else:
            self.valid_zone_polygons = []
        
        self.show_zones = show_zones
        self.invalid_zone_color = invalid_zone_color
        self.zone_alpha = zone_alpha
        
        # Setup colors for each zone
        num_zones = len(self.valid_zone_polygons)
        if valid_zone_colors is not None:
            self.valid_zone_colors = valid_zone_colors
        elif num_zones == 1:
            self.valid_zone_colors = [valid_zone_color]  # Use provided single color
        else:
            # Use default palette for multiple zones
            self.valid_zone_colors = [self.DEFAULT_ZONE_COLORS[i % len(self.DEFAULT_ZONE_COLORS)] 
                                      for i in range(num_zones)]
        
        # Transform valid zone polygons to BEV coordinates
        self.bev_valid_zone_polygons: List[np.ndarray] = []
        if self.valid_zone_polygons:
            self._transform_valid_zones()
        
        # Backward compatibility: single polygon property
        self.valid_zone_polygon = self.valid_zone_polygons[0] if self.valid_zone_polygons else None
        self.bev_valid_zone_polygon = self.bev_valid_zone_polygons[0] if self.bev_valid_zone_polygons else None
        
        self.position_history: dict = {}
        self.trail_history: dict = {}  # Lưu trail để vẽ
        
        self.vehicle_colors = vehicle_colors or {
            0: (100, 255, 100),   # person - xanh lá
            1: (255, 200, 100),   # bicycle - cam
            2: (100, 100, 255),   # car - đỏ
            3: (255, 100, 255),   # motorcycle - tím
            5: (100, 255, 255),   # bus - vàng
            7: (255, 100, 100),   # truck - xanh dương
        }
        self.default_color = (200, 200, 200)
        
        self._create_base_image()
    
    def _transform_valid_zones(self):
        """
        Transform tất cả valid zone polygons từ camera view sang BEV coordinates
        """
        self.bev_valid_zone_polygons = []
        
        for zone_polygon in self.valid_zone_polygons:
            if zone_polygon is None:
                continue
            
            bev_points = []
            polygon = zone_polygon.reshape(-1, 2)
            
            for point in polygon:
                bev_pt = self.transformer.transform_point((int(point[0]), int(point[1])))
                if bev_pt != (-1, -1):  # Valid point
                    # Clamp to BEV bounds
                    bev_x = max(0, min(self.transformer.bev_width - 1, bev_pt[0]))
                    bev_y = max(0, min(self.transformer.bev_height - 1, bev_pt[1]))
                    bev_points.append([bev_x, bev_y])
            
            if len(bev_points) >= 3:
                self.bev_valid_zone_polygons.append(np.array(bev_points, dtype=np.int32))
        
        # Update backward compatibility properties
        self.bev_valid_zone_polygon = self.bev_valid_zone_polygons[0] if self.bev_valid_zone_polygons else None
    
    def set_valid_zone(self, valid_zone_polygon: np.ndarray):
        """
        Đặt single valid zone polygon mới (backward compatibility)
        
        Args:
            valid_zone_polygon: Polygon vùng đường hợp lệ (từ camera view)
        """
        self.valid_zone_polygons = [valid_zone_polygon] if valid_zone_polygon is not None else []
        self.valid_zone_polygon = valid_zone_polygon
        self.valid_zone_colors = [self.DEFAULT_ZONE_COLORS[0]]
        self._transform_valid_zones()
        self._create_base_image()
    
    def set_valid_zones(self, valid_zone_polygons: List[np.ndarray], 
                        zone_colors: List[Tuple[int, int, int]] = None):
        """
        Đặt multiple valid zone polygons mới và cập nhật BEV visualization
        
        Args:
            valid_zone_polygons: List các polygons vùng đường hợp lệ (từ camera view)
            zone_colors: List màu cho từng zone (None = dùng default palette)
        """
        self.valid_zone_polygons = valid_zone_polygons if valid_zone_polygons else []
        
        # Update backward compatibility
        self.valid_zone_polygon = self.valid_zone_polygons[0] if self.valid_zone_polygons else None
        
        # Setup colors
        num_zones = len(self.valid_zone_polygons)
        if zone_colors is not None:
            self.valid_zone_colors = zone_colors
        else:
            self.valid_zone_colors = [self.DEFAULT_ZONE_COLORS[i % len(self.DEFAULT_ZONE_COLORS)] 
                                      for i in range(num_zones)]
        
        self._transform_valid_zones()
        self._create_base_image()
    
    def is_in_valid_zone(self, bev_point: Tuple[int, int]) -> bool:
        """
        Kiểm tra một điểm có nằm trong bất kỳ vùng valid (làn đường) nào hay không
        
        Args:
            bev_point: Tọa độ (x, y) trên BEV
            
        Returns:
            True nếu điểm nằm trong ít nhất một valid zone, False nếu ngoài tất cả
        """
        if not self.bev_valid_zone_polygons:
            return True  # Không có zone -> mặc định là valid
        
        for bev_polygon in self.bev_valid_zone_polygons:
            if bev_polygon is not None and len(bev_polygon) >= 3:
                result = cv2.pointPolygonTest(
                    bev_polygon, 
                    (float(bev_point[0]), float(bev_point[1])), 
                    False
                )
                if result >= 0:
                    return True
        return False
    
    def get_zone_index(self, bev_point: Tuple[int, int]) -> int:
        """
        Lấy index của zone chứa điểm trên BEV
        
        Args:
            bev_point: Tọa độ (x, y) trên BEV
            
        Returns:
            Index của zone chứa điểm, hoặc -1 nếu không nằm trong zone nào
        """
        for i, bev_polygon in enumerate(self.bev_valid_zone_polygons):
            if bev_polygon is not None and len(bev_polygon) >= 3:
                result = cv2.pointPolygonTest(
                    bev_polygon, 
                    (float(bev_point[0]), float(bev_point[1])), 
                    False
                )
                if result >= 0:
                    return i
        return -1
    
    def get_zone_status(self, bev_point: Tuple[int, int]) -> str:
        """
        Lấy trạng thái vùng của một điểm trên BEV
        
        Args:
            bev_point: Tọa độ (x, y) trên BEV
            
        Returns:
            "valid" nếu trong làn đường, "invalid" nếu ngoài làn đường
        """
        return "valid" if self.is_in_valid_zone(bev_point) else "invalid"
    
    def _create_base_image(self):
        """Tạo ảnh nền BEV với grid, markers và valid/invalid zones"""
        w, h = self.transformer.bev_width, self.transformer.bev_height
        self.base_image = np.full((h, w, 3), self.bg_color, dtype=np.uint8)
        
        center_x = self.transformer.bev_center_x
        origin_y = self.transformer.bev_origin_y
        scale = self.transformer.bev_scale
        
        # Vẽ valid/invalid zones trước grid
        if self.show_zones and self.bev_valid_zone_polygons:
            self._draw_zones()
        
        if self.show_grid:
            # Vẽ grid dọc (mỗi 1 meter)
            for x_offset in range(-10, 11):
                x = int(center_x + x_offset * scale)
                if 0 <= x < w:
                    color = self.grid_color if x_offset != 0 else (70, 70, 75)
                    thickness = 1 if x_offset != 0 else 2
                    cv2.line(self.base_image, (x, 0), (x, h), color, thickness)
            
            # Vẽ grid ngang (mỗi 5 meters)
            for distance in range(0, 60, 5):
                y = int(origin_y - distance * scale)
                if 0 <= y < h:
                    cv2.line(self.base_image, (0, y), (w, y), self.grid_color, 1)
        
        if self.show_distance_markers:
            # Markers khoảng cách
            for distance in [5, 10, 20, 30, 40, 50]:
                y = int(origin_y - distance * scale)
                if 10 < y < h - 10:
                    cv2.putText(self.base_image, f"{distance}m", (5, y + 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 125), 1)
        
        # Vẽ vị trí camera
        cam_y = int(origin_y)
        cv2.circle(self.base_image, (int(center_x), cam_y), 5, (100, 200, 255), -1)
        cv2.putText(self.base_image, "CAM", (int(center_x) - 15, cam_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 255), 1)
        
        # Header
        header = "IPM Bird's Eye View"
        (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.putText(self.base_image, header, ((w - tw) // 2, 20),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (180, 180, 185), 1)
        
        # Vẽ zone labels nếu có zones
        if self.show_zones and self.bev_valid_zone_polygons:
            self._draw_zone_labels()
    
    def _draw_zones(self):
        """
        Vẽ tất cả các vùng valid (mỗi zone 1 màu) và invalid (đỏ) lên base image
        """
        w, h = self.transformer.bev_width, self.transformer.bev_height
        
        # Tạo combined mask cho tất cả valid zones
        combined_valid_mask = np.zeros((h, w), dtype=np.uint8)
        for bev_polygon in self.bev_valid_zone_polygons:
            if bev_polygon is not None and len(bev_polygon) >= 3:
                cv2.fillPoly(combined_valid_mask, [bev_polygon], 255)
        
        invalid_mask = cv2.bitwise_not(combined_valid_mask)
        
        # Tạo overlay cho invalid zone (toàn bộ nền) - màu đỏ
        invalid_overlay = np.full_like(self.base_image, self.invalid_zone_color, dtype=np.uint8)
        
        # Blend invalid zone (vùng ngoài tất cả làn đường)
        invalid_region = cv2.bitwise_and(invalid_overlay, invalid_overlay, mask=invalid_mask)
        self.base_image = cv2.addWeighted(
            self.base_image, 1 - self.zone_alpha,
            invalid_region, self.zone_alpha,
            0
        )
        
        # Vẽ từng valid zone với màu riêng
        for i, bev_polygon in enumerate(self.bev_valid_zone_polygons):
            if bev_polygon is None or len(bev_polygon) < 3:
                continue
            
            zone_color = self.valid_zone_colors[i] if i < len(self.valid_zone_colors) else self.DEFAULT_ZONE_COLORS[0]
            
            # Tạo mask cho zone này
            zone_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(zone_mask, [bev_polygon], 255)
            
            # Tạo overlay cho zone này
            valid_overlay = np.zeros_like(self.base_image, dtype=np.uint8)
            cv2.fillPoly(valid_overlay, [bev_polygon], zone_color)
            
            # Blend valid zone
            valid_region = cv2.bitwise_and(valid_overlay, valid_overlay, mask=zone_mask)
            base_valid = cv2.bitwise_and(self.base_image, self.base_image, mask=zone_mask)
            blended_valid = cv2.addWeighted(base_valid, 0.4, valid_region, 0.6, 0)
            
            # Combine: giữ phần ngoài zone và thêm phần valid đã blend
            inv_zone_mask = cv2.bitwise_not(zone_mask)
            self.base_image = cv2.bitwise_and(self.base_image, self.base_image, mask=inv_zone_mask)
            self.base_image = cv2.add(self.base_image, blended_valid)
            
            # Vẽ viền phân cách giữa valid và invalid zone
            cv2.polylines(self.base_image, [bev_polygon], True, (255, 255, 255), 2)
    
    def _draw_zone_labels(self):
        """
        Vẽ nhãn cho các vùng valid (từng zone) và invalid
        """
        w, h = self.transformer.bev_width, self.transformer.bev_height
        
        # Vẽ label cho từng valid zone
        for i, bev_polygon in enumerate(self.bev_valid_zone_polygons):
            if bev_polygon is None or len(bev_polygon) < 3:
                continue
            
            zone_color = self.valid_zone_colors[i] if i < len(self.valid_zone_colors) else self.DEFAULT_ZONE_COLORS[0]
            
            # Tính tâm của zone
            M = cv2.moments(bev_polygon)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                centroid = np.mean(bev_polygon, axis=0)
                cx, cy = int(centroid[0]), int(centroid[1])
            
            # Label "Zone X" ở giữa mỗi valid zone
            zone_label = f"Zone {i + 1}" if len(self.bev_valid_zone_polygons) > 1 else "VALID"
            (vl_w, vl_h), _ = cv2.getTextSize(zone_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            cv2.rectangle(self.base_image,
                         (cx - vl_w // 2 - 4, cy - vl_h // 2 - 4),
                         (cx + vl_w // 2 + 4, cy + vl_h // 2 + 4),
                         zone_color, -1)
            cv2.rectangle(self.base_image,
                         (cx - vl_w // 2 - 4, cy - vl_h // 2 - 4),
                         (cx + vl_w // 2 + 4, cy + vl_h // 2 + 4),
                         (255, 255, 255), 1)
            cv2.putText(self.base_image, zone_label,
                       (cx - vl_w // 2, cy + vl_h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Label "INVALID" ở bên trái
        invalid_label = "INVALID"
        (il_w, il_h), _ = cv2.getTextSize(invalid_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        left_x = 25
        left_y = h // 2
        
        cv2.rectangle(self.base_image,
                     (left_x - il_w // 2 - 3, left_y - il_h // 2 - 3),
                     (left_x + il_w // 2 + 3, left_y + il_h // 2 + 3),
                     (0, 0, 150), -1)
        cv2.rectangle(self.base_image,
                     (left_x - il_w // 2 - 3, left_y - il_h // 2 - 3),
                     (left_x + il_w // 2 + 3, left_y + il_h // 2 + 3),
                     (255, 255, 255), 1)
        cv2.putText(self.base_image, invalid_label,
                   (left_x - il_w // 2, left_y + il_h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Label "INVALID" ở bên phải
        right_x = w - 25
        cv2.rectangle(self.base_image,
                     (right_x - il_w // 2 - 3, left_y - il_h // 2 - 3),
                     (right_x + il_w // 2 + 3, left_y + il_h // 2 + 3),
                     (0, 0, 150), -1)
        cv2.rectangle(self.base_image,
                     (right_x - il_w // 2 - 3, left_y - il_h // 2 - 3),
                     (right_x + il_w // 2 + 3, left_y + il_h // 2 + 3),
                     (255, 255, 255), 1)
        cv2.putText(self.base_image, invalid_label,
                   (right_x - il_w // 2, left_y + il_h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    def _draw_legend(self, bev_image: np.ndarray, valid_count: int, invalid_count: int, total_count: int):
        """
        Vẽ legend box với thông tin chi tiết về zone và thống kê
        
        Args:
            bev_image: Ảnh BEV để vẽ lên
            valid_count: Số xe trong vùng valid
            invalid_count: Số xe trong vùng invalid
            total_count: Tổng số xe
        """
        w = self.transformer.bev_width
        legend_width = 110
        legend_height = 100
        legend_x = w - legend_width - 8
        legend_y = 35
        
        # Vẽ background legend
        overlay = bev_image.copy()
        cv2.rectangle(overlay, 
                     (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height),
                     (25, 25, 28), -1)
        cv2.addWeighted(overlay, 0.85, bev_image, 0.15, 0, bev_image)
        
        # Viền legend
        cv2.rectangle(bev_image, 
                     (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height),
                     (255, 255, 255), 1)
        
        # Title
        cv2.putText(bev_image, "ZONE INFO", 
                   (legend_x + 15, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Đường kẻ phân cách
        cv2.line(bev_image, (legend_x + 5, legend_y + 22), 
                (legend_x + legend_width - 5, legend_y + 22), (80, 80, 85), 1)
        
        # Valid zone indicator
        cv2.rectangle(bev_image, 
                     (legend_x + 6, legend_y + 28), 
                     (legend_x + 18, legend_y + 40), 
                     (0, 180, 0), -1)
        cv2.putText(bev_image, f"Valid: {valid_count}", 
                   (legend_x + 24, legend_y + 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
        # Invalid zone indicator
        cv2.rectangle(bev_image, 
                     (legend_x + 6, legend_y + 46), 
                     (legend_x + 18, legend_y + 58), 
                     (0, 0, 180), -1)
        cv2.putText(bev_image, f"Invalid: {invalid_count}", 
                   (legend_x + 24, legend_y + 56),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        # Đường kẻ phân cách
        cv2.line(bev_image, (legend_x + 5, legend_y + 64), 
                (legend_x + legend_width - 5, legend_y + 64), (80, 80, 85), 1)
        
        # Total count
        cv2.putText(bev_image, f"Total: {total_count}", 
                   (legend_x + 18, legend_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Warning nếu có xe invalid
        if invalid_count > 0:
            cv2.putText(bev_image, "! WARNING", 
                       (legend_x + 18, legend_y + 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    def get_vehicle_color(self, class_id: int, tracker_id: int = None) -> Tuple[int, int, int]:
        """Lấy màu cho xe"""
        if tracker_id is not None:
            np.random.seed(tracker_id * 10)
            return tuple(np.random.randint(120, 240, 3).tolist())
        return self.vehicle_colors.get(class_id, self.default_color)
    
    def update_position_history(self, tracker_id: int, position: Tuple[int, int]):
        """Cập nhật lịch sử vị trí"""
        if tracker_id not in self.position_history:
            self.position_history[tracker_id] = []
            self.trail_history[tracker_id] = []
        
        self.position_history[tracker_id].append(position)
        self.trail_history[tracker_id].append(position)
        
        if len(self.position_history[tracker_id]) > self.history_length:
            self.position_history[tracker_id].pop(0)
        if len(self.trail_history[tracker_id]) > self.history_length * 3:
            self.trail_history[tracker_id].pop(0)
    
    def get_movement_direction(self, tracker_id: int) -> Optional[Tuple[float, float]]:
        """Tính hướng di chuyển của xe"""
        if tracker_id not in self.position_history:
            return None
        
        history = self.position_history[tracker_id]
        if len(history) < 2:
            return None
        
        old_pos = history[0]
        new_pos = history[-1]
        
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]
        
        length = np.sqrt(dx * dx + dy * dy)
        if length < 2:
            return None
        
        return (dx / length, dy / length)
    
    def clean_old_tracks(self, current_tracker_ids: List[int]):
        """Xóa lịch sử của các xe không còn được track"""
        ids_to_remove = [tid for tid in self.position_history.keys() 
                        if tid not in current_tracker_ids]
        for tid in ids_to_remove:
            del self.position_history[tid]
            if tid in self.trail_history:
                del self.trail_history[tid]
    
    def draw(
        self,
        detections: sv.Detections,
        class_names: Optional[dict] = None,
        show_ids: bool = True,
        show_labels: bool = True,  # Backwards compatibility with BirdEyeViewVisualizer
        show_trails: bool = True,
        show_speed: bool = False,
        show_zone_stats: bool = True
    ) -> np.ndarray:
        """
        Vẽ BEV với các xe từ detections
        
        Args:
            detections: Supervision Detections object
            class_names: Dict mapping class_id -> name
            show_ids: Hiển thị tracker IDs
            show_labels: Hiển thị class labels (backwards compatibility)
            show_trails: Hiển thị vệt di chuyển
            show_speed: Hiển thị tốc độ ước tính
            show_zone_stats: Hiển thị thống kê số xe trong từng zone
            
        Returns:
            BEV image
        """
        bev_image = self.base_image.copy()
        
        if detections is None or len(detections) == 0:
            # Hiển thị legend trống nếu có zones
            if show_zone_stats and self.show_zones and self.bev_valid_zone_polygons:
                self._draw_legend(bev_image, 0, 0, 0)
            return bev_image
        
        boxes = detections.xyxy
        class_ids = detections.class_id
        tracker_ids = detections.tracker_id if detections.tracker_id is not None else [None] * len(boxes)
        
        valid_tracker_ids = [tid for tid in tracker_ids if tid is not None]
        self.clean_old_tracks(valid_tracker_ids)
        
        # Đếm số xe trong từng zone
        valid_count = 0
        invalid_count = 0
        
        # Vẽ trails trước
        if show_trails:
            for tracker_id, trail in self.trail_history.items():
                if len(trail) >= 2:
                    color = self.get_vehicle_color(0, tracker_id)
                    # Fade effect cho trail
                    for i in range(1, len(trail)):
                        alpha = i / len(trail)
                        pt1 = trail[i-1]
                        pt2 = trail[i]
                        if (0 <= pt1[0] < self.transformer.bev_width and 
                            0 <= pt1[1] < self.transformer.bev_height and
                            0 <= pt2[0] < self.transformer.bev_width and 
                            0 <= pt2[1] < self.transformer.bev_height):
                            trail_color = tuple(int(c * alpha * 0.5) for c in color)
                            cv2.line(bev_image, pt1, pt2, trail_color, 1, cv2.LINE_AA)
        
        # Vẽ các xe
        for i, (box, class_id, tracker_id) in enumerate(zip(boxes, class_ids, tracker_ids)):
            try:
                bev_point = self.transformer.transform_box_to_point(box)
            except:
                continue
            
            if bev_point == (-1, -1):
                continue
            
            if not (0 <= bev_point[0] < self.transformer.bev_width and 
                    0 <= bev_point[1] < self.transformer.bev_height):
                continue
            
            if tracker_id is not None:
                self.update_position_history(tracker_id, bev_point)
            
            color = self.get_vehicle_color(class_id, tracker_id)
            
            # Kiểm tra xe có nằm trong vùng valid hay không
            is_valid = self.is_in_valid_zone(bev_point)
            
            # Đếm số xe
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
            
            # Vẽ xe (hình chữ nhật với hướng)
            x, y = bev_point
            vw, vh = 16, 28  # vehicle width, height in BEV
            
            direction = None
            if tracker_id is not None:
                direction = self.get_movement_direction(tracker_id)
            
            if direction is not None:
                # Vẽ xe với hướng
                dx, dy = direction
                angle = np.arctan2(dx, -dy)  # Góc xoay
                
                # Tạo hình chữ nhật xoay
                rect_pts = np.array([
                    [-vw/2, -vh/2], [vw/2, -vh/2],
                    [vw/2, vh/2], [-vw/2, vh/2]
                ], dtype=np.float32)
                
                # Rotation matrix
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                
                rotated_pts = rect_pts @ rot_matrix.T
                rotated_pts += [x, y]
                rotated_pts = rotated_pts.astype(np.int32)
                
                cv2.fillPoly(bev_image, [rotated_pts], color)
                
                # Viền xe: trắng nếu valid, đỏ đậm nếu invalid
                if is_valid:
                    cv2.polylines(bev_image, [rotated_pts], True, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    # Xe nằm ngoài làn đường - vẽ viền đỏ và cảnh báo
                    cv2.polylines(bev_image, [rotated_pts], True, (0, 0, 255), 2, cv2.LINE_AA)
                    # Vẽ dấu ! cảnh báo phía trên xe
                    cv2.putText(bev_image, "!", (x - 4, y - vh // 2 - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Mũi tên hướng
                arrow_len = 12
                arrow_end = (int(x + dx * arrow_len), int(y + dy * arrow_len))
                cv2.arrowedLine(bev_image, (x, y), arrow_end, (255, 255, 255), 2, tipLength=0.4)
            else:
                # Xe không có hướng - vẽ hình chữ nhật thường
                pt1 = (x - vw // 2, y - vh // 2)
                pt2 = (x + vw // 2, y + vh // 2)
                cv2.rectangle(bev_image, pt1, pt2, color, -1)
                
                # Viền xe: trắng nếu valid, đỏ đậm nếu invalid
                if is_valid:
                    cv2.rectangle(bev_image, pt1, pt2, (255, 255, 255), 1)
                else:
                    # Xe nằm ngoài làn đường - vẽ viền đỏ và dấu X cảnh báo
                    cv2.rectangle(bev_image, pt1, pt2, (0, 0, 255), 2)
                    cv2.line(bev_image, pt1, pt2, (0, 0, 255), 2)
                    cv2.line(bev_image, (pt1[0], pt2[1]), (pt2[0], pt1[1]), (0, 0, 255), 2)
                
                cv2.circle(bev_image, (x, y), 2, (255, 255, 255), -1)
            
            # Label
            if show_ids and tracker_id is not None:
                label = f"#{tracker_id}"
                label_color = (255, 255, 255) if is_valid else (0, 0, 255)
                cv2.putText(bev_image, label, (x - vw // 2, y - vh // 2 - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1)
        
        # Thống kê
        vehicle_count = len(detections)
        
        if show_zone_stats and self.show_zones and self.bev_valid_zone_polygons:
            # Vẽ legend box với thông tin chi tiết
            self._draw_legend(bev_image, valid_count, invalid_count, vehicle_count)
            
            # Hiển thị thống kê nhanh ở phía dưới
            stats_y = self.transformer.bev_height - 10
            
            # Valid count (màu xanh lá, bên trái)
            valid_text = f"In Lane: {valid_count}"
            cv2.putText(bev_image, valid_text, (10, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Invalid count (màu đỏ, bên phải)
            invalid_text = f"Out Lane: {invalid_count}"
            (inv_w, _), _ = cv2.getTextSize(invalid_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(bev_image, invalid_text, 
                       (self.transformer.bev_width - inv_w - 10, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            # Hiển thị thống kê đơn giản
            count_text = f"Vehicles: {vehicle_count}"
            cv2.putText(bev_image, count_text, (10, self.transformer.bev_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 185), 1)
        
        return bev_image


def create_transformer(
    method: str,
    frame: np.ndarray,
    source_polygon: np.ndarray = None,
    camera_height: float = 1.5,
    **kwargs
) -> 'BirdEyeViewTransformer | IPMBirdEyeViewTransformer':
    """
    Factory function để tạo BEV transformer
    
    Args:
        method: "homography" hoặc "ipm"
        frame: Frame để calibrate (cho IPM)
        source_polygon: Polygon nguồn (bắt buộc cho homography)
        camera_height: Chiều cao camera (cho IPM)
        **kwargs: Các tham số khác
        
    Returns:
        Transformer phù hợp
    """
    h, w = frame.shape[:2]
    
    if method.lower() == "homography":
        if source_polygon is None:
            raise ValueError("source_polygon is required for homography method")
        return BirdEyeViewTransformer(source_polygon, **kwargs)
    
    elif method.lower() == "ipm":
        transformer = IPMBirdEyeViewTransformer(
            frame_width=w,
            frame_height=h,
            camera_height=camera_height,
            **kwargs
        )
        # Auto calibrate từ frame
        transformer.calibrate_from_frame(frame)
        return transformer
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'homography' or 'ipm'")
