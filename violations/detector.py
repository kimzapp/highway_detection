"""
Violation Detection Module
Hệ thống phát hiện các loại vi phạm giao thông

Hỗ trợ các loại vi phạm:
- WRONG_LANE: Đi sai làn đường
- ... (có thể mở rộng thêm)
"""

import cv2
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import supervision as sv


class ViolationType(Enum):
    """Enum định nghĩa các loại vi phạm"""
    WRONG_LANE = auto()           # Đi sai làn đường / ngoài vùng cho phép
    INVALID_VEHICLE = auto()      # Phương tiện không hợp lệ theo danh sách cấu hình
    # Có thể mở rộng thêm các loại vi phạm khác:
    # SPEEDING = auto()           # Vượt quá tốc độ cho phép
    # ILLEGAL_OVERTAKE = auto()   # Vượt xe trái phép
    # STOP_LINE_VIOLATION = auto() # Vượt vạch dừng


@dataclass
class Violation:
    """
    Đại diện cho một vi phạm cụ thể
    """
    violation_type: ViolationType
    tracker_id: int
    class_id: int
    class_name: str
    position: Tuple[int, int]      # Vị trí trên camera view
    bev_position: Tuple[int, int]  # Vị trí trên BEV
    frame_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    extra_info: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Chuyển đổi thành dict để serialize"""
        return {
            'type': self.violation_type.name,
            'tracker_id': self.tracker_id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'position': self.position,
            'bev_position': self.bev_position,
            'frame_number': self.frame_number,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'extra_info': self.extra_info
        }


@dataclass
class VehicleViolationState:
    """
    Trạng thái vi phạm của một xe theo thời gian
    """
    tracker_id: int
    class_id: int
    class_name: str
    
    # Lịch sử vi phạm
    violations: List[Violation] = field(default_factory=list)
    
    # Trạng thái hiện tại
    is_currently_violating: bool = False
    current_violation_type: Optional[ViolationType] = None
    violation_start_frame: Optional[int] = None
    
    # Số frame liên tiếp trong trạng thái vi phạm (để tránh false positive)
    consecutive_violation_frames: int = 0
    consecutive_normal_frames: int = 0
    
    # Vị trí cuối cùng
    last_position: Optional[Tuple[int, int]] = None
    last_bev_position: Optional[Tuple[int, int]] = None
    
class ViolationDetector:
    """
    Bộ phát hiện vi phạm tổng hợp
    
    Kiến trúc mở rộng cho phép thêm các loại vi phạm mới dễ dàng.
    """
    
    def __init__(
        self,
        min_violation_frames: int = 45,
        min_normal_frames: int = 3,
        enabled_violations: Optional[Set[ViolationType]] = None,
        valid_vehicle_class_ids: Optional[Set[int]] = None,
    ):
        """
        Khởi tạo ViolationDetector
        
        Args:
            min_violation_frames: Số frame tối thiểu để xác nhận vi phạm
            min_normal_frames: Số frame tối thiểu để xác nhận hết vi phạm
            enabled_violations: Set các loại vi phạm được bật (None = tất cả)
        """
        self.min_violation_frames = min_violation_frames
        self.min_normal_frames = min_normal_frames
        self.valid_vehicle_class_ids: Set[int] = set(valid_vehicle_class_ids or {2})
        
        # Mặc định bật tất cả các loại vi phạm
        self.enabled_violations = enabled_violations or {v for v in ViolationType}
        
        # Lưu trạng thái vi phạm của từng xe
        self._vehicle_states: Dict[int, VehicleViolationState] = {}
        
        # Thống kê tổng hợp
        self._total_violations: Dict[ViolationType, int] = {v: 0 for v in ViolationType}
        self._violations_log: List[Violation] = []
        
        # Valid zone polygon cho WrongLane detection
        self._valid_zone_polygon: Optional[np.ndarray] = None
        self._valid_zone_polygons: List[np.ndarray] = []  # Hỗ trợ nhiều zone
        
        # BEV transformer reference (để transform điểm)
        self._bev_transformer = None
        
        # Frame hiện tại
        self._current_frame = 0
        
    def set_valid_zones(self, zone_polygons: List[np.ndarray]):
        """
        Đặt các vùng hợp lệ cho xe di chuyển
        
        Args:
            zone_polygons: Danh sách các polygon vùng hợp lệ
        """
        self._valid_zone_polygons = zone_polygons
        if zone_polygons:
            # Tạo combined polygon từ tất cả các zone (union)
            all_points = np.vstack(zone_polygons)
            self._valid_zone_polygon = cv2.convexHull(all_points)
    
    def set_bev_transformer(self, transformer):
        """
        Đặt BEV transformer để sử dụng cho việc transform điểm
        
        Args:
            transformer: BEV transformer instance
        """
        self._bev_transformer = transformer
    
    def enable_violation(self, violation_type: ViolationType):
        """Bật phát hiện một loại vi phạm"""
        self.enabled_violations.add(violation_type)
    
    def disable_violation(self, violation_type: ViolationType):
        """Tắt phát hiện một loại vi phạm"""
        self.enabled_violations.discard(violation_type)

    def set_valid_vehicle_classes(self, class_ids: Set[int]):
        """Đặt danh sách class phương tiện hợp lệ (không bị gắn cờ INVALID_VEHICLE)."""
        self.valid_vehicle_class_ids = set(class_ids or {2})
    
    def is_point_in_valid_zone(self, point: Tuple[int, int]) -> bool:
        """
        Kiểm tra một điểm có nằm trong vùng hợp lệ không
        
        Args:
            point: Tọa độ (x, y) trên camera view
            
        Returns:
            True nếu điểm nằm trong vùng hợp lệ
        """
        if not self._valid_zone_polygons:
            return True  # Không có zone nào được định nghĩa = tất cả đều hợp lệ
        
        # Kiểm tra với từng polygon
        for polygon in self._valid_zone_polygons:
            result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
            if result >= 0:  # Bên trong hoặc trên biên
                return True
        
        return False
    
    def _get_vehicle_position(self, box: np.ndarray) -> Tuple[int, int]:
        """
        Lấy vị trí đại diện của xe từ bounding box
        
        Sử dụng điểm giữa cạnh dưới (vị trí xe trên mặt đường)
        """
        x1, y1, x2, y2 = box
        center_bottom = (int((x1 + x2) / 2), int(y2))
        return center_bottom
    
    def _check_wrong_lane(
        self,
        tracker_id: int,
        position: Tuple[int, int],
        state: VehicleViolationState
    ) -> Optional[ViolationType]:
        """
        Kiểm tra vi phạm đi sai làn đường
        
        Args:
            tracker_id: ID của xe
            position: Vị trí xe trên camera view
            state: Trạng thái vi phạm của xe
            
        Returns:
            ViolationType.WRONG_LANE nếu vi phạm, None nếu không
        """
        if ViolationType.WRONG_LANE not in self.enabled_violations:
            return None
        
        if not self._valid_zone_polygons:
            return None  # Không có zone được định nghĩa
        
        is_in_valid_zone = self.is_point_in_valid_zone(position)
        
        if not is_in_valid_zone:
            return ViolationType.WRONG_LANE
        
        return None

    def _check_invalid_vehicle(self, class_id: int) -> Optional[ViolationType]:
        """Kiểm tra phương tiện không hợp lệ theo danh sách class được cấu hình."""
        if ViolationType.INVALID_VEHICLE not in self.enabled_violations:
            return None

        if class_id not in self.valid_vehicle_class_ids:
            return ViolationType.INVALID_VEHICLE

        return None
    
    def update(
        self,
        detections: sv.Detections,
        class_names: Dict[int, str],
        frame_number: int
    ) -> Dict[int, List[ViolationType]]:
        """
        Cập nhật và phát hiện vi phạm cho tất cả xe trong frame
        
        Args:
            detections: Supervision Detections với tracker IDs
            class_names: Dict mapping class_id -> class name
            frame_number: Số frame hiện tại
            
        Returns:
            Dict mapping tracker_id -> list of current ViolationType
        """
        self._current_frame = frame_number
        
        if detections is None or len(detections) == 0:
            return {}
        
        # Lấy thông tin detections
        boxes = detections.xyxy
        class_ids = detections.class_id
        tracker_ids = detections.tracker_id
        
        if tracker_ids is None:
            return {}
        
        current_violations: Dict[int, List[ViolationType]] = {}
        active_tracker_ids = set()
        
        for i, (box, class_id, tracker_id) in enumerate(zip(boxes, class_ids, tracker_ids)):
            if tracker_id is None:
                continue
            
            active_tracker_ids.add(tracker_id)
            class_name = class_names.get(class_id, f"class_{class_id}")
            
            # Lấy vị trí xe
            position = self._get_vehicle_position(box)
            
            # Tính BEV position nếu có transformer
            bev_position = (-1, -1)
            if self._bev_transformer is not None:
                try:
                    bev_position = self._bev_transformer.transform_point(position)
                except:
                    pass
            
            # Lấy hoặc tạo state cho xe
            if tracker_id not in self._vehicle_states:
                self._vehicle_states[tracker_id] = VehicleViolationState(
                    tracker_id=tracker_id,
                    class_id=class_id,
                    class_name=class_name
                )
            
            state = self._vehicle_states[tracker_id]
            state.last_position = position
            state.last_bev_position = bev_position
            
            # Kiểm tra các loại vi phạm
            detected_violations = []

            # 1. Kiểm tra phương tiện không hợp lệ
            invalid_vehicle = self._check_invalid_vehicle(int(class_id))
            if invalid_vehicle:
                detected_violations.append(invalid_vehicle)
            
            # 2. Kiểm tra vi phạm sai làn
            wrong_lane = self._check_wrong_lane(tracker_id, position, state)
            if wrong_lane:
                detected_violations.append(wrong_lane)
            
            # 3. Có thể thêm các kiểm tra vi phạm khác ở đây:
            # speeding = self._check_speeding(tracker_id, position, state)
            # if speeding:
            #     detected_violations.append(speeding)
            
            # Cập nhật trạng thái vi phạm
            self._update_violation_state(
                state=state,
                detected_violations=detected_violations,
                position=position,
                bev_position=bev_position,
                frame_number=frame_number,
                class_name=class_name
            )
            
            # Lấy danh sách vi phạm hiện tại (đã được xác nhận)
            if state.is_currently_violating and state.current_violation_type:
                current_violations[tracker_id] = [state.current_violation_type]
            else:
                current_violations[tracker_id] = []
        
        # Dọn dẹp state của các xe không còn được track
        self._cleanup_old_states(active_tracker_ids)
        
        return current_violations
    
    def _update_violation_state(
        self,
        state: VehicleViolationState,
        detected_violations: List[ViolationType],
        position: Tuple[int, int],
        bev_position: Tuple[int, int],
        frame_number: int,
        class_name: str
    ):
        """
        Cập nhật trạng thái vi phạm của xe
        
        Sử dụng cơ chế đếm frame để tránh false positive
        """
        if detected_violations:
            # Có phát hiện vi phạm
            violation_type = detected_violations[0]  # Ưu tiên vi phạm đầu tiên
            
            state.consecutive_violation_frames += 1
            state.consecutive_normal_frames = 0
            
            if not state.is_currently_violating:
                # Kiểm tra đủ số frame để xác nhận vi phạm
                if state.consecutive_violation_frames >= self.min_violation_frames:
                    state.is_currently_violating = True
                    state.current_violation_type = violation_type
                    state.violation_start_frame = frame_number - self.min_violation_frames + 1
                    
                    # Tạo record vi phạm
                    violation = Violation(
                        violation_type=violation_type,
                        tracker_id=state.tracker_id,
                        class_id=state.class_id,
                        class_name=class_name,
                        position=position,
                        bev_position=bev_position,
                        frame_number=state.violation_start_frame
                    )
                    state.violations.append(violation)
                    self._violations_log.append(violation)
                    self._total_violations[violation_type] += 1
        else:
            # Không có vi phạm
            state.consecutive_normal_frames += 1
            state.consecutive_violation_frames = 0
            
            if state.is_currently_violating:
                # Kiểm tra đủ số frame để xác nhận hết vi phạm
                if state.consecutive_normal_frames >= self.min_normal_frames:
                    state.is_currently_violating = False
                    state.current_violation_type = None
                    state.violation_start_frame = None
    
    def _cleanup_old_states(self, active_tracker_ids: Set[int]):
        """Xóa state của các xe không còn được track"""
        ids_to_remove = [tid for tid in self._vehicle_states.keys() 
                        if tid not in active_tracker_ids]
        for tid in ids_to_remove:
            del self._vehicle_states[tid]
    
    def get_vehicle_state(self, tracker_id: int) -> Optional[VehicleViolationState]:
        """Lấy trạng thái vi phạm của một xe"""
        return self._vehicle_states.get(tracker_id)
    
    def get_violating_vehicles(self) -> List[int]:
        """Lấy danh sách ID các xe đang vi phạm"""
        return [
            tid for tid, state in self._vehicle_states.items()
            if state.is_currently_violating
        ]
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê tổng hợp về vi phạm"""
        return {
            'total_violations': dict(self._total_violations),
            'current_violating_count': len(self.get_violating_vehicles()),
            'tracked_vehicles': len(self._vehicle_states),
            'violations_log_count': len(self._violations_log)
        }
    
    def get_violations_log(self) -> List[Violation]:
        """Lấy log tất cả các vi phạm đã ghi nhận"""
        return self._violations_log.copy()
    
    def reset(self):
        """Reset toàn bộ trạng thái"""
        self._vehicle_states.clear()
        self._total_violations = {v: 0 for v in ViolationType}
        self._violations_log.clear()
        self._current_frame = 0


class ViolationVisualizer:
    """
    Hiển thị thông tin vi phạm lên frame
    """
    
    # Màu sắc cho từng loại vi phạm (BGR)
    VIOLATION_COLORS: Dict[ViolationType, Tuple[int, int, int]] = {
        ViolationType.WRONG_LANE: (0, 0, 255),         # Đỏ
        ViolationType.INVALID_VEHICLE: (0, 165, 255),  # Cam
        # Thêm màu cho các loại vi phạm khác khi cần
    }
    
    # Tên hiển thị cho từng loại vi phạm
    VIOLATION_NAMES: Dict[ViolationType, str] = {
        ViolationType.WRONG_LANE: "SAI LAN DUONG",
        ViolationType.INVALID_VEHICLE: "PHUONG TIEN KHONG HOP LE",
        # Thêm tên cho các loại vi phạm khác khi cần
    }
    
    def __init__(
        self,
        detector: ViolationDetector,
        show_violation_box: bool = True,
        show_violation_label: bool = True,
        show_stats_panel: bool = True,
        warning_blink_interval: int = 10  # Frame interval cho nhấp nháy cảnh báo
    ):
        """
        Khởi tạo ViolationVisualizer
        
        Args:
            detector: ViolationDetector instance
            show_violation_box: Hiển thị khung vi phạm quanh xe
            show_violation_label: Hiển thị label vi phạm
            show_stats_panel: Hiển thị panel thống kê
            warning_blink_interval: Khoảng cách frame cho hiệu ứng nhấp nháy
        """
        self.detector = detector
        self.show_violation_box = show_violation_box
        self.show_violation_label = show_violation_label
        self.show_stats_panel = show_stats_panel
        self.warning_blink_interval = warning_blink_interval
        
        self._frame_count = 0
    
    def draw_violations(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        current_violations: Dict[int, List[ViolationType]],
        frame_number: int,
        copy_frame: bool = True,
    ) -> np.ndarray:
        """
        Vẽ thông tin vi phạm lên frame
        
        Args:
            frame: Frame gốc
            detections: Detections với tracker IDs
            current_violations: Dict từ ViolationDetector.update()
            frame_number: Số frame hiện tại
            
        Returns:
            Frame với thông tin vi phạm
        """
        self._frame_count = frame_number
        result = frame.copy() if copy_frame else frame
        
        if detections is None or len(detections) == 0:
            return result
        
        boxes = detections.xyxy
        tracker_ids = detections.tracker_id
        
        if tracker_ids is None:
            return result
        
        # Vẽ vi phạm cho từng xe
        for i, (box, tracker_id) in enumerate(zip(boxes, tracker_ids)):
            if tracker_id is None:
                continue
            
            violations = current_violations.get(tracker_id, [])
            
            if violations:
                result = self._draw_vehicle_violation(
                    result, box, tracker_id, violations
                )
        
        # Vẽ panel thống kê
        if self.show_stats_panel:
            result = self._draw_stats_panel(result, current_violations)
        
        return result
    
    def _draw_vehicle_violation(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        tracker_id: int,
        violations: List[ViolationType]
    ) -> np.ndarray:
        """Vẽ thông tin vi phạm cho một xe"""
        x1, y1, x2, y2 = map(int, box)
        
        # Lấy loại vi phạm chính và màu tương ứng
        primary_violation = violations[0] if violations else None
        color = self.VIOLATION_COLORS.get(primary_violation, (0, 0, 255))
        
        # Hiệu ứng nhấp nháy
        is_blink_on = (self._frame_count // self.warning_blink_interval) % 2 == 0
        
        if self.show_violation_box:
            # Vẽ khung cảnh báo
            thickness = 3 if is_blink_on else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Vẽ góc nhấn mạnh
            corner_length = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            self._draw_corner_brackets(frame, x1, y1, x2, y2, color, corner_length)
        
        if self.show_violation_label and primary_violation:
            # Vẽ label vi phạm
            label = self.VIOLATION_NAMES.get(primary_violation, "VI PHAM")
            label_with_id = f"#{tracker_id} {label}"
            
            # Background cho label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness_text = 2
            (text_w, text_h), baseline = cv2.getTextSize(label_with_id, font, font_scale, thickness_text)
            
            # Vị trí label (phía trên bounding box)
            label_x = x1
            label_y = y1 - 10
            
            # Đảm bảo label không ra ngoài frame
            if label_y - text_h - 5 < 0:
                label_y = y2 + text_h + 10
            
            # Vẽ background
            if is_blink_on:
                cv2.rectangle(frame, 
                             (label_x - 2, label_y - text_h - 5),
                             (label_x + text_w + 2, label_y + 5),
                             color, -1)
                cv2.putText(frame, label_with_id, (label_x, label_y),
                           font, font_scale, (255, 255, 255), thickness_text)
            else:
                cv2.rectangle(frame, 
                             (label_x - 2, label_y - text_h - 5),
                             (label_x + text_w + 2, label_y + 5),
                             (255, 255, 255), -1)
                cv2.putText(frame, label_with_id, (label_x, label_y),
                           font, font_scale, color, thickness_text)
        
        return frame
    
    def _draw_corner_brackets(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: Tuple[int, int, int],
        length: int
    ):
        """Vẽ góc nhấn mạnh cho bounding box"""
        thickness = 3
        
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
        
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
        
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
        
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)
    
    def _draw_stats_panel(
        self,
        frame: np.ndarray,
        current_violations: Dict[int, List[ViolationType]]
    ) -> np.ndarray:
        """Vẽ panel thống kê vi phạm"""
        h, w = frame.shape[:2]
        
        stats = self.detector.get_statistics()
        violating_count = stats['current_violating_count']
        total_violations = stats['total_violations']
        
        # Panel ở góc trên trái
        panel_width = 220
        panel_height = 80
        panel_x = 10
        panel_y = 50  # Dưới info text
        
        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (30, 30, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, 
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (100, 100, 100), 1)
        
        # Title
        cv2.putText(frame, "VIOLATION MONITOR", 
                   (panel_x + 10, panel_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Divider
        cv2.line(frame, (panel_x + 5, panel_y + 28), 
                (panel_x + panel_width - 5, panel_y + 28), (80, 80, 80), 1)
        
        # Current violating vehicles
        text_color = (0, 0, 255) if violating_count > 0 else (0, 255, 0)
        cv2.putText(frame, f"Dang vi pham: {violating_count}",
                   (panel_x + 10, panel_y + 48),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
        
        # Total wrong lane violations
        wrong_lane_total = total_violations.get(ViolationType.WRONG_LANE, 0)
        cv2.putText(frame, f"Tong sai lan: {wrong_lane_total}",
                   (panel_x + 10, panel_y + 68),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return frame
