"""
Video Processing Module
Xử lý video với YOLO detection và ByteTrack tracking
"""

import time
import multiprocessing as mp
import queue
import cv2
import numpy as np
import supervision as sv
from typing import Optional, Callable, Dict, Any, List

from models import load_model, PTModelHandler
from tracking.bytetrack import ByteTracker
from lane_mapping.road_zone import RoadZoneSelector, RoadZoneOverlay, MultiRoadZoneOverlay
from lane_mapping.bird_eye_view import (
    BirdEyeViewTransformer, BirdEyeViewVisualizer,
    IPMBirdEyeViewTransformer, IPMBirdEyeViewVisualizer,
    create_combined_view
)
from violations import ViolationDetector, ViolationVisualizer, ViolationType
from .fps_counter import FPSCounter


def _video_writer_process_main(
    output_path: str,
    fourcc_code: str,
    fps: int,
    frame_size: tuple[int, int],
    frame_queue,
    status_queue,
):
    """Worker process: open writer and consume frames from queue."""
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc_code), fps, frame_size)
    if not writer.isOpened():
        status_queue.put({"state": "error", "message": f"Cannot open output writer: {output_path}"})
        return

    status_queue.put({"state": "ready"})

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            writer.write(frame)
    except Exception as exc:
        status_queue.put({"state": "error", "message": str(exc)})
    finally:
        writer.release()
        status_queue.put({"state": "done"})


class AsyncVideoWriter:
    """Video writer in a dedicated process to reduce impact on main processing FPS."""

    def __init__(
        self,
        output_path: str,
        fourcc_code: str,
        fps: int,
        frame_size: tuple[int, int],
        max_queue_size: int = 240,
    ):
        self._ctx = mp.get_context("spawn")
        self._frame_queue = self._ctx.Queue(maxsize=max_queue_size)
        self._status_queue = self._ctx.Queue(maxsize=16)
        self._proc = self._ctx.Process(
            target=_video_writer_process_main,
            args=(output_path, fourcc_code, fps, frame_size, self._frame_queue, self._status_queue),
            name="AsyncVideoWriterProcess",
            daemon=True,
        )
        self._closed = False
        self._started = False
        self._error: Optional[str] = None
        self._dropped_frames = 0

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def start(self, timeout_s: float = 10.0):
        if self._started:
            return

        self._proc.start()
        self._started = True

        try:
            status = self._status_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise RuntimeError("Timeout waiting for writer process startup") from exc

        if status.get("state") != "ready":
            self._error = status.get("message", "Writer process failed to start")
            raise RuntimeError(self._error)

    def write(self, frame: np.ndarray):
        self._poll_status_queue()

        if self._closed:
            raise RuntimeError("Cannot write to closed AsyncVideoWriter")
        if self._error is not None:
            raise RuntimeError(self._error)

        # Non-blocking enqueue to keep processing loop responsive.
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop oldest frame when queue is saturated, then enqueue newest frame.
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                self._dropped_frames += 1

    def close(self, timeout_s: float = 30.0):
        self._poll_status_queue()

        if self._closed:
            return
        self._closed = True

        if self._started:
            try:
                self._frame_queue.put(None, timeout=2.0)
            except queue.Full:
                # Force termination path if writer is not draining.
                self._proc.terminate()

            self._proc.join(timeout=timeout_s)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5.0)

        # Surface any writer-side error reported before shutdown.
        self._poll_status_queue()

        if self._error is not None:
            raise RuntimeError(self._error)

    def _poll_status_queue(self):
        while True:
            try:
                status = self._status_queue.get_nowait()
            except queue.Empty:
                break

            if status.get("state") == "error":
                self._error = status.get("message", "Unknown writer process error")


class VideoProcessor:
    """
    Video Processor với YOLO detection và ByteTrack tracking
    
    Attributes:
        model_handler: Model handler instance (PT, ONNX, etc.)
        tracker: ByteTracker instance
        model_names: Dict mapping class_id to class name
    """
    
    def __init__(self, args, model_handler=None):
        """
        Khởi tạo VideoProcessor
        
        Attributes:
            args: Parsed command-line arguments
            model_handler: Model handler được tải lên device (có thể truyền vào nếu đã load sẵn)
            tracker: ByteTracker sẽ được khởi tạo sau khi biết fps

        """
        # Sử dụng model handler đã load sẵn nếu có, nếu không thì load mới
        if model_handler is not None:
            self.model_handler = model_handler
            print("Using pre-loaded model handler")
        else:
            # Load model với auto-detection định dạng
            self.model_handler = load_model(args.model, args.device)
            print(f"Loaded model from: {args.model}")
        
        self.model_names = self.model_handler.names
        
        self.device = args.device
        self.conf_threshold = args.conf_thres
        self.iou_threshold = args.iou_thres
        self.max_age = args.max_age
        self.img_size = args.img_size
        self.classes = args.classes
        
        self.show_boxes = args.show_boxes
        self.show_labels = args.show_labels
        self.show_traces = args.show_traces
        self.trace_length = args.trace_length
        
        # Performance optimization options
        self.use_half = getattr(args, 'half', True)  # Use FP16 for GPU
        self.skip_frames = max(0, int(getattr(args, 'skip_frames', 2)))  # Process every N+1 frames
        self.skip_bev_frames = getattr(args, 'skip_bev_frames', 0)  # Skip BEV every N frames
        self.min_violation_frames = max(1, int(getattr(args, 'min_violation_frames', 45)))
        
        # Tracker sẽ được khởi tạo khi biết fps
        self.tracker: Optional[ByteTracker] = None
        
        # Road zone overlay
        self.road_zone_overlay: Optional[RoadZoneOverlay] = None
        
        # Bird's Eye View
        self.bev_transformer = None  # Can be BirdEyeViewTransformer or IPMBirdEyeViewTransformer
        self.bev_visualizer = None   # Can be BirdEyeViewVisualizer or IPMBirdEyeViewVisualizer
        self.enable_bev: bool = True  # Enable BEV by default when zone is selected
        self.bev_width: int = args.bev_width
        self.bev_height: int = args.bev_height
        self.bev_method: str = getattr(args, 'bev_method', 'ipm')  # 'ipm' or 'homography'
        self.camera_height: float = getattr(args, 'camera_height', 1.5)  # meters
        self._last_bev_frame: Optional[np.ndarray] = None  # Cache last BEV frame
        
        # Callback functions
        self._on_frame_callback: Optional[Callable] = None
        self._on_detection_callback: Optional[Callable] = None
        
        # Violation detection
        self.violation_detector: Optional[ViolationDetector] = None
        self.violation_visualizer: Optional[ViolationVisualizer] = None
        self._current_violations: Dict[int, List[ViolationType]] = {}
        
        # FPS counter for processing performance
        self.fps_counter: FPSCounter = FPSCounter(window_size=30)
    
    def _init_tracker(self, fps: int = 30):
        """Khởi tạo tracker với fps thực tế"""
        # Reset FPS counter for new video
        self.fps_counter.reset()
        
        self.tracker = ByteTracker(
            track_activation_threshold=self.conf_threshold,
            lost_track_buffer=self.max_age,
            minimum_matching_threshold=self.iou_threshold,
            frame_rate=fps,
            box_viz=self.show_boxes,
            label_viz=self.show_labels,
            trace_viz=self.show_traces,
            trace_length=self.trace_length
        )

    def _normalize_zone_polygon_for_frame(
        self,
        zone_polygon: np.ndarray,
        frame_shape: tuple,
    ) -> np.ndarray:
        """Chuẩn hóa polygon theo kích thước frame thực để BEV không bị méo."""
        points = np.asarray(zone_polygon, dtype=np.float32).reshape(-1, 2)
        h, w = frame_shape[:2]

        points[:, 0] = np.clip(np.round(points[:, 0]), 0, w - 1)
        points[:, 1] = np.clip(np.round(points[:, 1]), 0, h - 1)

        deduplicated: List[np.ndarray] = []
        for point in points.astype(np.int32):
            if not deduplicated or not np.array_equal(point, deduplicated[-1]):
                deduplicated.append(point)

        if len(deduplicated) > 1 and np.array_equal(deduplicated[0], deduplicated[-1]):
            deduplicated.pop()

        if len(deduplicated) < 3:
            raise ValueError("Zone polygon phải có ít nhất 3 điểm phân biệt")

        return np.array(deduplicated, dtype=np.int32)

    def _normalize_zone_polygons_for_frame(
        self,
        zone_polygons: List[np.ndarray],
        frame_shape: tuple,
    ) -> List[np.ndarray]:
        """Chuẩn hóa toàn bộ zones theo frame hiện tại."""
        normalized_polygons: List[np.ndarray] = []
        for polygon in zone_polygons:
            normalized = self._normalize_zone_polygon_for_frame(polygon, frame_shape)
            if len(normalized) >= 3:
                normalized_polygons.append(normalized)

        if not normalized_polygons:
            raise ValueError("Không có zone hợp lệ sau khi chuẩn hóa")

        return normalized_polygons
    
    def _init_bev_transformer(
        self, 
        first_frame: np.ndarray, 
        zone_polygon: np.ndarray,
        zone_polygons: list = None,
        show_progress: bool = True,
    ):
        """
        Khởi tạo BEV transformer 
        
        Args:
            first_frame: Frame đầu tiên để calibrate IPM
            zone_polygon: Polygon vùng đường (primary/combined polygon for BEV transform)
            zone_polygons: List tất cả các valid zone polygons để hiển thị
            show_progress: Hiển thị thông tin tiến trình
        """
        method_used = None
        h, w = first_frame.shape[:2]

        zone_polygon = self._normalize_zone_polygon_for_frame(zone_polygon, first_frame.shape)
        
        # Nếu không có zone_polygons, wrap zone_polygon
        if zone_polygons is None:
            zone_polygons = [zone_polygon]
        else:
            zone_polygons = self._normalize_zone_polygons_for_frame(zone_polygons, first_frame.shape)

        if show_progress:
            x_min = int(np.min(zone_polygon[:, 0]))
            x_max = int(np.max(zone_polygon[:, 0]))
            y_min = int(np.min(zone_polygon[:, 1]))
            y_max = int(np.max(zone_polygon[:, 1]))
            print(f"BEV input frame: {w}x{h}")
            print(f"BEV zone bounds: X=[{x_min}, {x_max}] Y=[{y_min}, {y_max}] points={len(zone_polygon)}")
        
        # Thử IPM trước (nếu được chọn)
        if self.bev_method == 'ipm':
            try:
                if show_progress:
                    print("Initializing IPM Bird's Eye View...")
                
                self.bev_transformer = IPMBirdEyeViewTransformer(
                    frame_width=w,
                    frame_height=h,
                    camera_height=self.camera_height,
                    bev_width=self.bev_width,
                    bev_height=self.bev_height,
                    roi_polygon=zone_polygon,
                    auto_calibrate=True
                )
                
                # Calibrate từ frame
                calibrated = self.bev_transformer.calibrate_from_frame(first_frame)
                
                if calibrated:
                    info = self.bev_transformer.get_calibration_info()
                    if show_progress:
                        print(f"  Camera pitch: {info['pitch_angle_deg']:.1f}°")
                        print(f"  Focal length: {info['focal_length']:.0f} px")
                        if info['vanishing_point']:
                            print(f"  Vanishing point: {info['vanishing_point']}")
                
                # Test transform một điểm để đảm bảo hoạt động
                test_point = (w // 2, h * 3 // 4)
                test_result = self.bev_transformer.transform_point(test_point)
                
                if test_result == (-1, -1):
                    raise ValueError("IPM transform returned invalid point")
                
                # IPM thành công, tạo visualizer với tất cả zones
                self.bev_visualizer = IPMBirdEyeViewVisualizer(
                    transformer=self.bev_transformer,
                    bg_color=(30, 30, 35),
                    show_grid=True,
                    show_distance_markers=True,
                    valid_zone_polygons=zone_polygons,  # Truyền tất cả polygons để hiển thị từng zone
                    show_zones=True,
                    invalid_zone_color=(0, 0, 120),    # Đỏ đậm
                    zone_alpha=0.4
                )
                method_used = 'IPM'
                
            except Exception as e:
                if show_progress:
                    print(f"  IPM initialization failed: {e}")
                    print("  Falling back to Homography method...")
                self.bev_transformer = None
                self.bev_visualizer = None
        
        # Fallback về Homography nếu IPM thất bại hoặc không được chọn
        if self.bev_transformer is None:
            try:
                if show_progress and self.bev_method == 'homography':
                    print("Initializing Homography Bird's Eye View...")
                
                self.bev_transformer = BirdEyeViewTransformer(
                    source_polygon=zone_polygon,
                    bev_width=self.bev_width,
                    bev_height=self.bev_height,
                    margin=50
                )
                self.bev_visualizer = BirdEyeViewVisualizer(
                    transformer=self.bev_transformer,
                    bg_color=(40, 40, 40),
                    lane_color=(80, 80, 80),
                    lane_border_color=(255, 255, 0)
                )
                method_used = 'Homography'

                if show_progress:
                    matrix = self.bev_transformer.transform_matrix
                    det = float(np.linalg.det(matrix))
                    print(f"  Homography determinant: {det:.6f}")
                
            except Exception as e:
                if show_progress:
                    print(f"  Homography initialization also failed: {e}")
                    print("  Bird's Eye View will be disabled")
                self.bev_transformer = None
                self.bev_visualizer = None
                return
        
        if show_progress and method_used:
            print(f"Bird's Eye View enabled ({method_used})")
    
    def set_on_frame_callback(self, callback: Callable[[np.ndarray, sv.Detections, int], np.ndarray]):
        """
        Đặt callback được gọi sau mỗi frame
        
        Args:
            callback: Function(frame, detections, frame_count) -> processed_frame
        """
        self._on_frame_callback = callback
    
    def set_on_detection_callback(self, callback: Callable[[sv.Detections, int], None]):
        """
        Đặt callback được gọi khi có detections
        
        Args:
            callback: Function(detections, frame_count) -> None
        """
        self._on_detection_callback = callback
    
    def process_frame(
        self,
        frame: np.ndarray,
        return_timing: bool = False
    ) -> tuple[np.ndarray, sv.Detections] | tuple[np.ndarray, sv.Detections, Dict[str, float]]:
        """
        Xử lý một frame: detect và track
        
        Args:
            frame: Input frame (BGR)
            return_timing: Nếu True, trả thêm dict thời gian inference/tracking (giây)
            
        Returns:
            annotated_frame: Frame đã được annotate
            tracked_detections: Detections với tracker IDs
        """
        inference_start = time.perf_counter()
        detections = self.infer_detections(frame)
        inference_elapsed = time.perf_counter() - inference_start

        tracking_start = time.perf_counter()
        annotated_frame, tracked_detections = self.track_with_detections(frame, detections)
        tracking_elapsed = time.perf_counter() - tracking_start

        if return_timing:
            return annotated_frame, tracked_detections, {
                "inference": inference_elapsed,
                "tracking": tracking_elapsed,
            }

        return annotated_frame, tracked_detections

    def infer_detections(self, frame: np.ndarray) -> sv.Detections:
        """Run model inference and convert output to supervision detections."""
        # Run detection với model handler (sử dụng half precision nếu có)
        results = self.model_handler.predict(
            frame, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.img_size,
            half=self.use_half
        )
        
        # Convert to supervision Detections
        if isinstance(self.model_handler, PTModelHandler):
            # Ultralytics model - use built-in converter
            detections = sv.Detections.from_ultralytics(results[0])
            # Filter by classes if specified
            if self.classes is not None:
                detections = detections[np.isin(detections.class_id, self.classes)]
        else:
            # ONNX hoặc model khác - tạo Detections từ boxes, scores, class_ids
            boxes, scores, class_ids = self.model_handler.get_detections(results)
            
            if len(boxes) > 0:
                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=scores,
                    class_id=class_ids
                )
            else:
                detections = sv.Detections.empty()

        return detections

    def track_with_detections(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> tuple[np.ndarray, sv.Detections]:
        """Update tracker using provided detections and draw overlays."""

        # Update tracker and annotate (in-place để tránh copy)
        annotated_frame, tracked_detections = self.tracker.update_and_annotate(
            scene=frame,
            detections=detections,
            labels=None,
            copy_scene=False  # In-place để giảm memory copy mỗi frame
        )
        
        # Draw road zone overlay if defined
        if self.road_zone_overlay is not None:
            annotated_frame = self.road_zone_overlay.draw(annotated_frame)
        
        # Add labels with tracker IDs
        if self.show_labels and tracked_detections.tracker_id is not None and len(tracked_detections) > 0:
            labels = [
                f"#{tracker_id} {self.model_names[class_id]} {conf:.2f}"
                for tracker_id, class_id, conf in zip(
                    tracked_detections.tracker_id,
                    tracked_detections.class_id,
                    tracked_detections.confidence
                )
            ]
            if self.tracker.label_annotator:
                annotated_frame = self.tracker.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_detections,
                    labels=labels
                )

        return annotated_frame, tracked_detections
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        show_progress: bool = True,
        select_road_zone: bool = True,
        preset_zones: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Xử lý video file
        
        Args:
            video_path: Đường dẫn video input
            output_path: Đường dẫn lưu video output (None = không lưu)
            display: Hiển thị video trong quá trình xử lý
            show_progress: In tiến trình xử lý
            select_road_zone: Tạm dừng ở frame đầu để chọn vùng đường hợp lệ
            preset_zones: Danh sách zones được cung cấp sẵn (bỏ qua select_road_zone nếu có)
            
        Returns:
            Dict với thông tin xử lý (frames_processed, etc.)
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize tracker với fps thực tế
        self._init_tracker(fps)
        
        if show_progress:
            print(f"Video: {video_path}")
            print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Handle zones - either from preset or interactive selection
        first_frame = None
        zone_polygons = None
        
        # Use preset zones if provided
        if preset_zones is not None and len(preset_zones) > 0:
            zone_polygons = preset_zones
            # Read first frame for BEV initialization
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Cannot read first frame from video")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            zone_polygons = self._normalize_zone_polygons_for_frame(zone_polygons, first_frame.shape)
            
            if show_progress:
                print(f"Using {len(zone_polygons)} preset zone(s)")
            
            # Initialize road zone overlay, violation detector, and BEV for preset zones
            # Use MultiRoadZoneOverlay for multiple zones
            self.road_zone_overlay = MultiRoadZoneOverlay(
                zone_polygons=zone_polygons,
                alpha=0.2,
            )
            
            # Initialize Violation Detector with valid zones
            self.violation_detector = ViolationDetector(
                min_violation_frames=self.min_violation_frames,
                min_normal_frames=3,
                enabled_violations={ViolationType.WRONG_LANE}
            )
            # Convert zone_polygons to numpy arrays for detector
            np_zone_polygons = [np.array(z) for z in zone_polygons]
            self.violation_detector.set_valid_zones(np_zone_polygons)
            
            # Initialize Violation Visualizer
            self.violation_visualizer = ViolationVisualizer(
                detector=self.violation_detector,
                show_violation_box=True,
                show_violation_label=True,
                show_stats_panel=True
            )
            
            if show_progress:
                print("Violation Detector initialized (WRONG_LANE detection enabled)")
            
            # Initialize Bird's Eye View transformer
            if self.enable_bev:
                # Use combined polygon for BEV transform calibration
                bev_polygon = self.road_zone_overlay.get_combined_polygon()
                if bev_polygon is None:
                    bev_polygon = zone_polygons[0]
                
                self._init_bev_transformer(
                    first_frame=first_frame,
                    zone_polygon=bev_polygon,
                    zone_polygons=zone_polygons,  # Pass all zones for visualization
                    show_progress=show_progress
                )
                
                # Set BEV transformer to violation detector
                if self.bev_transformer is not None:
                    self.violation_detector.set_bev_transformer(self.bev_transformer)
            
            if show_progress:
                total_points = sum(len(z) for z in zone_polygons)
                print(f"Road zones defined: {len(zone_polygons)} zone(s), {total_points} total points")
                
        elif select_road_zone:
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Cannot read first frame from video")
            
            if show_progress:
                print("\n=== SELECT ROAD ZONES ===")
                print("Click to add points, N for new zone, Enter to confirm, Esc to skip")
            
            selector = RoadZoneSelector()
            zone_polygons = selector.select_zones(first_frame)
            
            if zone_polygons is not None and len(zone_polygons) > 0:
                zone_polygons = self._normalize_zone_polygons_for_frame(zone_polygons, first_frame.shape)
                # Use MultiRoadZoneOverlay for multiple zones
                self.road_zone_overlay = MultiRoadZoneOverlay(
                    zone_polygons=zone_polygons,
                    alpha=0.2,
                )
                
                # Initialize Violation Detector with valid zones
                self.violation_detector = ViolationDetector(
                    min_violation_frames=self.min_violation_frames,
                    min_normal_frames=3,
                    enabled_violations={ViolationType.WRONG_LANE}
                )
                # Convert zone_polygons to numpy arrays for detector
                np_zone_polygons = [np.array(z) for z in zone_polygons]
                self.violation_detector.set_valid_zones(np_zone_polygons)
                
                # Initialize Violation Visualizer
                self.violation_visualizer = ViolationVisualizer(
                    detector=self.violation_detector,
                    show_violation_box=True,
                    show_violation_label=True,
                    show_stats_panel=True
                )
                
                if show_progress:
                    print("Violation Detector initialized (WRONG_LANE detection enabled)")
                
                # Initialize Bird's Eye View transformer with primary zone
                # (uses first zone or combined polygon)
                if self.enable_bev:
                    # Use combined polygon for BEV transform calibration
                    bev_polygon = self.road_zone_overlay.get_combined_polygon()
                    if bev_polygon is None:
                        bev_polygon = zone_polygons[0]
                    
                    self._init_bev_transformer(
                        first_frame=first_frame,
                        zone_polygon=bev_polygon,
                        zone_polygons=zone_polygons,  # Pass all zones for visualization
                        show_progress=show_progress
                    )
                    
                    # Set BEV transformer to violation detector
                    if self.bev_transformer is not None:
                        self.violation_detector.set_bev_transformer(self.bev_transformer)
                
                if show_progress:
                    total_points = sum(len(z) for z in zone_polygons)
                    print(f"Road zones defined: {len(zone_polygons)} zone(s), {total_points} total points")
            else:
                self.road_zone_overlay = None
                self.bev_transformer = None
                self.bev_visualizer = None
                self.violation_detector = None
                self.violation_visualizer = None
                if show_progress:
                    print("No road zone defined, skipping...")
            
            # Reset video to beginning to process all frames including first
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Calculate output dimensions based on BEV
        output_width = width
        output_height = height
        if self.bev_visualizer is not None:
            # BEV will be added on the right side, scaled to match height
            bev_scale = height / self.bev_transformer.bev_height
            bev_display_width = int(self.bev_transformer.bev_width * bev_scale)
            output_width = width + bev_display_width
        
        # Video writer
        async_writer = None
        if output_path:
            import platform
            ext = output_path.lower().split('.')[-1]
            fourcc_code = 'XVID'
            
            # Chọn codec phù hợp theo OS và extension
            if platform.system() == 'Windows':
                if ext == 'avi':
                    fourcc_code = 'XVID'
                else:
                    # Trên Windows, mp4v thường hoạt động tốt hơn
                    fourcc_code = 'XVID'
                    # Đổi extension sang avi để đảm bảo tương thích
                    if ext == 'mp4':
                        output_path = output_path[:-4] + '.avi'
                        if show_progress:
                            print(f"Note: Changed output format to .avi for better compatibility")
            else:
                # Linux/Mac
                if ext == 'mp4':
                    fourcc_code = 'mp4v'
                elif ext == 'avi':
                    fourcc_code = 'XVID'
                else:
                    fourcc_code = 'mp4v'

            # Use separate process for disk/video encoding writes.
            async_writer = AsyncVideoWriter(
                output_path=output_path,
                fourcc_code=fourcc_code,
                fps=fps,
                frame_size=(output_width, output_height),
            )
            async_writer.start()
            
            if show_progress:
                print(f"Output will be saved to: {output_path}")
                if self.bev_visualizer:
                    print(f"Output resolution: {output_width}x{output_height} (with BEV)")
        
        frame_count = 0
        inferred_frame_count = 0
        stopped_by_user = False
        cached_detections = sv.Detections.empty()
        
        if show_progress:
            print("Processing... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                should_process_frame = (
                    self.skip_frames <= 0
                    or frame_count % (self.skip_frames + 1) == 0
                    or frame_count == 0
                )

                if should_process_frame:
                    inferred_frame_count += 1
                    cached_detections = self.infer_detections(frame)

                # Always update tracker. On skipped frames it receives cached detections.
                annotated_frame, tracked_detections = self.track_with_detections(
                    frame,
                    cached_detections,
                )

                if self.violation_detector is not None and len(tracked_detections) > 0:
                    self._current_violations = self.violation_detector.update(
                        detections=tracked_detections,
                        class_names=self.model_names,
                        frame_number=frame_count
                    )
                else:
                    self._current_violations = {}

                # Draw violations on both processed and skipped frames with latest state.
                if self.violation_visualizer is not None and len(tracked_detections) > 0:
                    annotated_frame = self.violation_visualizer.draw_violations(
                        frame=annotated_frame,
                        detections=tracked_detections,
                        current_violations=self._current_violations,
                        frame_number=frame_count
                    )
                
                # Call detection callback if set
                if self._on_detection_callback and len(tracked_detections) > 0:
                    self._on_detection_callback(tracked_detections, frame_count)
                
                # Call frame callback if set
                if self._on_frame_callback:
                    annotated_frame = self._on_frame_callback(
                        annotated_frame, tracked_detections, frame_count
                    )
                
                # Add frame info overlay with FPS
                proc_fps = self.fps_counter.avg_fps
                info_text = f"Frame: {frame_count}/{total_frames} | Tracks: {len(tracked_detections)} | FPS: {proc_fps:.1f}"
                cv2.putText(annotated_frame, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Create Bird's Eye View if enabled (với skip frames để tối ưu)
                display_frame = annotated_frame
                if self.bev_visualizer is not None:
                    # Skip BEV calculation để tăng FPS (dùng frame cache)
                    should_update_bev = (self.skip_bev_frames <= 0 or 
                                        frame_count % (self.skip_bev_frames + 1) == 0 or
                                        self._last_bev_frame is None)
                    
                    if should_update_bev:
                        bev_frame = self.bev_visualizer.draw(
                            detections=tracked_detections,
                            class_names=self.model_names,
                            show_ids=True,
                            show_labels=True,
                            current_violations=self._current_violations,
                        )
                        self._last_bev_frame = bev_frame
                    else:
                        bev_frame = self._last_bev_frame
                    
                    display_frame = create_combined_view(
                        camera_frame=annotated_frame,
                        bev_frame=bev_frame,
                        layout="horizontal"
                    )
                
                # Display if enabled
                if display:
                    cv2.imshow("ByteTrack - Object Tracking (with Bird's Eye View)", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stopped_by_user = True
                        if show_progress:
                            print("Stopped by user")
                        break
                
                # Save frame asynchronously to avoid blocking processing loop.
                if async_writer:
                    async_writer.write(display_frame)
                
                frame_count += 1
                
                # Update FPS counter
                self.fps_counter.tick()
                
                # Print progress
                if show_progress and frame_count % 100 == 0:
                    progress = 100 * frame_count / total_frames if total_frames > 0 else 0
                    print(
                        f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%), "
                        f"inferred={inferred_frame_count}"
                    )
        
        finally:
            # Cleanup
            cap.release()
            if async_writer:
                async_writer.close()
            cv2.destroyAllWindows()

        if show_progress and async_writer and async_writer.dropped_frames > 0:
            print(f"Dropped frames while writing: {async_writer.dropped_frames}")
        
        if show_progress:
            print(f"\nDone! Processed {frame_count} frames")
            print(f"Inference ran on {inferred_frame_count} frames (skip={self.skip_frames})")
            if output_path:
                print(f"Output saved to: {output_path}")
        
        return {
            "frames_processed": frame_count,
            "frames_inferred": inferred_frame_count,
            "total_frames": total_frames,
            "stopped_by_user": stopped_by_user,
            "output_path": output_path,
            "fps_stats": self.fps_counter.get_stats()
        }
    
    def reset_tracker(self):
        """Reset tracker state (useful khi chuyển sang video mới)"""
        if self.tracker:
            self.tracker.tracker.reset()
    
    def get_fps_stats(self) -> dict:
        """
        Lấy thống kê FPS xử lý.
        
        Returns:
            Dict với các thông số:
            - fps: FPS tức thời
            - avg_fps: FPS trung bình (sliding window 30 frames)
            - overall_fps: FPS tổng thể từ đầu
            - frame_count: Số frame đã xử lý
            - elapsed_time: Thời gian xử lý (giây)
        """
        return self.fps_counter.get_stats()
    
    @property
    def current_fps(self) -> float:
        """FPS xử lý hiện tại (trung bình)"""
        return self.fps_counter.avg_fps
