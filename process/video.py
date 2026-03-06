"""
Video Processing Module
Xử lý video với YOLO detection và ByteTrack tracking
"""

import cv2
import numpy as np
import supervision as sv
from typing import Optional, Callable, Dict, Any
from ultralytics import YOLO

from tracking.bytetrack import ByteTracker
from lane_detection.road_zone import RoadZoneSelector, RoadZoneOverlay
from lane_detection.bird_eye_view import BirdEyeViewTransformer, BirdEyeViewVisualizer, create_combined_view


class VideoProcessor:
    """
    Video Processor với YOLO detection và ByteTrack tracking
    
    Attributes:
        model: YOLO model
        tracker: ByteTracker instance
        model_names: Dict mapping class_id to class name
    """
    
    def __init__(self, args):
        """
        Khởi tạo VideoProcessor
        
        Attributes:
            args: Parsed command-line arguments
            model: YOLO model được tải lên device
            tracker: ByteTracker sẽ được khởi tạo sau khi biết fps

        """
        self.model = YOLO(args.model)
        self.model.to(args.device)
        self.model_names = self.model.names
        
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
        
        # Tracker sẽ được khởi tạo khi biết fps
        self.tracker: Optional[ByteTracker] = None
        
        # Road zone overlay
        self.road_zone_overlay: Optional[RoadZoneOverlay] = None
        
        # Bird's Eye View
        self.bev_transformer: Optional[BirdEyeViewTransformer] = None
        self.bev_visualizer: Optional[BirdEyeViewVisualizer] = None
        self.enable_bev: bool = True  # Enable BEV by default when zone is selected
        self.bev_width: int = args.bev_width
        self.bev_height: int = args.bev_height
        
        # Callback functions
        self._on_frame_callback: Optional[Callable] = None
        self._on_detection_callback: Optional[Callable] = None
    
    def _init_tracker(self, fps: int = 30):
        """Khởi tạo tracker với fps thực tế"""
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
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, sv.Detections]:
        """
        Xử lý một frame: detect và track
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            annotated_frame: Frame đã được annotate
            tracked_detections: Detections với tracker IDs
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False, conf=self.conf_threshold, imgsz=self.img_size)
        
        # Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Filter by classes if specified
        if self.classes is not None:
            detections = detections[np.isin(detections.class_id, self.classes)]
        
        # Update tracker and annotate
        annotated_frame, tracked_detections = self.tracker.update_and_annotate(
            scene=frame,
            detections=detections,
            labels=None
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
        select_road_zone: bool = True
    ) -> Dict[str, Any]:
        """
        Xử lý video file
        
        Args:
            video_path: Đường dẫn video input
            output_path: Đường dẫn lưu video output (None = không lưu)
            display: Hiển thị video trong quá trình xử lý
            show_progress: In tiến trình xử lý
            select_road_zone: Tạm dừng ở frame đầu để chọn vùng đường hợp lệ
            
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
        
        # Select road zone on first frame if enabled
        first_frame = None
        if select_road_zone:
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Cannot read first frame from video")
            
            if show_progress:
                print("\n=== SELECT ROAD ZONE ===")
                print("Click to add points, Enter to confirm, Esc to skip")
            
            selector = RoadZoneSelector()
            zone_polygon = selector.select_zone(first_frame)
            
            if zone_polygon is not None:
                self.road_zone_overlay = RoadZoneOverlay(
                    zone_polygon=zone_polygon,
                    fill_color=(0, 255, 0),
                    border_color=(255, 255, 0),
                    alpha=0.2,
                    label="Valid Lane"
                )
                
                # Initialize Bird's Eye View transformer
                if self.enable_bev:
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
                    if show_progress:
                        print(f"Bird's Eye View enabled")
                
                if show_progress:
                    print(f"Road zone defined with {len(zone_polygon)} points")
            else:
                self.road_zone_overlay = None
                self.bev_transformer = None
                self.bev_visualizer = None
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
        writer = None
        if output_path:
            import platform
            ext = output_path.lower().split('.')[-1]
            
            # Chọn codec phù hợp theo OS và extension
            if platform.system() == 'Windows':
                if ext == 'avi':
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                else:
                    # Trên Windows, mp4v thường hoạt động tốt hơn
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    # Đổi extension sang avi để đảm bảo tương thích
                    if ext == 'mp4':
                        output_path = output_path[:-4] + '.avi'
                        if show_progress:
                            print(f"Note: Changed output format to .avi for better compatibility")
            else:
                # Linux/Mac
                if ext == 'mp4':
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                elif ext == 'avi':
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
            
            if show_progress:
                print(f"Output will be saved to: {output_path}")
                if self.bev_visualizer:
                    print(f"Output resolution: {output_width}x{output_height} (with BEV)")
        
        frame_count = 0
        stopped_by_user = False
        
        if show_progress:
            print("Processing... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, tracked_detections = self.process_frame(frame)
                
                # Call detection callback if set
                if self._on_detection_callback and len(tracked_detections) > 0:
                    self._on_detection_callback(tracked_detections, frame_count)
                
                # Call frame callback if set
                if self._on_frame_callback:
                    annotated_frame = self._on_frame_callback(
                        annotated_frame, tracked_detections, frame_count
                    )
                
                # Add frame info overlay
                info_text = f"Frame: {frame_count}/{total_frames} | Tracks: {len(tracked_detections)}"
                cv2.putText(annotated_frame, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Create Bird's Eye View if enabled
                display_frame = annotated_frame
                if self.bev_visualizer is not None:
                    bev_frame = self.bev_visualizer.draw(
                        detections=tracked_detections,
                        class_names=self.model_names,
                        show_ids=True,
                        show_labels=True
                    )
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
                
                # Save frame
                if writer:
                    writer.write(display_frame)
                
                frame_count += 1
                
                # Print progress
                if show_progress and frame_count % 100 == 0:
                    progress = 100 * frame_count / total_frames if total_frames > 0 else 0
                    print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        if show_progress:
            print(f"\nDone! Processed {frame_count} frames")
            if output_path:
                print(f"Output saved to: {output_path}")
        
        return {
            "frames_processed": frame_count,
            "total_frames": total_frames,
            "stopped_by_user": stopped_by_user,
            "output_path": output_path
        }
    
    def reset_tracker(self):
        """Reset tracker state (useful khi chuyển sang video mới)"""
        if self.tracker:
            self.tracker.tracker.reset()
