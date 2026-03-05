"""
Object Tracking Pipeline với Supervision + YOLO
Sử dụng thư viện Supervision của Roboflow để tracking
"""

import cv2
import numpy as np
from typing import Optional

import supervision as sv
from ultralytics import YOLO


class ByteTracker:
    def __init__(self, track_activation_threshold=0.3,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
            box_viz=True,
            label_viz=True,
            trace_viz=True,
            trace_length=50
            ):
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.frame_rate = frame_rate
        self.box_viz = box_viz
        self.label_viz = label_viz
        self.trace_viz = trace_viz
        self.trace_length = trace_length

        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            frame_rate=self.frame_rate
        )

        if box_viz:
            box_annotator = sv.BoxAnnotator(
                thickness=2,
                color=sv.ColorPalette.from_hex(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'])
            )
        
        if label_viz:
            label_annotator = sv.LabelAnnotator(
                text_scale=0.5,
                text_thickness=1,
                text_padding=5
            )
        
        if trace_viz:
            trace_annotator = sv.TraceAnnotator(
                thickness=2,
                trace_length=trace_length,
                color=sv.ColorPalette.from_hex(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'])
            )

    def update(self, detections):
        """
        Cập nhật tracker với detections mới
        
        Args:
            detections: sv.Detections object chứa bounding boxes, class_id, confidence
        Returns:
            updated_detections: sv.Detections object có thêm tracker_id
        """
        updated_detections = self.tracker.update_with_detections(detections)
        return updated_detections
    
    def update_and_annotate(self, scene, detections, labels=None):
        """
        Cập nhật tracker và vẽ annotations lên scene
        
        Args:
            scene: Hình ảnh (frame) để vẽ
            detections: sv.Detections object chứa bounding boxes, class_id, confidence
            labels: List of labels tương ứng với detections (optional)
        Returns:
            annotated_scene: Hình ảnh đã được vẽ annotations
            updated_detections: sv.Detections object có thêm tracker_id
        """
        annotated_scene = scene.copy()
        updated_detections = self.update(detections)
        
        if self.trace_viz:
            annotated_scene = self.trace_annotator.annotate(
                scene=annotated_scene,
                detections=updated_detections
            )
        
        if self.box_viz:
            annotated_scene = self.box_annotator.annotate(
                scene=annotated_scene,
                detections=updated_detections
            )
        
        if self.label_viz and labels is not None:
            annotated_scene = self.label_annotator.annotate(
                scene=annotated_scene,
                detections=updated_detections,
                labels=labels
            )
        
        return annotated_scene, updated_detections


# ví dụ để chạy test
def run_tracking(
    video_path: str,
    model_path: str = "yolov8n.pt",
    output_path: Optional[str] = None,
    tracker_type: str = "bytetrack",
    show_labels: bool = True,
    show_trace: bool = True,
    trace_length: int = 50,
    confidence_threshold: float = 0.3,
    classes: Optional[list] = None
):
    """
    Chạy object tracking với YOLO + Supervision
    
    Args:
        video_path: Đường dẫn video input (hoặc 0 cho webcam)
        model_path: Đường dẫn YOLO model
        output_path: Đường dẫn lưu video output (None = không lưu)
        tracker_type: Loại tracker ("bytetrack" hoặc "botsort")
        show_labels: Hiển thị labels
        show_trace: Hiển thị đường đi của objects
        trace_length: Độ dài trace (số frames)
        confidence_threshold: Ngưỡng confidence
        classes: List class IDs cần track (None = tất cả)
    """
    # Load YOLO model
    model = YOLO(model_path)
    
    # Khởi tạo tracker từ Supervision
    if tracker_type == "bytetrack":
        tracker = sv.ByteTrack(
            track_activation_threshold=confidence_threshold,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
    else:  # botsort
        tracker = sv.ByteTrack()  # Supervision chủ yếu dùng ByteTrack
    
    # Annotators để vẽ
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color=sv.ColorPalette.from_hex(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'])
    )
    
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=5
    )
    
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=trace_length,
        color=sv.ColorPalette.from_hex(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'])
    )
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return
    
    # Lấy thông tin video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    # Video writer nếu cần lưu output
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Nhấn 'q' để thoát...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO detection
        results = model(frame, verbose=False, conf=confidence_threshold)
        
        # Chuyển sang Supervision Detections
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Lọc theo classes nếu cần
        if classes is not None:
            detections = detections[np.isin(detections.class_id, classes)]
        
        # Update tracker
        detections = tracker.update_with_detections(detections)
        
        # Vẽ annotations
        annotated_frame = frame.copy()
        
        # Vẽ trace (đường đi)
        if show_trace:
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
        
        # Vẽ bounding boxes
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        # Vẽ labels
        if show_labels and detections.tracker_id is not None:
            labels = [
                f"#{tracker_id} {model.names[class_id]} {conf:.2f}"
                for tracker_id, class_id, conf in zip(
                    detections.tracker_id,
                    detections.class_id,
                    detections.confidence
                )
            ]
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )
        
        # Hiển thị info
        info = f"Frame: {frame_count} | Tracks: {len(detections)}"
        cv2.putText(annotated_frame, info, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị
        cv2.imshow("Tracking", annotated_frame)
        
        # Lưu video
        if writer:
            writer.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Đã xử lý {frame_count} frames")
    if output_path:
        print(f"Video đã lưu tại: {output_path}")


def run_tracking_with_callback(
    video_path: str,
    model_path: str = "yolov8n.pt",
    callback=None,
    **kwargs
):
    """
    Chạy tracking với callback function để xử lý mỗi frame
    
    Args:
        video_path: Đường dẫn video
        model_path: Đường dẫn YOLO model
        callback: Function(frame, detections) -> frame
        **kwargs: Các tham số khác cho tracker
    """
    model = YOLO(model_path)
    tracker = sv.ByteTrack()
    
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)
        
        # Gọi callback nếu có
        if callback:
            frame = callback(frame, detections)
        
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Object Tracking với YOLO + Supervision")
    parser.add_argument("--video", type=str, default=None, help="Video path (hoặc 0 cho webcam)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--no-trace", action="store_true", help="Không hiển thị trace")
    parser.add_argument("--no-labels", action="store_true", help="Không hiển thị labels")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--classes", type=int, nargs="+", default=None, 
                        help="Class IDs để track (vd: --classes 0 2 cho person và car)")
    
    args = parser.parse_args()
    
    video_source = args.video if args.video else 0
    
    run_tracking(
        video_path=video_source,
        model_path=args.model,
        output_path=args.output,
        show_trace=not args.no_trace,
        show_labels=not args.no_labels,
        confidence_threshold=args.conf,
        classes=args.classes
    )
