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
    """
    ByteTrack tracker wrapper sử dụng Supervision
    Tham số:
        track_activation_threshold: Ngưỡng confidence để kích hoạt track mới
        lost_track_buffer: Số frames tối đa để giữ track khi không có detections
        minimum_matching_threshold: Ngưỡng IoU để matching detections với tracks
        frame_rate: Frame rate của video (để tính toán thời gian mất track)
        box_viz: Hiển thị bounding boxes   
        label_viz: Hiển thị labels
        trace_viz: Hiển thị đường đi của objects
        trace_length: Độ dài trace (số frames)
    """
    def __init__(self, track_activation_threshold=0.3,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
            box_viz=True,
            label_viz=True,
            trace_viz=True,
            trace_length=25
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

        self.box_annotator = None
        self.label_annotator = None
        self.trace_annotator = None

        if box_viz:
            self.box_annotator = sv.BoxAnnotator(
                thickness=2,
                color=sv.ColorPalette.from_hex(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'])
            )
        
        if label_viz:
            self.label_annotator = sv.LabelAnnotator(
                text_scale=0.5,
                text_thickness=1,
                text_padding=2
            )
        
        if trace_viz:
            self.trace_annotator = sv.TraceAnnotator(
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
    
    def update_and_annotate(self, scene, detections, labels=None, copy_scene=True):
        """
        Cập nhật tracker và vẽ annotations lên scene
        
        Args:
            scene: Hình ảnh (frame) để vẽ
            detections: sv.Detections object chứa bounding boxes, class_id, confidence
            labels: List of labels tương ứng với detections (optional)
            copy_scene: Copy scene trước khi annotate (False = in-place, nhanh hơn)
        Returns:
            annotated_scene: Hình ảnh đã được vẽ annotations
            updated_detections: sv.Detections object có thêm tracker_id
        """
        # Only copy if needed (saving memory allocation time)
        annotated_scene = scene.copy() if copy_scene else scene
        updated_detections = self.update(detections)
        
        # Skip annotation entirely if nothing to draw
        if not (self.trace_viz or self.box_viz):
            return annotated_scene, updated_detections
        
        if self.trace_viz and self.trace_annotator:
            annotated_scene = self.trace_annotator.annotate(
                scene=annotated_scene,
                detections=updated_detections
            )
        
        if self.box_viz and self.box_annotator:
            annotated_scene = self.box_annotator.annotate(
                scene=annotated_scene,
                detections=updated_detections
            )
        
        if self.label_viz and self.label_annotator and labels is not None:
            annotated_scene = self.label_annotator.annotate(
                scene=annotated_scene,
                detections=updated_detections,
                labels=labels
            )
        
        return annotated_scene, updated_detections
