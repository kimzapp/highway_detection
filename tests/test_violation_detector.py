"""Regression tests for wrong-lane detection behavior."""

import os
import sys

import numpy as np
import supervision as sv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from violations.detector import ViolationDetector, ViolationType


class IdentityBevTransformer:
    """Simple transformer for deterministic tests."""

    bev_width = 1920
    bev_height = 1080

    def transform_point(self, point):
        return int(point[0]), int(point[1])


def _make_detections(boxes, tracker_ids=None, class_ids=None):
    boxes = np.asarray(boxes, dtype=np.float32)
    n = len(boxes)

    if tracker_ids is None:
        tracker_ids = np.arange(1, n + 1, dtype=np.int64)
    if class_ids is None:
        class_ids = np.zeros(n, dtype=np.int64)

    return sv.Detections(
        xyxy=boxes,
        confidence=np.ones(n, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=np.int64),
        tracker_id=np.asarray(tracker_ids, dtype=np.int64),
    )


def test_wrong_lane_majority_outside_camera_points():
    detector = ViolationDetector(min_violation_frames=1, min_normal_frames=1)

    # Valid zone là phần bên trái; box cắt biên sao cho 2/3 điểm đáy nằm ngoài.
    zone = np.array([[0, 0], [40, 0], [40, 200], [0, 200]], dtype=np.int32)
    detector.set_valid_zones([zone])

    detections = _make_detections([[35, 50, 65, 100]])
    violations = detector.update(detections, {0: "car"}, frame_number=1)

    assert 1 in violations
    assert ViolationType.WRONG_LANE in violations[1]


def test_bev_decision_priority_over_camera_zone():
    detector = ViolationDetector(min_violation_frames=1, min_normal_frames=1)

    camera_zone = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.int32)
    detector.set_valid_zones([camera_zone])
    detector.set_bev_transformer(IdentityBevTransformer())

    # Mô phỏng BEV zone lệch so với camera zone để đảm bảo detector ưu tiên BEV.
    detector._bev_valid_zone_polygons = [
        np.array([[0, 0], [40, 0], [40, 200], [0, 200]], dtype=np.int32)
    ]

    detections = _make_detections([[120, 50, 160, 100]])
    violations = detector.update(detections, {0: "car"}, frame_number=1)

    assert 1 in violations
    assert ViolationType.WRONG_LANE in violations[1]


def test_no_violation_when_fully_inside_valid_zone():
    detector = ViolationDetector(min_violation_frames=1, min_normal_frames=1)

    zone = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.int32)
    detector.set_valid_zones([zone])
    detector.set_bev_transformer(IdentityBevTransformer())

    detections = _make_detections([[60, 50, 120, 100]])
    violations = detector.update(detections, {0: "car"}, frame_number=1)

    assert 1 in violations
    assert violations[1] == []
