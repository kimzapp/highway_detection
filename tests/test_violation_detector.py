"""Tests for violation detector behaviors, including invalid-vehicle detection."""

import os
import sys

import numpy as np
import supervision as sv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from violations.detector import ViolationDetector, ViolationType


def _single_detection(class_id: int, tracker_id: int, box=None) -> sv.Detections:
    if box is None:
        box = [10.0, 10.0, 30.0, 30.0]

    return sv.Detections(
        xyxy=np.array([box], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([class_id], dtype=np.int32),
        tracker_id=np.array([tracker_id], dtype=np.int32),
    )


def test_invalid_vehicle_confirmed_after_min_frames():
    detector = ViolationDetector(
        min_violation_frames=3,
        min_normal_frames=2,
        enabled_violations={ViolationType.INVALID_VEHICLE},
        valid_vehicle_class_ids={2},
    )

    class_names = {2: "Car", 3: "Motorcycle"}

    for frame_id in range(1, 4):
        current = detector.update(_single_detection(class_id=3, tracker_id=101), class_names, frame_id)

    assert current[101] == [ViolationType.INVALID_VEHICLE]
    assert len(detector.get_violations_log()) == 1


def test_valid_vehicle_not_flagged_invalid_vehicle():
    detector = ViolationDetector(
        min_violation_frames=2,
        min_normal_frames=2,
        enabled_violations={ViolationType.INVALID_VEHICLE},
        valid_vehicle_class_ids={2},
    )

    class_names = {2: "Car"}

    current = detector.update(_single_detection(class_id=2, tracker_id=9), class_names, frame_number=1)

    assert current[9] == []
    assert detector.get_violations_log() == []


def test_invalid_vehicle_state_clears_after_normal_frames():
    detector = ViolationDetector(
        min_violation_frames=2,
        min_normal_frames=2,
        enabled_violations={ViolationType.INVALID_VEHICLE},
        valid_vehicle_class_ids={2},
    )
    class_names = {2: "Car", 3: "Motorcycle"}

    # Confirm invalid-vehicle first.
    detector.update(_single_detection(class_id=3, tracker_id=77), class_names, frame_number=1)
    detector.update(_single_detection(class_id=3, tracker_id=77), class_names, frame_number=2)

    # Vehicle becomes valid for enough frames -> violation state should clear.
    current = detector.update(_single_detection(class_id=2, tracker_id=77), class_names, frame_number=3)
    assert current[77] == [ViolationType.INVALID_VEHICLE]

    current = detector.update(_single_detection(class_id=2, tracker_id=77), class_names, frame_number=4)
    assert current[77] == []


def test_wrong_lane_still_detects_for_valid_vehicle():
    detector = ViolationDetector(
        min_violation_frames=2,
        min_normal_frames=1,
        enabled_violations={ViolationType.WRONG_LANE, ViolationType.INVALID_VEHICLE},
        valid_vehicle_class_ids={2},
    )

    # Valid zone is far away from test box's center-bottom point (20, 30).
    valid_zone = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.int32)
    detector.set_valid_zones([valid_zone])

    class_names = {2: "Car"}

    detector.update(_single_detection(class_id=2, tracker_id=5), class_names, frame_number=1)
    current = detector.update(_single_detection(class_id=2, tracker_id=5), class_names, frame_number=2)

    assert current[5] == [ViolationType.WRONG_LANE]
