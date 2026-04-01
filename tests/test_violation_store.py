"""Tests for SQLite violation storage."""

from datetime import datetime

from storage import ViolationStore
from violations import Violation, ViolationType


def _sample_metadata(video_path: str, fps: float = 30.0):
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return {
        "video_path": video_path,
        "fps": fps,
        "total_frames": 300,
        "width": 1280,
        "height": 720,
        "processing_config": {"enable_bev": True},
        "output_video_path": "outputs/out.mp4",
        "started_at": now,
        "finished_at": now,
    }


def _sample_violation(frame_number: int, tracker_id: int = 1):
    return Violation(
        violation_type=ViolationType.WRONG_LANE,
        tracker_id=tracker_id,
        class_id=2,
        class_name="car",
        position=(100, 200),
        bev_position=(10, 20),
        frame_number=frame_number,
        confidence=0.91,
    )


def test_save_and_read_video_result(tmp_path):
    db_path = str(tmp_path / "violations.db")
    store = ViolationStore(db_path=db_path)

    video_path = str(tmp_path / "video_a.mp4")
    save_result = store.save_video_result(
        video_metadata=_sample_metadata(video_path, fps=25.0),
        violations=[_sample_violation(frame_number=50)],
    )

    assert save_result["violations_saved"] == 1
    payload = store.get_video_result(video_path)
    assert payload is not None
    assert payload["video"]["video_path"] == video_path
    assert len(payload["violations"]) == 1
    assert payload["violations"][0]["violation_type"] == "WRONG_LANE"


def test_overwrite_same_video_key(tmp_path):
    db_path = str(tmp_path / "violations.db")
    store = ViolationStore(db_path=db_path)

    video_path = str(tmp_path / "video_overwrite.mp4")
    store.save_video_result(
        video_metadata=_sample_metadata(video_path, fps=20.0),
        violations=[_sample_violation(frame_number=10), _sample_violation(frame_number=20)],
    )

    store.save_video_result(
        video_metadata=_sample_metadata(video_path, fps=20.0),
        violations=[_sample_violation(frame_number=99, tracker_id=7)],
    )

    payload = store.get_video_result(video_path)
    assert payload is not None
    assert len(payload["violations"]) == 1
    assert payload["violations"][0]["frame_number"] == 99
    assert payload["violations"][0]["tracker_id"] == 7


def test_violation_id_deterministic(tmp_path):
    store = ViolationStore(db_path=str(tmp_path / "violations.db"))
    video_key = store.make_video_key(str(tmp_path / "video_deterministic.mp4"))

    vid_1 = store.make_violation_id(video_key, tracker_id=42, violation_type="WRONG_LANE", frame_number=123)
    vid_2 = store.make_violation_id(video_key, tracker_id=42, violation_type="WRONG_LANE", frame_number=123)
    vid_3 = store.make_violation_id(video_key, tracker_id=42, violation_type="WRONG_LANE", frame_number=124)

    assert vid_1 == vid_2
    assert vid_1 != vid_3


def test_violation_time_sec_from_frame_and_fps(tmp_path):
    db_path = str(tmp_path / "violations.db")
    store = ViolationStore(db_path=db_path)

    video_path = str(tmp_path / "video_time.mp4")
    store.save_video_result(
        video_metadata=_sample_metadata(video_path, fps=25.0),
        violations=[_sample_violation(frame_number=50)],
    )

    payload = store.get_video_result(video_path)
    assert payload is not None
    violation = payload["violations"][0]
    assert abs(violation["violation_time_sec"] - 2.0) < 1e-6
