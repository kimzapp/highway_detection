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
    assert payload["violations"][0]["start_frame"] == 50
    assert payload["violations"][0]["end_frame"] == 50


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


def test_save_and_read_violation_with_end_frame(tmp_path):
    db_path = str(tmp_path / "violations.db")
    store = ViolationStore(db_path=db_path)

    video_path = str(tmp_path / "video_event.mp4")
    store.save_video_result(
        video_metadata=_sample_metadata(video_path, fps=20.0),
        violations=[
            {
                "type": "WRONG_LANE",
                "tracker_id": 10,
                "class_id": 2,
                "class_name": "car",
                "position": (100, 200),
                "bev_position": (10, 20),
                "frame_number": 40,
                "start_frame": 40,
                "end_frame": 50,
            }
        ],
    )

    payload = store.get_video_result(video_path)
    assert payload is not None
    violation = payload["violations"][0]
    assert violation["start_frame"] == 40
    assert violation["end_frame"] == 50
    assert abs(violation["violation_time_sec"] - 2.0) < 1e-6
    assert abs(violation["end_time_sec"] - 2.5) < 1e-6


def test_list_videos_with_started_and_finished_filters(tmp_path):
    db_path = str(tmp_path / "violations.db")
    store = ViolationStore(db_path=db_path)

    video_a = str(tmp_path / "video_a.mp4")
    video_b = str(tmp_path / "video_b.mp4")

    meta_a = _sample_metadata(video_a, fps=30.0)
    meta_a["started_at"] = "2026-03-01T10:00:00Z"
    meta_a["finished_at"] = "2026-03-01T10:10:00Z"

    meta_b = _sample_metadata(video_b, fps=30.0)
    meta_b["started_at"] = "2026-04-01T11:00:00Z"
    meta_b["finished_at"] = "2026-04-01T11:12:00Z"

    store.save_video_result(video_metadata=meta_a, violations=[_sample_violation(frame_number=12)])
    store.save_video_result(video_metadata=meta_b, violations=[_sample_violation(frame_number=20)])

    result_started = store.list_videos(
        started_from="2026-04-01T00:00:00Z",
        started_to="2026-04-02T00:00:00Z",
    )
    assert len(result_started) == 1
    assert result_started[0]["file_name"] == "video_b.mp4"

    result_finished = store.list_videos(
        finished_from="2026-03-01T00:00:00Z",
        finished_to="2026-03-02T00:00:00Z",
    )
    assert len(result_finished) == 1
    assert result_finished[0]["file_name"] == "video_a.mp4"


def test_get_violations_by_video_with_type_filter(tmp_path):
    db_path = str(tmp_path / "violations.db")
    store = ViolationStore(db_path=db_path)

    video_path = str(tmp_path / "video_types.mp4")
    video_key = store.make_video_key(video_path)

    store.save_video_result(
        video_metadata=_sample_metadata(video_path, fps=30.0),
        violations=[
            _sample_violation(frame_number=12, tracker_id=1),
            Violation(
                violation_type=ViolationType.INVALID_VEHICLE,
                tracker_id=2,
                class_id=0,
                class_name="person",
                position=(120, 250),
                bev_position=(11, 21),
                frame_number=13,
            ),
        ],
    )

    all_rows = store.get_violations_by_video(video_key)
    assert len(all_rows) == 2

    wrong_lane_rows = store.get_violations_by_video(video_key, violation_type="WRONG_LANE")
    assert len(wrong_lane_rows) == 1
    assert wrong_lane_rows[0]["violation_type"] == "WRONG_LANE"

    invalid_vehicle_rows = store.get_violations_by_video(video_key, violation_type="INVALID_VEHICLE")
    assert len(invalid_vehicle_rows) == 1
    assert invalid_vehicle_rows[0]["violation_type"] == "INVALID_VEHICLE"


def test_artifact_path_fallback_discovery_and_persist(tmp_path):
    class LocalArtifactStore(ViolationStore):
        def __init__(self, db_path: str, artifact_root: str):
            self._artifact_root_override = artifact_root
            super().__init__(db_path=db_path)

        def _artifact_root_dir(self) -> str:
            return self._artifact_root_override

    db_path = str(tmp_path / "violations.db")
    artifact_root = str(tmp_path / "violation_artifacts")
    store = LocalArtifactStore(db_path=db_path, artifact_root=artifact_root)

    video_path = str(tmp_path / "video_artifact_lookup.mp4")
    metadata = _sample_metadata(video_path, fps=20.0)

    frame_number = 15
    tracker_id = 21
    violation_type = "WRONG_LANE"
    video_key = store.make_video_key(video_path)
    violation_id = store.make_violation_id(video_key, tracker_id, violation_type, frame_number)

    store.save_video_result(
        video_metadata=metadata,
        violations=[
            {
                "type": violation_type,
                "tracker_id": tracker_id,
                "class_id": 2,
                "class_name": "car",
                "position": (10, 20),
                "bev_position": (5, 8),
                "frame_number": frame_number,
                "start_frame": frame_number,
                "end_frame": frame_number + 3,
                "artifact_clip_path": None,
            }
        ],
    )

    clip_dir = tmp_path / "violation_artifacts" / video_key
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / f"{violation_id}_trk{tracker_id}_{violation_type}_{frame_number}.mp4"
    clip_path.write_bytes(b"\x00\x00\x00\x00")

    rows = store.get_violations_by_video(video_key)
    assert len(rows) == 1
    assert rows[0]["artifact_clip_path"] == str(clip_path.resolve())

    payload = store.get_video_result_by_key(video_key)
    assert payload is not None
    assert payload["violations"][0]["artifact_clip_path"] == str(clip_path.resolve())
