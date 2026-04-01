"""Tests for async violation artifact writer workflow."""

from __future__ import annotations

import os

import numpy as np

from process.artifact_writer import (
    AsyncViolationArtifactWriter,
    cleanup_video_artifacts,
    get_video_artifact_dir,
)


def _detection(tracker_id: int, class_name: str = "car"):
    return [
        {
            "tracker_id": tracker_id,
            "class_id": 2,
            "class_name": class_name,
            "confidence": 0.91,
            "bbox": [10, 10, 40, 40],
        }
    ]


def test_cleanup_video_artifacts_removes_stale_content(tmp_path):
    video_path = str(tmp_path / "video_demo.mp4")
    open(video_path, "wb").close()

    artifact_root = str(tmp_path / "artifacts")
    target_dir = cleanup_video_artifacts(video_path, artifact_root=artifact_root)

    stale_file = os.path.join(target_dir, "old.txt")
    with open(stale_file, "w", encoding="utf-8") as f:
        f.write("stale")

    cleaned_dir = cleanup_video_artifacts(video_path, artifact_root=artifact_root)
    assert cleaned_dir == target_dir
    assert os.path.isdir(cleaned_dir)
    assert not os.path.exists(stale_file)


def test_async_artifact_writer_creates_clip_for_event(tmp_path):
    video_path = str(tmp_path / "video_demo.mp4")
    open(video_path, "wb").close()

    artifact_root = str(tmp_path / "artifacts")
    cleanup_video_artifacts(video_path, artifact_root=artifact_root)

    writer = AsyncViolationArtifactWriter(
        video_path=video_path,
        fps=20.0,
        artifact_root=artifact_root,
        max_queue_size=64,
        max_buffer_frames=64,
    )
    writer.start()

    violation_id = "viol_001"
    writer.on_violation_started(
        {
            "violation_id": violation_id,
            "tracker_id": 7,
            "class_name": "car",
            "violation_type": "WRONG_LANE",
            "start_frame": 1,
            "fps": 20.0,
        }
    )

    for frame_number in range(6):
        frame = np.zeros((72, 128, 3), dtype=np.uint8)
        if frame_number in (1, 2, 3):
            detections = _detection(7)
        else:
            detections = []
        writer.enqueue_frame(frame_number, frame, detections)

    writer.on_violation_ended({"violation_id": violation_id, "end_frame": 3})
    summary = writer.close()

    artifact_paths = summary.get("artifact_paths") or {}
    clip_path = artifact_paths.get(violation_id)

    assert clip_path is not None
    assert os.path.exists(clip_path)
    assert os.path.getsize(clip_path) > 0
    assert get_video_artifact_dir(video_path, artifact_root=artifact_root) in clip_path
