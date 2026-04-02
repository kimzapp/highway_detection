"""Asynchronous violation artifact writer with per-video cleanup helpers."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import queue
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import supervision as sv


def make_video_key(video_path: str) -> str:
    normalized_path = os.path.normcase(os.path.abspath(video_path))
    size = -1
    mtime_ns = -1
    try:
        stat = os.stat(video_path)
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
    except OSError:
        pass

    raw = f"{normalized_path}|{size}|{mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def make_violation_id(video_key: str, tracker_id: int, violation_type: str, frame_number: int) -> str:
    raw = f"{video_key}|{tracker_id}|{violation_type}|{frame_number}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def get_default_artifact_root() -> str:
    project_root = Path(__file__).resolve().parent.parent
    root = os.path.abspath(str(project_root / "outputs" / "violation_artifacts"))
    os.makedirs(root, exist_ok=True)
    return root


def get_artifact_root(artifact_root: Optional[str] = None) -> str:
    if artifact_root and str(artifact_root).strip():
        root = os.path.abspath(str(artifact_root).strip())
        os.makedirs(root, exist_ok=True)
        return root
    return get_default_artifact_root()


def get_video_artifact_dir(video_path: str, artifact_root: Optional[str] = None) -> str:
    root = get_artifact_root(artifact_root)
    return os.path.join(root, make_video_key(video_path))


def cleanup_video_artifacts(video_path: str, artifact_root: Optional[str] = None) -> str:
    """Remove stale artifacts for a video before a new processing run."""
    target_dir = get_video_artifact_dir(video_path, artifact_root)
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def serialize_tracked_detections(
    detections: Optional[sv.Detections],
    class_names: Dict[int, str],
) -> List[Dict[str, Any]]:
    if detections is None or len(detections) == 0:
        return []

    boxes = detections.xyxy
    class_ids = detections.class_id
    tracker_ids = detections.tracker_id
    confidences = detections.confidence

    if tracker_ids is None:
        return []

    rows: List[Dict[str, Any]] = []
    for box, class_id, tracker_id, confidence in zip(boxes, class_ids, tracker_ids, confidences):
        if tracker_id is None:
            continue

        rows.append(
            {
                "tracker_id": int(tracker_id),
                "class_id": int(class_id),
                "class_name": str(class_names.get(int(class_id), f"class_{int(class_id)}")),
                "confidence": float(confidence),
                "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            }
        )

    return rows


def _draw_primary_target(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    tracker_id: int,
) -> Optional[Dict[str, Any]]:
    target = None
    for row in detections:
        if int(row.get("tracker_id", -1)) == int(tracker_id):
            target = row
            break

    if target is None:
        return None

    x1, y1, x2, y2 = target["bbox"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    label = f"#{target['tracker_id']} {target['class_name']} {target['confidence']:.2f}"
    cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + 260, y1), (0, 0, 255), -1)
    cv2.putText(frame, label, (x1 + 4, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return target


def _draw_metadata_overlay(
    frame: np.ndarray,
    *,
    violation_id: str,
    tracker_id: int,
    class_name: str,
    violation_type: str,
    start_frame: int,
    end_frame: int,
    current_frame: int,
    fps: float,
):
    fps_safe = fps if fps > 0 else 1.0
    current_time_sec = current_frame / fps_safe
    start_time_sec = start_frame / fps_safe
    end_time_sec = end_frame / fps_safe

    lines = [
        f"Violation ID: {violation_id}",
        f"Tracker ID: {tracker_id} | Class: {class_name}",
        f"Type: {violation_type}",
        f"Frame: {current_frame} | t={current_time_sec:.2f}s",
        f"Range: {start_frame}-{end_frame} ({start_time_sec:.2f}s-{end_time_sec:.2f}s)",
    ]

    box_w = 620
    box_h = 18 + (len(lines) * 24)
    cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (0, 220, 255), 1)

    y = 36
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
        y += 24


def _draw_valid_zones(frame: np.ndarray, valid_zone_polygons: List[List[List[int]]]):
    if not valid_zone_polygons:
        return

    overlay = frame.copy()
    for points in valid_zone_polygons:
        polygon = np.array(points, dtype=np.int32)
        if polygon.shape[0] < 3:
            continue

        cv2.fillPoly(overlay, [polygon], (30, 120, 30))
        cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)

        anchor_x = int(np.min(polygon[:, 0]))
        anchor_y = int(np.min(polygon[:, 1]))
        label_pos = (max(0, anchor_x), max(18, anchor_y - 6))
        cv2.putText(frame, "VALID ZONE", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def _artifact_worker_main(
    command_queue,
    status_queue,
    *,
    artifact_dir: str,
    fps: float,
    max_buffer_frames: int,
    valid_zone_polygons: Optional[List[List[List[int]]]],
):
    os.makedirs(artifact_dir, exist_ok=True)
    frame_buffer: deque = deque(maxlen=max_buffer_frames)
    events: Dict[str, Dict[str, Any]] = {}
    completed_paths: Dict[str, Optional[str]] = {}
    safe_zone_polygons = list(valid_zone_polygons or [])

    status_queue.put({"state": "ready"})

    def _ensure_writer(event: Dict[str, Any], frame: np.ndarray):
        if event.get("writer") is not None:
            return event["writer"]

        height, width = frame.shape[:2]
        path = os.path.join(
            artifact_dir,
            f"{event['violation_id']}_trk{event['tracker_id']}_{event['violation_type']}_{event['start_frame']}.mp4",
        )
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), event["fps"], (width, height))
        if not writer.isOpened():
            return None

        event["writer"] = writer
        event["clip_path"] = os.path.abspath(path)
        return writer

    def _close_event(violation_id: str):
        event = events.get(violation_id)
        if event is None:
            return

        writer = event.get("writer")
        if writer is not None:
            writer.release()

        completed_paths[violation_id] = event.get("clip_path")
        events.pop(violation_id, None)

    def _write_event_frame(event: Dict[str, Any], frame_number: int, frame: np.ndarray, detections: List[Dict[str, Any]]):
        if frame_number < int(event["start_frame"]):
            return

        if event.get("end_frame") is not None and frame_number > int(event["end_frame"]):
            return

        if frame_number <= int(event.get("last_written_frame", -1)):
            return

        writer = _ensure_writer(event, frame)
        if writer is None:
            return

        out_frame = frame.copy()
        _draw_valid_zones(out_frame, safe_zone_polygons)
        current_target = _draw_primary_target(out_frame, detections, int(event["tracker_id"]))
        if current_target is not None:
            event["last_target_detection"] = dict(current_target)
        elif event.get("last_target_detection") is not None:
            _draw_primary_target(
                out_frame,
                [event["last_target_detection"]],
                int(event["tracker_id"]),
            )
        _draw_metadata_overlay(
            out_frame,
            violation_id=str(event["violation_id"]),
            tracker_id=int(event["tracker_id"]),
            class_name=str(event["class_name"]),
            violation_type=str(event["violation_type"]),
            start_frame=int(event["start_frame"]),
            end_frame=int(event.get("end_frame") if event.get("end_frame") is not None else frame_number),
            current_frame=frame_number,
            fps=float(event["fps"]),
        )
        writer.write(out_frame)
        event["last_written_frame"] = frame_number

    def _flush_frame(frame_number: int, frame: np.ndarray, detections: List[Dict[str, Any]]):
        for violation_id, event in list(events.items()):
            _write_event_frame(event, frame_number, frame, detections)
            end_frame = event.get("end_frame")
            if end_frame is not None and frame_number >= int(end_frame):
                _close_event(violation_id)

    try:
        while True:
            command = command_queue.get()
            cmd_type = command.get("type")

            if cmd_type == "shutdown":
                break

            if cmd_type == "frame":
                frame_number = int(command["frame_number"])
                frame = command["frame"]
                detections = command.get("detections", [])
                frame_buffer.append((frame_number, frame, detections))
                _flush_frame(frame_number, frame, detections)
                continue

            if cmd_type == "violation_started":
                violation_id = str(command["violation_id"])
                events[violation_id] = {
                    "violation_id": violation_id,
                    "tracker_id": int(command["tracker_id"]),
                    "class_name": str(command.get("class_name") or "unknown"),
                    "violation_type": str(command.get("violation_type") or "UNKNOWN"),
                    "start_frame": int(command["start_frame"]),
                    "end_frame": None,
                    "writer": None,
                    "clip_path": None,
                    "last_written_frame": -1,
                    "last_target_detection": None,
                    "fps": float(command.get("fps") or fps),
                }

                for buffered_frame_number, buffered_frame, buffered_detections in list(frame_buffer):
                    _write_event_frame(
                        events[violation_id],
                        buffered_frame_number,
                        buffered_frame,
                        buffered_detections,
                    )
                continue

            if cmd_type == "violation_ended":
                violation_id = str(command["violation_id"])
                event = events.get(violation_id)
                if event is None:
                    continue
                event["end_frame"] = int(command["end_frame"])

                if int(event.get("last_written_frame", -1)) >= int(event["end_frame"]):
                    _close_event(violation_id)
                continue

    except Exception as exc:
        status_queue.put({"state": "error", "message": str(exc)})
    finally:
        for violation_id in list(events.keys()):
            _close_event(violation_id)
        status_queue.put({"state": "done", "artifact_paths": completed_paths})


class AsyncViolationArtifactWriter:
    """Asynchronous artifact writer for violation evidence clips."""

    def __init__(
        self,
        *,
        video_path: str,
        fps: float,
        artifact_root: Optional[str] = None,
        valid_zone_polygons: Optional[List[Any]] = None,
        max_queue_size: int = 800,
        max_buffer_frames: int = 320,
    ):
        self.video_path = video_path
        self.video_key = make_video_key(video_path)
        self.artifact_root = get_artifact_root(artifact_root)
        self.artifact_dir = os.path.join(self.artifact_root, self.video_key)
        self.fps = float(fps)
        self.max_buffer_frames = max(60, int(max_buffer_frames))
        self.valid_zone_polygons = self._normalize_zone_polygons(valid_zone_polygons)

        self._ctx = mp.get_context("spawn")
        self._command_queue = self._ctx.Queue(maxsize=max_queue_size)
        self._status_queue = self._ctx.Queue(maxsize=16)
        self._process = self._ctx.Process(
            target=_artifact_worker_main,
            args=(self._command_queue, self._status_queue),
            kwargs={
                "artifact_dir": self.artifact_dir,
                "fps": self.fps,
                "max_buffer_frames": self.max_buffer_frames,
                "valid_zone_polygons": self.valid_zone_polygons,
            },
            name="AsyncViolationArtifactWriter",
            daemon=True,
        )

        self._started = False
        self._closed = False
        self._error: Optional[str] = None
        self._dropped_messages = 0
        self._dropped_messages_by_type: Dict[str, int] = {}

    @staticmethod
    def _is_critical_command(command_type: str) -> bool:
        return command_type in {"violation_started", "violation_ended", "shutdown"}

    @staticmethod
    def _normalize_zone_polygons(valid_zone_polygons: Optional[List[Any]]) -> List[List[List[int]]]:
        if not valid_zone_polygons:
            return []

        normalized: List[List[List[int]]] = []
        for polygon in valid_zone_polygons:
            points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
            if points.shape[0] < 3:
                continue
            normalized.append(points.tolist())

        return normalized

    @property
    def dropped_messages(self) -> int:
        return self._dropped_messages

    @property
    def dropped_messages_by_type(self) -> Dict[str, int]:
        return dict(self._dropped_messages_by_type)

    def start(self, timeout_s: float = 10.0):
        if self._started:
            return

        os.makedirs(self.artifact_dir, exist_ok=True)
        self._process.start()
        self._started = True

        try:
            status = self._status_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise RuntimeError("Timeout waiting for artifact writer startup") from exc

        if status.get("state") != "ready":
            self._error = status.get("message", "Artifact writer failed to start")
            raise RuntimeError(self._error)

    def _put_command(
        self,
        command: Dict[str, Any],
        *,
        timeout_s: float = 0.2,
        critical_retries: int = 15,
    ) -> bool:
        self._poll_status_queue()
        if self._error is not None:
            raise RuntimeError(self._error)

        cmd_type = str(command.get("type") or "unknown")
        is_critical = self._is_critical_command(cmd_type)

        if is_critical:
            for _ in range(max(1, int(critical_retries))):
                try:
                    self._command_queue.put(command, timeout=timeout_s)
                    return True
                except queue.Full:
                    self._poll_status_queue()
                    if self._error is not None:
                        raise RuntimeError(self._error)
                    if not self._process.is_alive():
                        raise RuntimeError("Artifact writer process stopped unexpectedly")

            raise RuntimeError(f"Artifact command queue saturated while enqueueing critical command: {cmd_type}")

        try:
            self._command_queue.put(command, timeout=timeout_s)
            return True
        except queue.Full:
            self._dropped_messages += 1
            self._dropped_messages_by_type[cmd_type] = self._dropped_messages_by_type.get(cmd_type, 0) + 1
            return False

    def enqueue_frame(self, frame_number: int, frame: np.ndarray, detections: List[Dict[str, Any]]):
        if self._closed or not self._started:
            return

        self._put_command(
            {
                "type": "frame",
                "frame_number": int(frame_number),
                "frame": frame,
                "detections": detections,
            }
        )

    def on_violation_started(self, event: Dict[str, Any]):
        if self._closed or not self._started:
            return

        payload = {
            "type": "violation_started",
            "violation_id": str(event["violation_id"]),
            "tracker_id": int(event["tracker_id"]),
            "class_name": str(event.get("class_name") or "unknown"),
            "violation_type": str(event.get("violation_type") or "UNKNOWN"),
            "start_frame": int(event["start_frame"]),
            "fps": float(event.get("fps") or self.fps),
        }
        self._put_command(payload)

    def on_violation_ended(self, event: Dict[str, Any]):
        if self._closed or not self._started:
            return

        payload = {
            "type": "violation_ended",
            "violation_id": str(event["violation_id"]),
            "end_frame": int(event["end_frame"]),
        }
        self._put_command(payload)

    def close(self, timeout_s: float = 12.0) -> Dict[str, Any]:
        if self._closed:
            return {
                "artifact_paths": {},
                "dropped_messages": self._dropped_messages,
                "dropped_messages_by_type": self.dropped_messages_by_type,
            }
        self._closed = True

        done_payload: Optional[Dict[str, Any]] = None

        if self._started:
            try:
                self._put_command({"type": "shutdown"}, timeout_s=0.2, critical_retries=25)
            except RuntimeError:
                self._process.terminate()

            started_at = time.monotonic()
            while (time.monotonic() - started_at) < timeout_s:
                done_payload = self._poll_status_queue(done_only=True) or done_payload
                if done_payload is not None:
                    break
                if not self._process.is_alive():
                    break
                self._process.join(timeout=0.25)

            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)

        done_payload = self._poll_status_queue(done_only=True) or done_payload
        if self._error is not None:
            raise RuntimeError(self._error)

        artifact_paths = {}
        if done_payload is not None:
            artifact_paths = dict(done_payload.get("artifact_paths") or {})

        return {
            "artifact_paths": artifact_paths,
            "dropped_messages": self._dropped_messages,
            "dropped_messages_by_type": self.dropped_messages_by_type,
            "artifact_dir": self.artifact_dir,
        }

    def _poll_status_queue(self, done_only: bool = False) -> Optional[Dict[str, Any]]:
        done_payload = None
        while True:
            try:
                status = self._status_queue.get_nowait()
            except queue.Empty:
                break

            if status.get("state") == "error":
                self._error = status.get("message", "Unknown artifact writer error")
            if status.get("state") == "done":
                done_payload = status
                if done_only:
                    break

        return done_payload
