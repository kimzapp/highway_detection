"""SQLite persistence for processed videos and violation records."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from violations import Violation


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class ViolationStore:
    """Store and query processed-video metadata with violation events."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            db_path = str(project_root / "outputs" / "violations.db")

        self._db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._ensure_schema()

    @property
    def db_path(self) -> str:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_key TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    fps REAL NOT NULL,
                    total_frames INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    processing_config_json TEXT NOT NULL,
                    output_video_path TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS violations (
                    violation_id TEXT PRIMARY KEY,
                    video_key TEXT NOT NULL,
                    tracker_id INTEGER NOT NULL,
                    violation_type TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    violation_time_sec REAL NOT NULL,
                    class_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    camera_x INTEGER NOT NULL,
                    camera_y INTEGER NOT NULL,
                    bev_x INTEGER,
                    bev_y INTEGER,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(video_key) REFERENCES videos(video_key) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_violations_video_key
                    ON violations(video_key);
                CREATE INDEX IF NOT EXISTS idx_violations_type
                    ON violations(violation_type);
                CREATE INDEX IF NOT EXISTS idx_violations_frame
                    ON violations(frame_number);
                """
            )

    def make_video_key(self, video_path: str) -> str:
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

    def make_violation_id(
        self,
        video_key: str,
        tracker_id: int,
        violation_type: str,
        frame_number: int,
    ) -> str:
        raw = f"{video_key}|{tracker_id}|{violation_type}|{frame_number}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def save_video_result(
        self,
        video_metadata: Dict[str, Any],
        violations: Iterable[Violation | Dict[str, Any]],
    ) -> Dict[str, Any]:
        video_path = str(video_metadata["video_path"])
        video_key = self.make_video_key(video_path)
        output_video_path = video_metadata.get("output_video_path")
        now_iso = _utc_now_iso()

        rows = []
        fps = float(video_metadata.get("fps") or 0.0)
        fps_safe = fps if fps > 0 else 1.0

        for item in violations:
            item_dict = item.to_dict() if isinstance(item, Violation) else dict(item)

            frame_number = int(item_dict.get("frame_number", 0))
            violation_type = str(item_dict.get("type", "UNKNOWN"))
            tracker_id = int(item_dict.get("tracker_id", -1))
            class_id = int(item_dict.get("class_id", -1))
            class_name = str(item_dict.get("class_name", "unknown"))
            position = tuple(item_dict.get("position") or (-1, -1))
            bev_position = tuple(item_dict.get("bev_position") or (-1, -1))
            confidence = float(item_dict.get("confidence", 0.0))

            rows.append(
                (
                    self.make_violation_id(
                        video_key=video_key,
                        tracker_id=tracker_id,
                        violation_type=violation_type,
                        frame_number=frame_number,
                    ),
                    video_key,
                    tracker_id,
                    violation_type,
                    frame_number,
                    frame_number / fps_safe,
                    class_id,
                    class_name,
                    int(position[0]),
                    int(position[1]),
                    int(bev_position[0]) if len(bev_position) > 0 else -1,
                    int(bev_position[1]) if len(bev_position) > 1 else -1,
                    confidence,
                    now_iso,
                )
            )

        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM violations WHERE video_key = ?", (video_key,))
            conn.execute(
                """
                INSERT INTO videos (
                    video_key,
                    video_path,
                    file_name,
                    fps,
                    total_frames,
                    width,
                    height,
                    processing_config_json,
                    output_video_path,
                    started_at,
                    finished_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_key) DO UPDATE SET
                    video_path=excluded.video_path,
                    file_name=excluded.file_name,
                    fps=excluded.fps,
                    total_frames=excluded.total_frames,
                    width=excluded.width,
                    height=excluded.height,
                    processing_config_json=excluded.processing_config_json,
                    output_video_path=excluded.output_video_path,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    updated_at=excluded.updated_at
                """,
                (
                    video_key,
                    video_path,
                    os.path.basename(video_path),
                    float(video_metadata.get("fps") or 0.0),
                    int(video_metadata.get("total_frames") or 0),
                    int(video_metadata.get("width") or 0),
                    int(video_metadata.get("height") or 0),
                    json.dumps(video_metadata.get("processing_config") or {}, ensure_ascii=False),
                    output_video_path,
                    str(video_metadata.get("started_at") or now_iso),
                    str(video_metadata.get("finished_at") or now_iso),
                    now_iso,
                ),
            )

            if rows:
                conn.executemany(
                    """
                    INSERT INTO violations (
                        violation_id,
                        video_key,
                        tracker_id,
                        violation_type,
                        frame_number,
                        violation_time_sec,
                        class_id,
                        class_name,
                        camera_x,
                        camera_y,
                        bev_x,
                        bev_y,
                        confidence,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

            conn.commit()

        return {
            "video_key": video_key,
            "violations_saved": len(rows),
            "db_path": self._db_path,
        }

    def get_video_result(self, video_path: str) -> Optional[Dict[str, Any]]:
        video_key = self.make_video_key(video_path)

        with self._connect() as conn:
            video_row = conn.execute(
                """
                SELECT video_key, video_path, file_name, fps, total_frames, width, height,
                       processing_config_json, output_video_path, started_at, finished_at, updated_at
                FROM videos WHERE video_key = ?
                """,
                (video_key,),
            ).fetchone()

            if video_row is None:
                return None

            violation_rows = conn.execute(
                """
                SELECT violation_id, tracker_id, violation_type, frame_number, violation_time_sec,
                       class_id, class_name, camera_x, camera_y, bev_x, bev_y, confidence, created_at
                FROM violations
                WHERE video_key = ?
                ORDER BY frame_number ASC, tracker_id ASC
                """,
                (video_key,),
            ).fetchall()

        video = dict(video_row)
        video["processing_config"] = json.loads(video.pop("processing_config_json"))

        violations = []
        for row in violation_rows:
            record = dict(row)
            record["position"] = [record.pop("camera_x"), record.pop("camera_y")]
            record["bev_position"] = [record.pop("bev_x"), record.pop("bev_y")]
            violations.append(record)

        return {
            "video": video,
            "violations": violations,
        }