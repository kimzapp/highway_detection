"""SQLite persistence for processed videos and violation records."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from violations import Violation


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class ViolationStore:
    """Store and query processed-video metadata with violation events."""

    ARTIFACTS_DIR_NAME = "violation_artifacts"

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
                """
            )
            self._create_violations_table(conn)
            self._migrate_violations_table(conn)
            self._ensure_violation_indexes(conn)

    def _create_violations_table(self, conn: sqlite3.Connection):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (
                violation_id TEXT PRIMARY KEY,
                video_key TEXT NOT NULL,
                tracker_id INTEGER NOT NULL,
                violation_type TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                violation_time_sec REAL NOT NULL,
                end_time_sec REAL NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                camera_x INTEGER NOT NULL,
                camera_y INTEGER NOT NULL,
                bev_x INTEGER,
                bev_y INTEGER,
                artifact_clip_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(video_key) REFERENCES videos(video_key) ON DELETE CASCADE
            )
            """
        )

    def _ensure_violation_indexes(self, conn: sqlite3.Connection):
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_violations_video_key
                ON violations(video_key);
            CREATE INDEX IF NOT EXISTS idx_violations_type
                ON violations(violation_type);
            CREATE INDEX IF NOT EXISTS idx_violations_frame
                ON violations(frame_number);
            """
        )

    def _migrate_violations_table(self, conn: sqlite3.Connection):
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='violations'"
        ).fetchone()
        if table_exists is None:
            return

        columns_info = conn.execute("PRAGMA table_info(violations)").fetchall()
        existing_columns = {str(row[1]) for row in columns_info}
        required_columns = {
            "violation_id",
            "video_key",
            "tracker_id",
            "violation_type",
            "frame_number",
            "start_frame",
            "end_frame",
            "violation_time_sec",
            "end_time_sec",
            "class_id",
            "class_name",
            "camera_x",
            "camera_y",
            "bev_x",
            "bev_y",
            "artifact_clip_path",
            "created_at",
        }

        needs_migration = ("confidence" in existing_columns) or (not required_columns.issubset(existing_columns))
        if not needs_migration:
            return

        legacy_table = "violations_legacy"
        conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")
        conn.execute(f"ALTER TABLE violations RENAME TO {legacy_table}")
        self._create_violations_table(conn)

        legacy_rows = conn.execute(f"SELECT * FROM {legacy_table}").fetchall()
        migrated_rows = []
        now_iso = _utc_now_iso()

        for row in legacy_rows:
            record = dict(row)
            frame_number = int(record.get("frame_number") or 0)
            start_frame = int(record.get("start_frame") or frame_number)
            end_frame = int(record.get("end_frame") or frame_number)
            if end_frame < start_frame:
                end_frame = start_frame

            violation_time_sec = float(record.get("violation_time_sec") or (start_frame if start_frame >= 0 else 0))
            end_time_sec = float(record.get("end_time_sec") or violation_time_sec)

            migrated_rows.append(
                (
                    str(record.get("violation_id") or ""),
                    str(record.get("video_key") or ""),
                    int(record.get("tracker_id") or -1),
                    str(record.get("violation_type") or "UNKNOWN"),
                    frame_number,
                    start_frame,
                    end_frame,
                    violation_time_sec,
                    end_time_sec,
                    int(record.get("class_id") or -1),
                    str(record.get("class_name") or "unknown"),
                    int(record.get("camera_x") or -1),
                    int(record.get("camera_y") or -1),
                    int(record.get("bev_x") or -1),
                    int(record.get("bev_y") or -1),
                    record.get("artifact_clip_path"),
                    str(record.get("created_at") or now_iso),
                )
            )

        if migrated_rows:
            conn.executemany(
                """
                INSERT INTO violations (
                    violation_id,
                    video_key,
                    tracker_id,
                    violation_type,
                    frame_number,
                    start_frame,
                    end_frame,
                    violation_time_sec,
                    end_time_sec,
                    class_id,
                    class_name,
                    camera_x,
                    camera_y,
                    bev_x,
                    bev_y,
                    artifact_clip_path,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                migrated_rows,
            )

        conn.execute(f"DROP TABLE {legacy_table}")

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

    def _artifact_root_dir(self) -> str:
        project_root = Path(__file__).resolve().parent.parent
        return os.path.abspath(str(project_root / "outputs" / self.ARTIFACTS_DIR_NAME))

    def _discover_artifact_clip_path(self, video_key: str, violation_id: str) -> Optional[str]:
        artifact_dir = os.path.join(self._artifact_root_dir(), str(video_key))
        if not os.path.isdir(artifact_dir):
            return None

        prefix = f"{violation_id}_"
        try:
            for entry in os.scandir(artifact_dir):
                if not entry.is_file():
                    continue
                name = entry.name
                if name.startswith(prefix) and name.lower().endswith(".mp4"):
                    return os.path.abspath(entry.path)
        except OSError:
            return None

        return None

    def _resolve_artifact_clip_path(
        self,
        *,
        video_key: str,
        violation_id: str,
        artifact_clip_path: Optional[str],
    ) -> Optional[str]:
        if artifact_clip_path:
            normalized = os.path.abspath(str(artifact_clip_path))
            if os.path.exists(normalized):
                return normalized

        return self._discover_artifact_clip_path(video_key=video_key, violation_id=violation_id)

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
            start_frame = int(item_dict.get("start_frame", frame_number))
            end_frame_raw = item_dict.get("end_frame", frame_number)
            end_frame = int(end_frame_raw if end_frame_raw is not None else frame_number)
            if end_frame < start_frame:
                end_frame = start_frame

            violation_type = str(item_dict.get("type", "UNKNOWN"))
            tracker_id = int(item_dict.get("tracker_id", -1))
            class_id = int(item_dict.get("class_id", -1))
            class_name = str(item_dict.get("class_name", "unknown"))
            position = tuple(item_dict.get("position") or (-1, -1))
            bev_position = tuple(item_dict.get("bev_position") or (-1, -1))

            violation_id = self.make_violation_id(
                video_key=video_key,
                tracker_id=tracker_id,
                violation_type=violation_type,
                frame_number=frame_number,
            )

            artifact_clip_path = item_dict.get("artifact_clip_path")
            if not artifact_clip_path:
                extra_info = item_dict.get("extra_info") or {}
                artifact_clip_path = extra_info.get("artifact_clip_path")

            rows.append(
                (
                    violation_id,
                    video_key,
                    tracker_id,
                    violation_type,
                    frame_number,
                    start_frame,
                    end_frame,
                    start_frame / fps_safe,
                    end_frame / fps_safe,
                    class_id,
                    class_name,
                    int(position[0]),
                    int(position[1]),
                    int(bev_position[0]) if len(bev_position) > 0 else -1,
                    int(bev_position[1]) if len(bev_position) > 1 else -1,
                    str(artifact_clip_path) if artifact_clip_path else None,
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
                        start_frame,
                        end_frame,
                        violation_time_sec,
                        end_time_sec,
                        class_id,
                        class_name,
                        camera_x,
                        camera_y,
                        bev_x,
                        bev_y,
                        artifact_clip_path,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

        return self.get_video_result_by_key(video_key)

    def list_videos(
        self,
        *,
        started_from: Optional[str] = None,
        started_to: Optional[str] = None,
        finished_from: Optional[str] = None,
        finished_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all processed videos with violation counters and optional time filters."""
        where_parts = []
        params: List[Any] = []

        if started_from:
            where_parts.append("v.started_at >= ?")
            params.append(started_from)
        if started_to:
            where_parts.append("v.started_at <= ?")
            params.append(started_to)
        if finished_from:
            where_parts.append("v.finished_at >= ?")
            params.append(finished_from)
        if finished_to:
            where_parts.append("v.finished_at <= ?")
            params.append(finished_to)

        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)

        query = f"""
            SELECT
                v.video_key,
                v.video_path,
                v.file_name,
                v.fps,
                v.total_frames,
                v.width,
                v.height,
                v.output_video_path,
                v.started_at,
                v.finished_at,
                v.updated_at,
                COUNT(vi.violation_id) AS violation_count
            FROM videos v
            LEFT JOIN violations vi ON vi.video_key = v.video_key
            {where_clause}
            GROUP BY
                v.video_key,
                v.video_path,
                v.file_name,
                v.fps,
                v.total_frames,
                v.width,
                v.height,
                v.output_video_path,
                v.started_at,
                v.finished_at,
                v.updated_at
            ORDER BY v.started_at DESC, v.updated_at DESC
        """

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def get_video_result_by_key(self, video_key: str) -> Optional[Dict[str, Any]]:
        """Read video metadata and all violations by stable video key."""

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
                  SELECT violation_id, tracker_id, violation_type, frame_number, start_frame, end_frame,
                      violation_time_sec, end_time_sec, class_id, class_name,
                      camera_x, camera_y, bev_x, bev_y, artifact_clip_path, created_at
                FROM violations
                WHERE video_key = ?
                ORDER BY frame_number ASC, tracker_id ASC
                """,
                (video_key,),
            ).fetchall()

            updates: List[tuple[str, str]] = []
            hydrated_rows: List[Dict[str, Any]] = []
            for row in violation_rows:
                record = dict(row)
                resolved_path = self._resolve_artifact_clip_path(
                    video_key=video_key,
                    violation_id=str(record.get("violation_id") or ""),
                    artifact_clip_path=record.get("artifact_clip_path"),
                )
                if resolved_path and resolved_path != record.get("artifact_clip_path"):
                    record["artifact_clip_path"] = resolved_path
                    updates.append((resolved_path, str(record.get("violation_id") or "")))
                hydrated_rows.append(record)

            if updates:
                conn.executemany(
                    """
                    UPDATE violations
                    SET artifact_clip_path = ?
                    WHERE violation_id = ?
                    """,
                    updates,
                )
                conn.commit()

        return self._build_video_payload_from_dicts(dict(video_row), hydrated_rows)

    def get_violations_by_video(
        self,
        video_key: str,
        *,
        violation_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get violations for a single video with optional violation-type filter."""
        query = """
             SELECT violation_id, tracker_id, violation_type, frame_number, start_frame, end_frame,
                 violation_time_sec, end_time_sec, class_id, class_name,
                 camera_x, camera_y, bev_x, bev_y, artifact_clip_path, created_at
            FROM violations
            WHERE video_key = ?
        """
        params: List[Any] = [video_key]
        if violation_type and violation_type != "Tất cả":
            query += " AND violation_type = ?"
            params.append(violation_type)
        query += " ORDER BY frame_number ASC, tracker_id ASC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

            updates: List[tuple[str, str]] = []
            hydrated_records: List[Dict[str, Any]] = []
            for row in rows:
                record = dict(row)
                resolved_path = self._resolve_artifact_clip_path(
                    video_key=video_key,
                    violation_id=str(record.get("violation_id") or ""),
                    artifact_clip_path=record.get("artifact_clip_path"),
                )
                if resolved_path and resolved_path != record.get("artifact_clip_path"):
                    record["artifact_clip_path"] = resolved_path
                    updates.append((resolved_path, str(record.get("violation_id") or "")))
                hydrated_records.append(record)

            if updates:
                conn.executemany(
                    """
                    UPDATE violations
                    SET artifact_clip_path = ?
                    WHERE violation_id = ?
                    """,
                    updates,
                )
                conn.commit()

        violations: List[Dict[str, Any]] = []
        for record in hydrated_records:
            record["position"] = [record.pop("camera_x"), record.pop("camera_y")]
            record["bev_position"] = [record.pop("bev_x"), record.pop("bev_y")]
            violations.append(record)
        return violations

    def _build_video_payload_from_dicts(
        self,
        video: Dict[str, Any],
        violation_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        video["processing_config"] = json.loads(video.pop("processing_config_json"))

        violations: List[Dict[str, Any]] = []
        for record in violation_records:
            record["position"] = [record.pop("camera_x"), record.pop("camera_y")]
            record["bev_position"] = [record.pop("bev_x"), record.pop("bev_y")]
            violations.append(record)

        return {
            "video": video,
            "violations": violations,
        }

    def _build_video_payload(
        self,
        video_row: Optional[sqlite3.Row],
        violation_rows: Iterable[sqlite3.Row],
    ) -> Optional[Dict[str, Any]]:
        if video_row is None:
            return None

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