"""Database history browser page."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from storage import ViolationStore


class HistoryDatabasePage(QWidget):
    """Browse entire violation database by video and inspect violation details."""

    def __init__(self, store: ViolationStore, parent=None):
        super().__init__(parent)
        self._store = store
        self._videos: List[Dict[str, Any]] = []
        self._selected_video_key: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        title = QLabel("Database Lịch Sử Vi Phạm")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        root_layout.addWidget(title)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Loại vi phạm:"))
        self._violation_type_combo = QComboBox()
        self._violation_type_combo.addItems(["Tất cả", "WRONG_LANE", "INVALID_VEHICLE"])
        self._violation_type_combo.currentTextChanged.connect(self._refresh_violations_table)
        filter_row.addWidget(self._violation_type_combo)

        filter_row.addSpacing(12)

        self._use_started_filter = QCheckBox("Lọc started_at")
        self._use_started_filter.toggled.connect(self._on_time_filter_toggled)
        filter_row.addWidget(self._use_started_filter)

        filter_row.addWidget(QLabel("Từ"))
        self._started_from = QDateTimeEdit(QDateTime.currentDateTime().addDays(-30))
        self._started_from.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self._started_from.setCalendarPopup(True)
        self._started_from.setEnabled(False)
        filter_row.addWidget(self._started_from)

        filter_row.addWidget(QLabel("Đến"))
        self._started_to = QDateTimeEdit(QDateTime.currentDateTime())
        self._started_to.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self._started_to.setCalendarPopup(True)
        self._started_to.setEnabled(False)
        filter_row.addWidget(self._started_to)

        filter_row.addSpacing(12)

        self._use_finished_filter = QCheckBox("Lọc finished_at")
        self._use_finished_filter.toggled.connect(self._on_time_filter_toggled)
        filter_row.addWidget(self._use_finished_filter)

        filter_row.addWidget(QLabel("Từ"))
        self._finished_from = QDateTimeEdit(QDateTime.currentDateTime().addDays(-30))
        self._finished_from.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self._finished_from.setCalendarPopup(True)
        self._finished_from.setEnabled(False)
        filter_row.addWidget(self._finished_from)

        filter_row.addWidget(QLabel("Đến"))
        self._finished_to = QDateTimeEdit(QDateTime.currentDateTime())
        self._finished_to.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self._finished_to.setCalendarPopup(True)
        self._finished_to.setEnabled(False)
        filter_row.addWidget(self._finished_to)

        filter_row.addStretch()

        self._apply_btn = QPushButton("Áp dụng")
        self._apply_btn.clicked.connect(self.refresh_data)
        filter_row.addWidget(self._apply_btn)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.clicked.connect(self._reset_filters)
        filter_row.addWidget(self._reset_btn)

        self._refresh_btn = QPushButton("Làm mới")
        self._refresh_btn.clicked.connect(self.refresh_data)
        filter_row.addWidget(self._refresh_btn)

        root_layout.addLayout(filter_row)

        splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(QLabel("Danh sách video đã xử lý"))

        self._video_table = QTableWidget(0, 4)
        self._video_table.setHorizontalHeaderLabels([
            "Tên video",
            "Started",
            "Finished",
            "Số vi phạm",
        ])
        self._video_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._video_table.setSelectionMode(QTableWidget.SingleSelection)
        self._video_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._video_table.setAlternatingRowColors(True)
        self._video_table.verticalHeader().setVisible(False)
        self._video_table.horizontalHeader().setStretchLastSection(True)
        self._video_table.itemSelectionChanged.connect(self._on_video_selection_changed)
        left_layout.addWidget(self._video_table, 1)

        self._video_count_label = QLabel("Video: 0")
        self._video_count_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(self._video_count_label)

        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        right_layout.addWidget(QLabel("Chi tiết vi phạm"))

        self._video_meta_label = QLabel("Chưa chọn video.")
        self._video_meta_label.setWordWrap(True)
        self._video_meta_label.setStyleSheet("color: #444; padding: 6px;")
        right_layout.addWidget(self._video_meta_label)

        self._violation_table = QTableWidget(0, 7)
        self._violation_table.setHorizontalHeaderLabels([
            "Violation ID",
            "Time (s)",
            "Type",
            "Tracker",
            "Class",
            "Frame",
            "Confidence",
        ])
        self._violation_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._violation_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._violation_table.setAlternatingRowColors(True)
        self._violation_table.verticalHeader().setVisible(False)
        self._violation_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self._violation_table, 1)

        self._violation_count_label = QLabel("Tổng vi phạm: 0")
        self._violation_count_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(self._violation_count_label)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        root_layout.addWidget(splitter, 1)

        self._status_label = QLabel("Sẵn sàng")
        self._status_label.setStyleSheet("color: #666;")
        root_layout.addWidget(self._status_label)

    def _on_time_filter_toggled(self):
        use_started = self._use_started_filter.isChecked()
        use_finished = self._use_finished_filter.isChecked()

        self._started_from.setEnabled(use_started)
        self._started_to.setEnabled(use_started)
        self._finished_from.setEnabled(use_finished)
        self._finished_to.setEnabled(use_finished)

    def _reset_filters(self):
        self._violation_type_combo.setCurrentText("Tất cả")
        self._use_started_filter.setChecked(False)
        self._use_finished_filter.setChecked(False)
        now = QDateTime.currentDateTime()
        self._started_from.setDateTime(now.addDays(-30))
        self._started_to.setDateTime(now)
        self._finished_from.setDateTime(now.addDays(-30))
        self._finished_to.setDateTime(now)
        self.refresh_data()

    def refresh_data(self):
        started_from = self._qdatetime_to_iso(self._started_from) if self._use_started_filter.isChecked() else None
        started_to = self._qdatetime_to_iso(self._started_to) if self._use_started_filter.isChecked() else None
        finished_from = self._qdatetime_to_iso(self._finished_from) if self._use_finished_filter.isChecked() else None
        finished_to = self._qdatetime_to_iso(self._finished_to) if self._use_finished_filter.isChecked() else None

        selected_before = self._selected_video_key
        self._videos = self._store.list_videos(
            started_from=started_from,
            started_to=started_to,
            finished_from=finished_from,
            finished_to=finished_to,
        )

        self._video_table.setRowCount(len(self._videos))
        self._video_count_label.setText(f"Video: {len(self._videos)}")

        selected_row = 0
        for row_idx, video in enumerate(self._videos):
            name_item = QTableWidgetItem(str(video.get("file_name", "")))
            name_item.setData(Qt.UserRole, video.get("video_key", ""))
            self._video_table.setItem(row_idx, 0, name_item)
            self._video_table.setItem(row_idx, 1, QTableWidgetItem(str(video.get("started_at", ""))))
            self._video_table.setItem(row_idx, 2, QTableWidgetItem(str(video.get("finished_at", ""))))
            self._video_table.setItem(row_idx, 3, QTableWidgetItem(str(video.get("violation_count", 0))))

            if selected_before and video.get("video_key") == selected_before:
                selected_row = row_idx

        if not self._videos:
            self._selected_video_key = None
            self._video_meta_label.setText("Không có dữ liệu trong database với bộ lọc hiện tại.")
            self._violation_table.setRowCount(0)
            self._violation_count_label.setText("Tổng vi phạm: 0")
            self._status_label.setText("Không có video phù hợp.")
            return

        # Force sync detail panel even when Qt does not emit selection-changed
        # for a re-selected row that stays the same.
        self._video_table.blockSignals(True)
        self._video_table.selectRow(selected_row)
        self._video_table.blockSignals(False)
        self._on_video_selection_changed()
        self._status_label.setText(f"Đã tải {len(self._videos)} video từ database.")

    def _on_video_selection_changed(self):
        items = self._video_table.selectedItems()
        if not items:
            return

        row = items[0].row()
        if row < 0 or row >= len(self._videos):
            return

        video = self._videos[row]
        self._selected_video_key = str(video.get("video_key", ""))
        self._video_meta_label.setText(
            " | ".join(
                [
                    f"Video: {video.get('file_name', '')}",
                    f"FPS: {video.get('fps', 0)}",
                    f"Frames: {video.get('total_frames', 0)}",
                    f"Resolution: {video.get('width', 0)}x{video.get('height', 0)}",
                    f"Bắt đầu: {video.get('started_at', '')}",
                    f"Kết thúc: {video.get('finished_at', '')}",
                ]
            )
        )
        self._refresh_violations_table()

    def _refresh_violations_table(self):
        if not self._selected_video_key:
            self._violation_table.setRowCount(0)
            self._violation_count_label.setText("Tổng vi phạm: 0")
            return

        violations = self._store.get_violations_by_video(
            self._selected_video_key,
            violation_type=self._violation_type_combo.currentText(),
        )
        self._violation_table.setRowCount(len(violations))
        self._violation_count_label.setText(f"Tổng vi phạm: {len(violations)}")

        for row_idx, violation in enumerate(violations):
            class_name = violation.get("class_name", "")
            class_id = violation.get("class_id", "")
            class_display = f"{class_name} ({class_id})"

            values = [
                violation.get("violation_id", ""),
                f"{float(violation.get('violation_time_sec', 0.0)):.2f}",
                violation.get("violation_type", ""),
                str(violation.get("tracker_id", "")),
                class_display,
                str(violation.get("frame_number", "")),
                f"{float(violation.get('confidence', 0.0)):.2f}",
            ]
            for col_idx, value in enumerate(values):
                self._violation_table.setItem(row_idx, col_idx, QTableWidgetItem(value))

    @staticmethod
    def _qdatetime_to_iso(value: QDateTime) -> str:
        utc = value.toUTC()
        return utc.toString("yyyy-MM-ddTHH:mm:ss'Z'")
