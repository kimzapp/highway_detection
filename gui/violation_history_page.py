"""History page for persisted violation records."""

from __future__ import annotations

from typing import Any, Dict, List

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ViolationHistoryPage(QWidget):
    """Simple page to inspect saved violations for the processed video."""

    back_to_home_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_violations: List[Dict[str, Any]] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Lịch Sử Sai Phạm")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self._video_meta_label = QLabel("Chưa có dữ liệu.")
        self._video_meta_label.setWordWrap(True)
        self._video_meta_label.setStyleSheet("color: #444; padding: 6px;")
        layout.addWidget(self._video_meta_label)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Lọc loại vi phạm:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["Tất cả", "WRONG_LANE", "INVALID_VEHICLE"])
        self._filter_combo.currentTextChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_combo)

        filter_row.addStretch()

        self._stats_label = QLabel("Tổng: 0")
        self._stats_label.setStyleSheet("font-weight: bold;")
        filter_row.addWidget(self._stats_label)
        layout.addLayout(filter_row)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            [
                "Violation ID",
                "Time (s)",
                "Type",
                "Tracker",
                "Class",
                "Frame",
                "Confidence",
            ]
        )
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self._back_home_btn = QPushButton("Xử Lý Video Khác")
        self._back_home_btn.clicked.connect(self.back_to_home_requested.emit)
        button_row.addWidget(self._back_home_btn)
        layout.addLayout(button_row)

    def set_data(self, video_data: Dict[str, Any], violations: List[Dict[str, Any]]):
        self._all_violations = list(violations or [])

        if video_data:
            self._video_meta_label.setText(
                " | ".join(
                    [
                        f"Video: {video_data.get('file_name', '')}",
                        f"FPS: {video_data.get('fps', 0)}",
                        f"Frames: {video_data.get('total_frames', 0)}",
                        f"Resolution: {video_data.get('width', 0)}x{video_data.get('height', 0)}",
                        f"Bắt đầu: {video_data.get('started_at', '')}",
                        f"Kết thúc: {video_data.get('finished_at', '')}",
                    ]
                )
            )
        else:
            self._video_meta_label.setText("Không tìm thấy metadata video.")

        self._apply_filter()

    def _apply_filter(self):
        selected = self._filter_combo.currentText()
        if selected == "Tất cả":
            filtered = self._all_violations
        else:
            filtered = [v for v in self._all_violations if v.get("violation_type") == selected]

        self._stats_label.setText(f"Tổng: {len(filtered)}")
        self._table.setRowCount(len(filtered))

        for row_idx, violation in enumerate(filtered):
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
                self._table.setItem(row_idx, col_idx, QTableWidgetItem(value))
