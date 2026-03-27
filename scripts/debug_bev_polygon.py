"""
Simple BEV debug script.

Features:
- Select polygon zone on the first frame (using existing RoadZoneSelector UI).
- Build BirdEyeViewTransformer from selected polygon.
- Visualize mapping on each frame.
- Click points on source view to see transformed points in BEV.

Usage:
    python scripts/debug_bev_polygon.py --source path/to/video.mp4
    python scripts/debug_bev_polygon.py --source 0
"""

import argparse
from typing import List, Tuple, Union

import cv2
import numpy as np

from lane_mapping.bird_eye_view import BirdEyeViewTransformer, BirdEyeViewVisualizer, create_combined_view
from lane_mapping.road_zone import RoadZoneSelector


APP_LIKE_WINDOW = "ByteTrack - Object Tracking (with Bird's Eye View)"


def parse_source(source: str) -> Union[int, str]:
    if source.isdigit():
        return int(source)
    return source


def draw_source_overlay(frame: np.ndarray, polygon: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
    output = frame.copy()

    if polygon is not None and len(polygon) >= 3:
        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon.astype(np.int32)], (0, 180, 0))
        output = cv2.addWeighted(overlay, 0.2, output, 0.8, 0)
        cv2.polylines(output, [polygon.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)

    for i, pt in enumerate(points):
        cv2.circle(output, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            output,
            str(i + 1),
            (pt[0] + 8, pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    help_text = "L-Click: add point | C: clear points | Space: pause | Q/Esc: quit"
    cv2.rectangle(output, (8, 8), (700, 36), (20, 20, 20), -1)
    cv2.putText(output, help_text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    return output


def draw_bev_debug_points(
    bev_frame: np.ndarray,
    transformer: BirdEyeViewTransformer,
    points: List[Tuple[int, int]],
) -> np.ndarray:
    bev = bev_frame.copy()

    for i, pt in enumerate(points):
        bev_pt = transformer.transform_point(pt)
        if 0 <= bev_pt[0] < transformer.bev_width and 0 <= bev_pt[1] < transformer.bev_height:
            cv2.circle(bev, bev_pt, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                bev,
                str(i + 1),
                (bev_pt[0] + 8, bev_pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    status = "BEV style: APP | Click points to inspect mapping"
    cv2.rectangle(bev, (8, 8), (500, 36), (20, 20, 20), -1)
    cv2.putText(bev, status, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    return bev


def main():
    parser = argparse.ArgumentParser(description="Select polygon zone and debug BEV mapping.")
    parser.add_argument("--source", default="0", help="Video path or camera index. Example: 0 or video.mp4")
    parser.add_argument("--bev-width", type=int, default=400, help="BEV canvas width")
    parser.add_argument("--bev-height", type=int, default=600, help="BEV canvas height")
    parser.add_argument("--margin", type=int, default=50, help="BEV margin")
    parser.add_argument("--no-suggestion", action="store_true", help="Disable lane suggestion when selecting polygon")
    args = parser.parse_args()

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}")
        return

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        print("[ERROR] Cannot read first frame from source.")
        cap.release()
        return

    selector = RoadZoneSelector(enable_suggestion=not args.no_suggestion)
    zone_polygon = selector.select_zone(first_frame)
    if zone_polygon is None or len(zone_polygon) < 4:
        print("[INFO] Selection cancelled or insufficient points (need at least 4 points).")
        cap.release()
        return

    transformer = BirdEyeViewTransformer(
        source_polygon=zone_polygon,
        bev_width=args.bev_width,
        bev_height=args.bev_height,
        margin=args.margin,
    )
    visualizer = BirdEyeViewVisualizer(transformer=transformer, show_zones=True)

    clicked_points: List[Tuple[int, int]] = []
    paused = False
    current_frame = first_frame.copy()

    cam_width = int(first_frame.shape[1])

    def on_mouse(event, x, y, flags, param):
        del flags, param
        # Combined view layout="horizontal": camera frame ở bên trái.
        if event == cv2.EVENT_LBUTTONDOWN and x < cam_width:
            clicked_points.append((x, y))

    cv2.namedWindow(APP_LIKE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(APP_LIKE_WINDOW, on_mouse)

    while True:
        if not paused:
            ok, frame = cap.read()
            if ok and frame is not None:
                current_frame = frame
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                current_frame = frame

        src_view = draw_source_overlay(current_frame, zone_polygon, clicked_points)

        # Render BEV bằng đúng flow như app hiện tại.
        bev_frame = visualizer.draw(
            detections=None,
            class_names=None,
            show_ids=True,
            show_labels=True,
            current_violations=None,
        )
        bev_view = draw_bev_debug_points(
            bev_frame=bev_frame,
            transformer=transformer,
            points=clicked_points,
        )
        display_frame = create_combined_view(
            camera_frame=src_view,
            bev_frame=bev_view,
            layout="horizontal",
        )

        cv2.imshow(APP_LIKE_WINDOW, display_frame)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key in (ord("c"), ord("C")):
            clicked_points.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
