from tracking.bytetrack import ByteTracker
from ultralytics import YOLO


def argparse():
    import argparse

    parser = argparse.ArgumentParser(description="Object Tracking with YOLO and ByteTrack", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to the YOLO model")
    parser.add_argument("--video", type=str, default="input.mp4", help="Path to the input video")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save the output video")
    args = parser.parse_args()
    return args


def main():
    args = argparse()
    pass