from process.video import VideoProcessor


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Object Tracking with YOLO and ByteTrack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input source options
    parser.add_argument("--source", type=str, default="video", choices=["video", "webcam", "rtsp", "images"],
                        help="Input source type")
    parser.add_argument("--input", type=str, default="input.mp4", help="Path to input (video file, rtsp url, or image folder)")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save the output")
    
    # Model options
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to the YOLO model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on ('cpu' or 'cuda')")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size for the model")
    
    # Tracker options
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for detection NMS")
    parser.add_argument("--track-activation-thres", type=float, default=0.4,
                        help="Confidence threshold to activate new tracks")
    parser.add_argument("--track-match-thres", type=float, default=0.7,
                        help="Minimum IoU threshold to match detections to existing tracks")
    parser.add_argument("--max-age", type=int, default=90, help="Maximum frames to keep track without detections")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="Class IDs to track (e.g., 2 5 7)")
    
    # Visualization options
    parser.add_argument("--show-boxes", action="store_true", default=True, help="Show bounding boxes")
    parser.add_argument("--show-labels", action="store_true", default=True, help="Show labels")
    parser.add_argument("--show-traces", action="store_true", default=True, help="Show tracking traces")
    parser.add_argument("--trace-length", type=int, default=50, help="Length of tracking traces")
    
    # Output options
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--display", action="store_true", help="Display video during processing")
    
    # Road zone options
    parser.add_argument("--select-zone", action="store_true", default=True, 
                        help="Pause at first frame to select valid road zone")
    parser.add_argument("--no-select-zone", dest="select_zone", action="store_false",
                        help="Skip road zone selection")
    
    # Bird's Eye View options
    parser.add_argument("--enable-bev", action="store_true", default=True,
                        help="Enable Bird's Eye View display")
    parser.add_argument("--no-bev", dest="enable_bev", action="store_false",
                        help="Disable Bird's Eye View display")
    parser.add_argument("--bev-width", type=int, default=400,
                        help="Width of Bird's Eye View panel")
    parser.add_argument("--bev-height", type=int, default=600,
                        help="Height of Bird's Eye View panel")
    parser.add_argument("--bev-method", type=str, default="ipm", choices=["ipm", "homography"],
                        help="BEV transform method: 'ipm' (Inverse Perspective Mapping) or 'homography'")
    parser.add_argument("--camera-height", type=float, default=1.5,
                        help="Camera height in meters (for IPM method)")
    
    # Performance optimization options
    parser.add_argument("--half", action="store_true", default=True,
                        help="Use FP16 half-precision inference (CUDA only, faster)")
    parser.add_argument("--no-half", dest="half", action="store_false",
                        help="Disable FP16 half-precision inference")
    parser.add_argument("--skip-bev-frames", type=int, default=0,
                        help="Skip BEV update every N frames (0=no skip, higher=faster but less smooth)")
    parser.add_argument("--skip-frames", type=int, default=2,
                        help="Skip detection/tracking every N frames (2 => process 1 frame then skip 2)")
    parser.add_argument("--render-hold-frames", type=int, default=2,
                        help="Hold recent overlays for N frames when skipped frame has no tracks (anti-flicker)")
    parser.add_argument("--violation-hold-frames", type=int, default=2,
                        help="Keep last violation labels for N frames when detections are briefly missing")
    parser.add_argument("--min-violation-frames", type=int, default=45,
                        help="Minimum consecutive frames outside valid zone before counting wrong-lane violation")
    parser.add_argument("--enable-invalid-vehicle", action="store_true", default=False,
                        help="Enable invalid-vehicle violation detection based on valid class list")
    parser.add_argument("--no-invalid-vehicle", dest="enable_invalid_vehicle_detection", action="store_false",
                        help="Disable invalid-vehicle violation detection")
    parser.add_argument("--valid-vehicle-classes", nargs="+", type=int, default=[2],
                        help="Class IDs considered valid vehicles for invalid-vehicle detection")
    
    return parser.parse_args()


def process_video_source(args):
    """Xử lý video file"""
    processor = VideoProcessor(args)
    
    # Set BEV options
    processor.enable_bev = args.enable_bev
    processor.bev_width = args.bev_width
    processor.bev_height = args.bev_height
    processor.bev_method = args.bev_method
    processor.camera_height = args.camera_height
    
    output_path = args.output if args.save_video else None
    
    processor.process_video(
        video_path=args.input,
        output_path=output_path,
        display=args.display,
        select_road_zone=args.select_zone
    )


def process_webcam_source(args):
    """Xử lý webcam (TODO: implement)"""
    print("Webcam processing not implemented yet")
    # TODO: Implement webcam processing
    pass


def process_rtsp_source(args):
    """Xử lý RTSP stream (TODO: implement)"""
    print("RTSP processing not implemented yet")
    # TODO: Implement RTSP streaming
    pass


def process_images_source(args):
    """Xử lý folder ảnh (TODO: implement)"""
    print("Images processing not implemented yet")
    # TODO: Implement image folder processing
    pass


def main():
    args = parse_args()
    
    # Route to appropriate processor based on source type
    source_handlers = {
        "video": process_video_source,
        "webcam": process_webcam_source,
        "rtsp": process_rtsp_source,
        "images": process_images_source
    }
    
    handler = source_handlers.get(args.source)
    if handler:
        handler(args)
    else:
        print(f"Unknown source type: {args.source}")


if __name__ == "__main__":
    main()