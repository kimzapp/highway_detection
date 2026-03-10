import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO .pt model"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version"
    )

    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input size"
    )

    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX graph"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda:0...)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    model = YOLO(args.model)

    # Export
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        device=args.device
    )


if __name__ == "__main__":
    main()