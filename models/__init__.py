"""
Models Module
Hỗ trợ nhiều định dạng model cho object detection

Supported formats:
- .pt, .pth: PyTorch/Ultralytics YOLO models
- .onnx: ONNX models

Usage:
    from models import load_model
    
    # Load model (auto-detect format)
    handler = load_model("model.pt", device="cuda")
    
    # Or specify handler directly
    from models import PTModelHandler, ONNXModelHandler
    
    pt_handler = PTModelHandler("model.pt", device="cpu")
    pt_handler.load()
    
    onnx_handler = ONNXModelHandler("model.onnx", device="cuda")
    onnx_handler.load()
    
    # Run inference
    results = handler.predict(image, conf=0.25, iou=0.45)
    boxes, scores, class_ids = handler.get_detections(results)
"""

from .base import BaseModelHandler
from .pt_handler import PTModelHandler
from .onnx_handler import ONNXModelHandler
from .loader import (
    load_model,
    create_handler,
    get_supported_formats,
    get_handler_for_format,
    HANDLERS
)


__all__ = [
    # Base
    'BaseModelHandler',
    
    # Handlers
    'PTModelHandler',
    'ONNXModelHandler',
    
    # Factory functions
    'load_model',
    'create_handler',
    'get_supported_formats',
    'get_handler_for_format',
    'HANDLERS',
]
