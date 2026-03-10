"""
Model Loader Factory
Factory pattern để tạo model handler phù hợp với định dạng file
"""

import os
from typing import Optional, Type, List

from .base import BaseModelHandler
from .pt_handler import PTModelHandler
from .onnx_handler import ONNXModelHandler


# Registry các handlers được hỗ trợ
HANDLERS: List[Type[BaseModelHandler]] = [
    PTModelHandler,
    ONNXModelHandler,
]


def get_supported_formats() -> List[str]:
    """
    Lấy danh sách các định dạng được hỗ trợ
    
    Returns:
        List các extensions được hỗ trợ (vd: ['.pt', '.pth', '.onnx'])
    """
    formats = []
    for handler_cls in HANDLERS:
        formats.extend(handler_cls.SUPPORTED_EXTENSIONS)
    return formats


def get_handler_for_format(model_path: str) -> Optional[Type[BaseModelHandler]]:
    """
    Tìm handler class phù hợp cho file model
    
    Args:
        model_path: Đường dẫn đến file model
        
    Returns:
        Handler class hoặc None nếu không hỗ trợ
    """
    for handler_cls in HANDLERS:
        if handler_cls.supports_format(model_path):
            return handler_cls
    return None


def load_model(
    model_path: str,
    device: str = "cpu",
    auto_load: bool = True
) -> BaseModelHandler:
    """
    Factory function để load model từ file
    
    Tự động detect định dạng và sử dụng handler phù hợp.
    
    Args:
        model_path: Đường dẫn đến file model
        device: Device để chạy model ('cpu' hoặc 'cuda')
        auto_load: Tự động gọi load() sau khi tạo handler
        
    Returns:
        Model handler instance
        
    Raises:
        ValueError: Nếu định dạng không được hỗ trợ
        FileNotFoundError: Nếu file không tồn tại
        RuntimeError: Nếu load model thất bại
    """
    # Kiểm tra file tồn tại
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file không tồn tại: {model_path}")
    
    # Tìm handler phù hợp
    handler_cls = get_handler_for_format(model_path)
    
    if handler_cls is None:
        ext = os.path.splitext(model_path)[1]
        supported = get_supported_formats()
        raise ValueError(
            f"Định dạng '{ext}' không được hỗ trợ. "
            f"Các định dạng hỗ trợ: {supported}"
        )
    
    # Tạo handler instance
    handler = handler_cls(model_path, device)
    
    # Load model nếu auto_load
    if auto_load:
        success = handler.load()
        if not success:
            raise RuntimeError(f"Không thể load model từ: {model_path}")
    
    return handler


def create_handler(
    model_path: str,
    device: str = "cpu"
) -> BaseModelHandler:
    """
    Alias cho load_model() với auto_load=False
    
    Tạo handler instance mà không load model ngay.
    Hữu ích khi cần configure handler trước khi load.
    
    Args:
        model_path: Đường dẫn đến file model
        device: Device để chạy model
        
    Returns:
        Model handler instance (chưa load)
    """
    return load_model(model_path, device, auto_load=False)
