"""
Base Model Handler
Abstract base class định nghĩa interface chung cho tất cả model handlers
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class BaseModelHandler(ABC):
    """
    Abstract base class cho model handlers
    
    Attributes:
        model_path: Đường dẫn đến file model
        device: Device để chạy model ('cpu' hoặc 'cuda')
        names: Dict mapping class_id to class name
    """
    
    SUPPORTED_EXTENSIONS: List[str] = []
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Khởi tạo model handler
        
        Args:
            model_path: Đường dẫn đến file model
            device: Device để chạy model
        """
        self.model_path = model_path
        self.device = device
        self.names: Dict[int, str] = {}
        self._model = None
        
    @abstractmethod
    def load(self) -> bool:
        """
        Load model từ file
        
        Returns:
            True nếu load thành công, False nếu thất bại
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        imgsz: int = 640,
        **kwargs
    ) -> Any:
        """
        Chạy inference trên image
        
        Args:
            image: Input image (BGR format)
            conf: Confidence threshold
            iou: IoU threshold cho NMS
            classes: List class IDs để filter (None = tất cả)
            imgsz: Input image size
            **kwargs: Các tham số bổ sung
            
        Returns:
            Detection results
        """
        pass
    
    @abstractmethod
    def get_detections(
        self,
        results: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trích xuất detections từ results
        
        Args:
            results: Raw results từ predict()
            
        Returns:
            Tuple (boxes, scores, class_ids):
                - boxes: np.ndarray shape (N, 4) với format [x1, y1, x2, y2]
                - scores: np.ndarray shape (N,) confidence scores
                - class_ids: np.ndarray shape (N,) class IDs
        """
        pass
    
    def to(self, device: str):
        """
        Chuyển model sang device khác
        
        Args:
            device: Target device ('cpu' hoặc 'cuda')
        """
        self.device = device
    
    @property
    def model(self):
        """Trả về model đã load"""
        return self._model
    
    @classmethod
    def supports_format(cls, model_path: str) -> bool:
        """
        Kiểm tra xem handler có hỗ trợ định dạng file này không
        
        Args:
            model_path: Đường dẫn đến file model
            
        Returns:
            True nếu hỗ trợ, False nếu không
        """
        ext = model_path.lower().split('.')[-1]
        return f".{ext}" in cls.SUPPORTED_EXTENSIONS
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_path='{self.model_path}', device='{self.device}')"
