"""
PyTorch Model Handler
Handler cho models định dạng .pt (Ultralytics YOLO)
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base import BaseModelHandler


class PTModelHandler(BaseModelHandler):
    """
    Handler cho PyTorch/Ultralytics YOLO models (.pt)
    
    Sử dụng trực tiếp ultralytics package để load và inference
    """
    
    SUPPORTED_EXTENSIONS = [".pt", ".pth"]
    
    def __init__(self, model_path: str, device: str = "cpu"):
        super().__init__(model_path, device)
        self._ultralytics_model = None
        
    def load(self) -> bool:
        """Load YOLO model từ file .pt"""
        try:
            from ultralytics import YOLO
            
            self._ultralytics_model = YOLO(self.model_path)
            self._ultralytics_model.to(self.device)
            self._model = self._ultralytics_model
            
            # Lấy class names
            self.names = self._ultralytics_model.names
            
            return True
            
        except Exception as e:
            print(f"Error loading PT model: {e}")
            return False
    
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
        Chạy inference với Ultralytics YOLO
        
        Returns:
            Ultralytics Results object
        """
        if self._ultralytics_model is None:
            raise RuntimeError("Model chưa được load. Gọi load() trước.")
        
        results = self._ultralytics_model(
            image,
            conf=conf,
            iou=iou,
            classes=classes,
            imgsz=imgsz,
            verbose=False,
            **kwargs
        )
        
        return results
    
    def get_detections(
        self,
        results: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trích xuất detections từ Ultralytics Results
        
        Args:
            results: Ultralytics Results object (list)
            
        Returns:
            Tuple (boxes, scores, class_ids)
        """
        # Ultralytics trả về list of Results
        if isinstance(results, list):
            results = results[0]
        
        boxes = results.boxes
        
        if boxes is None or len(boxes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32)
            )
        
        # Lấy xyxy boxes, confidence, class_ids
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(np.int32)
        
        return xyxy, conf, cls
    
    def to(self, device: str):
        """Chuyển model sang device khác"""
        super().to(device)
        if self._ultralytics_model is not None:
            self._ultralytics_model.to(device)
    
    @property
    def ultralytics_model(self):
        """Trả về Ultralytics YOLO model gốc để dùng các chức năng đặc biệt"""
        return self._ultralytics_model
