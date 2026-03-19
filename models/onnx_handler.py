"""
ONNX Model Handler
Handler cho models định dạng .onnx
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import sys
import traceback
import numpy as np
import cv2

from .base import BaseModelHandler


# Keep add_dll_directory handles alive for process lifetime.
_DLL_DIRECTORY_HANDLES = []


def _prepare_onnxruntime_dll_paths() -> List[str]:
    """Register likely ONNX Runtime DLL folders for frozen Windows builds."""
    if os.name != 'nt':
        return []

    candidates = []

    # PyInstaller one-dir executable location.
    exe_dir = os.path.dirname(getattr(sys, 'executable', '') or '')
    if exe_dir:
        candidates.extend([
            exe_dir,
            os.path.join(exe_dir, '_internal'),
            os.path.join(exe_dir, '_internal', 'onnxruntime', 'capi'),
        ])

    # PyInstaller one-file extraction location.
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        candidates.extend([
            meipass,
            os.path.join(meipass, 'onnxruntime', 'capi'),
            os.path.join(meipass, '_internal'),
            os.path.join(meipass, '_internal', 'onnxruntime', 'capi'),
        ])

    existing_dirs = []
    seen = set()
    for d in candidates:
        norm = os.path.normpath(d)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isdir(norm):
            existing_dirs.append(norm)

    if hasattr(os, 'add_dll_directory'):
        for dll_dir in existing_dirs:
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(dll_dir))
            except OSError:
                pass

    current_path = os.environ.get('PATH', '')
    path_parts = [p for p in current_path.split(os.pathsep) if p]
    for dll_dir in reversed(existing_dirs):
        if dll_dir not in path_parts:
            path_parts.insert(0, dll_dir)
    os.environ['PATH'] = os.pathsep.join(path_parts)

    return existing_dirs


class ONNXModelHandler(BaseModelHandler):
    """
    Handler cho ONNX models (.onnx)
    
    Sử dụng onnxruntime để inference
    Hỗ trợ cả CPU và GPU (CUDA)
    """
    
    SUPPORTED_EXTENSIONS = [".onnx"]
    
    # COCO class names mặc định (có thể override qua metadata)
    DEFAULT_COCO_NAMES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
        54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
        59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
        79: 'toothbrush'
    }
    
    def __init__(self, model_path: str, device: str = "cpu"):
        super().__init__(model_path, device)
        self._session = None
        self._input_name = None
        self._output_names = None
        self._input_shape = None
        
    def load(self) -> bool:
        """Load ONNX model"""
        try:
            prepared_dirs = _prepare_onnxruntime_dll_paths()
            import onnxruntime as ort

            if prepared_dirs:
                print("ONNX runtime DLL search paths prepared:")
                for d in prepared_dirs:
                    print(f"  - {d}")
            
            # Chọn provider dựa trên device
            if "cuda" in self.device.lower():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            # Tạo session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4 # Tăng số thread để cải thiện hiệu năng trên CPU
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            
            
            self._session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Lấy input/output info
            model_inputs = self._session.get_inputs()
            model_outputs = self._session.get_outputs()
            
            self._input_name = model_inputs[0].name
            self._input_shape = model_inputs[0].shape
            self._output_names = [o.name for o in model_outputs]
            
            # Lấy class names từ metadata hoặc dùng default
            self._load_class_names()
            
            self._model = self._session
            
            print(f"ONNX model loaded successfully")
            print(f"  Input: {self._input_name} {self._input_shape}")
            print(f"  Outputs: {self._output_names}")
            print(f"  Provider: {self._session.get_providers()}")
            
            return True
            
        except ImportError as e:
            # ImportError trong môi trường frozen thường là lỗi thiếu DLL, không hẳn thiếu package.
            print("Error: Không thể import onnxruntime.")
            print(f"  ImportError: {e}")
            if e.__cause__ is not None:
                print(f"  Cause: {repr(e.__cause__)}")
            if e.__context__ is not None:
                print(f"  Context: {repr(e.__context__)}")

            traceback_text = traceback.format_exc().strip()
            if traceback_text:
                print("  Traceback:")
                print(traceback_text)

            print(
                "  Gợi ý: nếu traceback có 'DLL load failed' hoặc 'specified module could not be found', "
                "đây là lỗi thiếu DLL/phụ thuộc runtime, không phải chưa cài pip package."
            )
            return False
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            return False
    
    def _load_class_names(self):
        """Load class names từ model metadata hoặc dùng default"""
        try:
            metadata = self._session.get_modelmeta()
            custom_metadata = metadata.custom_metadata_map
            
            if 'names' in custom_metadata:
                # Parse names từ metadata (format: "{0: 'class0', 1: 'class1', ...}")
                import ast
                self.names = ast.literal_eval(custom_metadata['names'])
            else:
                self.names = self.DEFAULT_COCO_NAMES.copy()
        except Exception:
            self.names = self.DEFAULT_COCO_NAMES.copy()
    
    def _preprocess(
        self, 
        image: np.ndarray, 
        imgsz: int = 640
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image cho ONNX inference
        
        Args:
            image: Input image (BGR)
            imgsz: Target size
            
        Returns:
            Tuple (preprocessed_image, ratio, (pad_w, pad_h))
        """
        # Letterbox resize (giữ aspect ratio)
        h, w = image.shape[:2]
        ratio = min(imgsz / h, imgsz / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Padding
        pad_h = imgsz - new_h
        pad_w = imgsz - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # BGR to RGB, normalize, transpose
        img = padded[:, :, ::-1]  # BGR to RGB
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.ascontiguousarray(img)
        
        return img, ratio, (left, top)
    
    def _postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float,
        pad: Tuple[int, int],
        original_shape: Tuple[int, int],
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess ONNX outputs
        
        Args:
            outputs: Raw ONNX outputs
            ratio: Resize ratio
            pad: (pad_w, pad_h) padding
            original_shape: (H, W) original image shape
            conf: Confidence threshold
            iou: IoU threshold cho NMS
            classes: Class filter
            
        Returns:
            Tuple (boxes, scores, class_ids)
        """
        # YOLOv8 output shape: (1, num_classes + 4, num_anchors)
        # Transpose to (1, num_anchors, num_classes + 4)
        output = outputs[0]
        
        if output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)
        
        output = output[0]  # Remove batch dimension
        
        # Split boxes and scores
        boxes = output[:, :4]  # x_center, y_center, width, height
        scores = output[:, 4:]  # class scores
        
        # Lấy class có score cao nhất
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences >= conf
        
        # Filter by classes nếu được chỉ định
        if classes is not None:
            class_mask = np.isin(class_ids, classes)
            mask = mask & class_mask
        
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32)
            )
        
        # Convert xywh to xyxy
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Scale boxes về original image
        pad_w, pad_h = pad
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio
        
        # Clip to image bounds
        h, w = original_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
        
        # NMS
        indices = self._nms(boxes, confidences, iou)
        
        return boxes[indices], confidences[indices], class_ids[indices]
    
    def _nms(
        self, 
        boxes: np.ndarray, 
        scores: np.ndarray, 
        iou_threshold: float
    ) -> np.ndarray:
        """
        Non-Maximum Suppression
        
        Args:
            boxes: (N, 4) boxes [x1, y1, x2, y2]
            scores: (N,) confidence scores
            iou_threshold: IoU threshold
            
        Returns:
            Indices of kept boxes
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Compute IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU < threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep)
    
    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        imgsz: int = 640,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chạy inference với ONNX Runtime
        
        Returns:
            Dict chứa raw outputs và processed detections
        """
        if self._session is None:
            raise RuntimeError("Model chưa được load. Gọi load() trước.")
        
        original_shape = image.shape[:2]
        
        # Preprocess
        input_tensor, ratio, pad = self._preprocess(image, imgsz)
        
        # Inference
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})
        
        # Postprocess
        boxes, scores, class_ids = self._postprocess(
            outputs, ratio, pad, original_shape, conf, iou, classes
        )
        
        return {
            'raw_outputs': outputs,
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'original_shape': original_shape
        }
    
    def get_detections(
        self,
        results: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trích xuất detections từ results
        
        Args:
            results: Dict từ predict()
            
        Returns:
            Tuple (boxes, scores, class_ids)
        """
        if isinstance(results, dict):
            return results['boxes'], results['scores'], results['class_ids']
        
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32)
        )
    
    def to(self, device: str):
        """
        Chuyển model sang device khác
        
        Note: ONNX Runtime cần reload session với provider khác
        """
        if device != self.device:
            self.device = device
            # Reload với provider mới
            if self._session is not None:
                self.load()
