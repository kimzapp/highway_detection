"""
FPS Counter Module
Tính toán và theo dõi FPS xử lý
"""

import time
from collections import deque
from typing import Optional


class FPSCounter:
    """
    FPS Counter có thể tái sử dụng cho video processing.
    
    Hỗ trợ:
    - Tính FPS trung bình theo sliding window
    - Tính FPS tức thời
    - Reset counter
    - Callback khi có FPS mới (cho GUI updates)
    
    Example:
        fps_counter = FPSCounter(window_size=30)
        
        while processing:
            fps_counter.tick()
            current_fps = fps_counter.fps
            avg_fps = fps_counter.avg_fps
    """
    
    def __init__(self, window_size: int = 30):
        """
        Khởi tạo FPS Counter
        
        Args:
            window_size: Số frame để tính FPS trung bình (default: 30)
        """
        self._window_size = window_size
        self._timestamps: deque = deque(maxlen=window_size)
        self._last_tick: Optional[float] = None
        self._instant_fps: float = 0.0
        self._frame_count: int = 0
        self._start_time: Optional[float] = None
    
    def tick(self) -> float:
        """
        Gọi mỗi khi xử lý xong một frame.
        
        Returns:
            FPS tức thời
        """
        current_time = time.perf_counter()
        
        # Khởi tạo start time nếu chưa có
        if self._start_time is None:
            self._start_time = current_time
        
        # Tính FPS tức thời
        if self._last_tick is not None:
            delta = current_time - self._last_tick
            if delta > 0:
                self._instant_fps = 1.0 / delta
        
        self._last_tick = current_time
        self._timestamps.append(current_time)
        self._frame_count += 1
        
        return self._instant_fps
    
    @property
    def fps(self) -> float:
        """FPS tức thời (frame gần nhất)"""
        return self._instant_fps
    
    @property
    def avg_fps(self) -> float:
        """FPS trung bình theo sliding window"""
        if len(self._timestamps) < 2:
            return 0.0
        
        time_span = self._timestamps[-1] - self._timestamps[0]
        if time_span <= 0:
            return 0.0
        
        return (len(self._timestamps) - 1) / time_span
    
    @property
    def overall_fps(self) -> float:
        """FPS tổng thể từ đầu đến hiện tại"""
        if self._start_time is None or self._frame_count < 2:
            return 0.0
        
        elapsed = time.perf_counter() - self._start_time
        if elapsed <= 0:
            return 0.0
        
        return self._frame_count / elapsed
    
    @property
    def frame_count(self) -> int:
        """Tổng số frame đã xử lý"""
        return self._frame_count
    
    @property
    def elapsed_time(self) -> float:
        """Thời gian đã trôi qua (giây)"""
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time
    
    def reset(self):
        """Reset counter về trạng thái ban đầu"""
        self._timestamps.clear()
        self._last_tick = None
        self._instant_fps = 0.0
        self._frame_count = 0
        self._start_time = None
    
    def get_stats(self) -> dict:
        """
        Lấy tất cả thống kê FPS
        
        Returns:
            Dict với các thông số:
            - fps: FPS tức thời
            - avg_fps: FPS trung bình (sliding window)
            - overall_fps: FPS tổng thể
            - frame_count: Số frame đã xử lý
            - elapsed_time: Thời gian đã chạy (giây)
        """
        return {
            'fps': self._instant_fps,
            'avg_fps': self.avg_fps,
            'overall_fps': self.overall_fps,
            'frame_count': self._frame_count,
            'elapsed_time': self.elapsed_time
        }
    
    def __str__(self) -> str:
        return f"FPS: {self._instant_fps:.1f} (avg: {self.avg_fps:.1f})"
    
    def __repr__(self) -> str:
        return f"FPSCounter(fps={self._instant_fps:.1f}, avg={self.avg_fps:.1f}, frames={self._frame_count})"
