"""
Video processing utilities for frame extraction and video handling.
支持视频编解码的工具函数
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
import tempfile
from pathlib import Path


class VideoProcessor:
    """视频处理器 - 用于视频帧提取和重建"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    def extract_frames(
        self, 
        video_path: str, 
        num_frames: Optional[int] = None,
        fps: Optional[float] = None,
        max_frames: int = 100
    ) -> Tuple[List[np.ndarray], dict]:
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            num_frames: 要提取的帧数，如果为None则提取所有帧
            fps: 提取帧的帧率，如果为None则使用视频原始帧率
            max_frames: 最大帧数限制
            
        Returns:
            frames: 帧列表 (numpy arrays)
            metadata: 视频元数据
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # 获取视频元数据
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata = {
            'total_frames': total_frames,
            'fps': video_fps,
            'width': width,
            'height': height,
            'duration': total_frames / video_fps if video_fps > 0 else 0
        }
        
        frames = []
        
        # 确定采样策略
        if num_frames is not None:
            # 均匀采样指定数量的帧
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        elif fps is not None:
            # 按指定帧率采样
            interval = int(video_fps / fps) if fps < video_fps else 1
            frame_indices = range(0, total_frames, interval)
        else:
            # 提取所有帧，但限制最大数量
            frame_indices = range(0, min(total_frames, max_frames))
        
        # 提取帧
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        
        metadata['extracted_frames'] = len(frames)
        return frames, metadata
    
    def save_frames_as_video(
        self, 
        frames: List[np.ndarray], 
        output_path: str,
        fps: float = 30.0,
        codec: str = 'mp4v'
    ):
        """
        将帧序列保存为视频文件
        
        Args:
            frames: 帧列表
            output_path: 输出视频路径
            fps: 帧率
            codec: 视频编码器
        """
        if not frames:
            raise ValueError("No frames to save")
        
        # 获取帧尺寸
        height, width = frames[0].shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # 确保是BGR格式
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved to: {output_path}")
    
    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息字典
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        
        info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
        
        cap.release()
        return info


class VideoFrameEncoder:
    """视频帧编码器 - 简单的帧压缩"""
    
    def __init__(self, quality: int = 95):
        """
        Args:
            quality: JPEG压缩质量 (0-100)
        """
        self.quality = quality
    
    def encode_frame(self, frame: np.ndarray) -> bytes:
        """
        编码单个帧为JPEG字节
        
        Args:
            frame: 帧数组 (RGB或BGR)
            
        Returns:
            编码后的字节数据
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        return encoded.tobytes()
    
    def decode_frame(self, data: bytes) -> np.ndarray:
        """
        解码JPEG字节为帧
        
        Args:
            data: 编码的字节数据
            
        Returns:
            解码后的帧数组
        """
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    
    def encode_frames(self, frames: List[np.ndarray]) -> List[bytes]:
        """批量编码帧"""
        return [self.encode_frame(frame) for frame in frames]
    
    def decode_frames(self, encoded_frames: List[bytes]) -> List[np.ndarray]:
        """批量解码帧"""
        return [self.decode_frame(data) for data in encoded_frames]


def create_sample_video(output_path: str, duration: int = 5, fps: int = 30):
    """
    创建示例视频用于测试
    
    Args:
        output_path: 输出视频路径
        duration: 视频时长（秒）
        fps: 帧率
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for i in range(total_frames):
        # 创建渐变色帧
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加颜色渐变效果
        color_value = int(255 * (i / total_frames))
        frame[:, :] = [color_value, 128, 255 - color_value]
        
        # 添加文字
        text = f"Frame {i+1}/{total_frames}"
        cv2.putText(frame, text, (50, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Sample video created: {output_path}")


if __name__ == "__main__":
    # 测试代码
    print("Video utilities module loaded successfully!")
    
    # 创建测试视频
    temp_dir = tempfile.gettempdir()
    test_video_path = os.path.join(temp_dir, "test_video.mp4")
    create_sample_video(test_video_path, duration=3, fps=10)
    
    # 测试视频处理
    processor = VideoProcessor()
    info = processor.get_video_info(test_video_path)
    print(f"\nVideo info: {info}")
    
    # 提取帧
    frames, metadata = processor.extract_frames(test_video_path, num_frames=5)
    print(f"\nExtracted {len(frames)} frames")
    print(f"Metadata: {metadata}")
