"""
生成式视频编解码示例代码
Generative Video Codec Example Code

这个脚本展示了如何使用CLIP和语言模型进行视频编解码和字幕生成。
This script demonstrates how to use CLIP and language models for video codec and caption generation.

功能 Features:
1. 视频帧提取 - Video frame extraction
2. 帧编码/解码 - Frame encoding/decoding  
3. CLIP特征提取 - CLIP feature extraction
4. 视频字幕生成 - Video caption generation
5. 视频重建 - Video reconstruction
"""

import os
import sys
import argparse
import torch
import numpy as np
import tempfile
from pathlib import Path

# 导入自定义模块
from video_utils import VideoProcessor, VideoFrameEncoder, create_sample_video
from video_caption import VideoCaptionGenerator, VideoCLIPEncoder


def example_1_video_frame_extraction():
    """
    示例1: 视频帧提取
    Example 1: Video Frame Extraction
    """
    print("\n" + "=" * 70)
    print("示例1: 视频帧提取 / Example 1: Video Frame Extraction")
    print("=" * 70)
    
    # 创建测试视频
    temp_dir = tempfile.gettempdir()
    test_video = os.path.join(temp_dir, "test_extraction.mp4")
    print(f"\n创建测试视频... / Creating test video...")
    create_sample_video(test_video, duration=5, fps=30)
    
    # 初始化视频处理器
    processor = VideoProcessor()
    
    # 获取视频信息
    print(f"\n视频信息 / Video Info:")
    info = processor.get_video_info(test_video)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 提取帧
    print(f"\n提取帧... / Extracting frames...")
    frames, metadata = processor.extract_frames(test_video, num_frames=10)
    
    print(f"\n提取的帧数 / Extracted frames: {len(frames)}")
    print(f"帧尺寸 / Frame shape: {frames[0].shape}")
    print(f"元数据 / Metadata: {metadata}")
    
    return frames, test_video


def example_2_frame_encoding_decoding(frames):
    """
    示例2: 帧编码和解码
    Example 2: Frame Encoding and Decoding
    """
    print("\n" + "=" * 70)
    print("示例2: 帧编码和解码 / Example 2: Frame Encoding and Decoding")
    print("=" * 70)
    
    # 初始化编码器
    encoder = VideoFrameEncoder(quality=85)
    
    # 编码帧
    print(f"\n编码 {len(frames)} 帧... / Encoding {len(frames)} frames...")
    encoded_frames = encoder.encode_frames(frames)
    
    # 计算压缩率
    original_size = sum(frame.nbytes for frame in frames)
    encoded_size = sum(len(data) for data in encoded_frames)
    compression_ratio = original_size / encoded_size
    
    print(f"\n原始大小 / Original size: {original_size / 1024:.2f} KB")
    print(f"编码后大小 / Encoded size: {encoded_size / 1024:.2f} KB")
    print(f"压缩率 / Compression ratio: {compression_ratio:.2f}x")
    
    # 解码帧
    print(f"\n解码帧... / Decoding frames...")
    decoded_frames = encoder.decode_frames(encoded_frames)
    
    print(f"解码的帧数 / Decoded frames: {len(decoded_frames)}")
    
    return decoded_frames


def example_3_clip_feature_extraction(frames):
    """
    示例3: CLIP特征提取
    Example 3: CLIP Feature Extraction
    """
    print("\n" + "=" * 70)
    print("示例3: CLIP特征提取 / Example 3: CLIP Feature Extraction")
    print("=" * 70)
    
    # 初始化CLIP编码器
    print("\n初始化CLIP编码器... / Initializing CLIP encoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_encoder = VideoCLIPEncoder("ViT-B/32", device=device)
    
    # 提取特征
    print(f"\n从 {len(frames)} 帧提取CLIP特征... / Extracting CLIP features from {len(frames)} frames...")
    features = clip_encoder.encode_frames(frames[:5])  # 只使用前5帧作为示例
    
    print(f"\n特征形状 / Feature shape: {features.shape}")
    print(f"特征维度 / Feature dimension: {features.shape[1]}")
    
    # 测试不同的聚合方法
    print(f"\n测试特征聚合方法 / Testing aggregation methods:")
    for method in ["mean", "max", "attention"]:
        aggregated = clip_encoder.aggregate_features(features, method=method)
        print(f"  {method}: {aggregated.shape}")
    
    return features


def example_4_video_caption_generation(video_path):
    """
    示例4: 视频字幕生成
    Example 4: Video Caption Generation
    """
    print("\n" + "=" * 70)
    print("示例4: 视频字幕生成 / Example 4: Video Caption Generation")
    print("=" * 70)
    
    try:
        # 初始化字幕生成器
        print("\n初始化字幕生成器... / Initializing caption generator...")
        generator = VideoCaptionGenerator(
            model_type="gpt2",
            clip_model_type="ViT-B/32",
            prefix_length=10
        )
        
        # 生成字幕
        print(f"\n从视频生成字幕... / Generating caption from video...")
        caption = generator.generate_caption_from_video(
            video_path,
            num_frames=8,
            aggregation_method="mean",
            max_length=30,
            temperature=0.8
        )
        
        print("\n" + "=" * 70)
        print("生成的字幕 / Generated Caption:")
        print(f"  \"{caption}\"")
        print("=" * 70)
        
        return caption
        
    except Exception as e:
        print(f"\n错误 / Error: {str(e)}")
        print("注意: 此示例需要GPT-2模型。如果遇到内存问题，请尝试减少帧数。")
        print("Note: This example requires GPT-2 model. If you encounter memory issues, try reducing the number of frames.")
        return None


def example_5_video_reconstruction(frames, output_path):
    """
    示例5: 视频重建
    Example 5: Video Reconstruction
    """
    print("\n" + "=" * 70)
    print("示例5: 视频重建 / Example 5: Video Reconstruction")
    print("=" * 70)
    
    processor = VideoProcessor()
    
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, "reconstructed_video.mp4")
    else:
        output_file = output_path
    
    print(f"\n重建视频到 / Reconstructing video to: {output_file}")
    
    processor.save_frames_as_video(
        frames,
        output_file,
        fps=10.0,
        codec='mp4v'
    )
    
    print(f"✓ 视频已保存 / Video saved successfully!")
    
    return output_file


def run_all_examples():
    """
    运行所有示例
    Run All Examples
    """
    print("\n" + "=" * 70)
    print("生成式视频编解码完整示例")
    print("Generative Video Codec - Complete Examples")
    print("=" * 70)
    
    # 示例1: 提取帧
    frames, video_path = example_1_video_frame_extraction()
    
    # 示例2: 编码解码
    decoded_frames = example_2_frame_encoding_decoding(frames[:5])
    
    # 示例3: CLIP特征提取
    features = example_3_clip_feature_extraction(frames)
    
    # 示例4: 视频字幕生成
    caption = example_4_video_caption_generation(video_path)
    
    # 示例5: 视频重建
    temp_dir = tempfile.gettempdir()
    reconstructed_video = example_5_video_reconstruction(frames[:10], os.path.join(temp_dir, "reconstructed.mp4"))
    
    print("\n" + "=" * 70)
    print("所有示例完成! / All examples completed!")
    print("=" * 70)
    print(f"\n生成的文件 / Generated files:")
    print(f"  - 原始视频 / Original video: {video_path}")
    print(f"  - 重建视频 / Reconstructed video: {reconstructed_video}")
    if caption:
        print(f"  - 字幕 / Caption: \"{caption}\"")
    print("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成式视频编解码示例 / Generative Video Codec Examples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法 / Example Usage:
  
  运行所有示例 / Run all examples:
    python video_codec_example.py --all
  
  只运行特定示例 / Run specific example:
    python video_codec_example.py --example 1
    python video_codec_example.py --example 4
  
  处理自定义视频 / Process custom video:
    python video_codec_example.py --video /path/to/video.mp4 --example 4
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='运行所有示例 / Run all examples'
    )
    
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='运行特定示例 (1-5) / Run specific example (1-5)'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='输入视频路径 / Input video path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出视频路径 / Output video path'
    )
    
    args = parser.parse_args()
    
    # 如果没有参数，运行所有示例
    if not any(vars(args).values()):
        args.all = True
    
    if args.all:
        run_all_examples()
    elif args.example:
        if args.example == 1:
            frames, video_path = example_1_video_frame_extraction()
        elif args.example == 2:
            frames, _ = example_1_video_frame_extraction()
            example_2_frame_encoding_decoding(frames[:5])
        elif args.example == 3:
            frames, _ = example_1_video_frame_extraction()
            example_3_clip_feature_extraction(frames)
        elif args.example == 4:
            if args.video and os.path.exists(args.video):
                video_path = args.video
            else:
                frames, video_path = example_1_video_frame_extraction()
            example_4_video_caption_generation(video_path)
        elif args.example == 5:
            frames, _ = example_1_video_frame_extraction()
            example_5_video_reconstruction(frames[:10], args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
