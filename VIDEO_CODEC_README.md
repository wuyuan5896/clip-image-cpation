# 生成式视频编解码 (Generative Video Codec)

这是基于CLIP和语言模型的视频编解码和字幕生成系统。

This is a video codec and caption generation system based on CLIP and language models.

## 功能特性 (Features)

### 1. 视频处理 (Video Processing)
- ✅ 视频帧提取 - Extract frames from videos
- ✅ 多种采样策略 (均匀采样、按帧率采样) - Multiple sampling strategies
- ✅ 视频元数据获取 - Get video metadata
- ✅ 视频重建 - Reconstruct videos from frames

### 2. 帧编解码 (Frame Encoding/Decoding)
- ✅ JPEG压缩编码 - JPEG compression encoding
- ✅ 可配置的压缩质量 - Configurable compression quality
- ✅ 批量编解码 - Batch encoding/decoding

### 3. CLIP特征提取 (CLIP Feature Extraction)
- ✅ 使用CLIP编码视频帧 - Encode video frames using CLIP
- ✅ 多种特征聚合方法 (均值、最大值、注意力) - Multiple aggregation methods
- ✅ 支持不同的CLIP模型 - Support different CLIP models

### 4. 视频字幕生成 (Video Caption Generation)
- ✅ 基于CLIP+GPT-2的字幕生成 - Caption generation based on CLIP+GPT-2
- ✅ 可配置的生成参数 - Configurable generation parameters
- ✅ 时序特征聚合 - Temporal feature aggregation

## 安装依赖 (Installation)

```bash
# 基础依赖
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install numpy pillow

# CLIP
pip install git+https://github.com/openai/CLIP.git
```

## 快速开始 (Quick Start)

### 运行所有示例 (Run All Examples)

```bash
python video_codec_example.py --all
```

### 运行特定示例 (Run Specific Example)

```bash
# 示例1: 视频帧提取
python video_codec_example.py --example 1

# 示例2: 帧编解码
python video_codec_example.py --example 2

# 示例3: CLIP特征提取
python video_codec_example.py --example 3

# 示例4: 视频字幕生成
python video_codec_example.py --example 4

# 示例5: 视频重建
python video_codec_example.py --example 5
```

### 处理自定义视频 (Process Custom Video)

```bash
python video_codec_example.py --video /path/to/your/video.mp4 --example 4
```

## 代码示例 (Code Examples)

### 1. 视频帧提取 (Video Frame Extraction)

```python
from video_utils import VideoProcessor

# 创建处理器
processor = VideoProcessor()

# 提取帧
frames, metadata = processor.extract_frames(
    "video.mp4", 
    num_frames=10  # 提取10帧
)

print(f"提取了 {len(frames)} 帧")
print(f"视频信息: {metadata}")
```

### 2. 帧编码和解码 (Frame Encoding/Decoding)

```python
from video_utils import VideoFrameEncoder

# 创建编码器
encoder = VideoFrameEncoder(quality=90)

# 编码帧
encoded = encoder.encode_frames(frames)

# 解码帧
decoded = encoder.decode_frames(encoded)

print(f"压缩率: {original_size / encoded_size:.2f}x")
```

### 3. CLIP特征提取 (CLIP Feature Extraction)

```python
from video_caption import VideoCLIPEncoder
import torch

# 初始化编码器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_encoder = VideoCLIPEncoder("ViT-B/32", device=device)

# 提取特征
features = clip_encoder.encode_frames(frames)

# 聚合特征
aggregated = clip_encoder.aggregate_features(features, method="mean")

print(f"特征形状: {features.shape}")
```

### 4. 视频字幕生成 (Video Caption Generation)

```python
from video_caption import VideoCaptionGenerator

# 初始化生成器
generator = VideoCaptionGenerator(
    model_type="gpt2",
    clip_model_type="ViT-B/32",
    prefix_length=10
)

# 生成字幕
caption = generator.generate_caption_from_video(
    "video.mp4",
    num_frames=8,
    aggregation_method="mean",
    max_length=30
)

print(f"生成的字幕: {caption}")
```

### 5. 视频重建 (Video Reconstruction)

```python
from video_utils import VideoProcessor

processor = VideoProcessor()

# 保存帧为视频
processor.save_frames_as_video(
    frames,
    "output.mp4",
    fps=30.0,
    codec='mp4v'
)
```

## 模块说明 (Module Description)

### video_utils.py

视频处理工具模块，提供基础的视频I/O和帧处理功能。

Video processing utilities module, provides basic video I/O and frame processing.

**主要类 (Main Classes):**
- `VideoProcessor`: 视频处理器
- `VideoFrameEncoder`: 帧编码器

### video_caption.py

视频字幕生成模块，基于CLIP和语言模型。

Video caption generation module, based on CLIP and language models.

**主要类 (Main Classes):**
- `VideoCLIPEncoder`: CLIP视频编码器
- `VideoMappingNetwork`: 特征映射网络
- `VideoCaptionGenerator`: 字幕生成器

### video_codec_example.py

完整的示例代码，展示所有功能的使用方法。

Complete example code demonstrating all features.

## 技术架构 (Technical Architecture)

```
视频输入 (Video Input)
    ↓
帧提取 (Frame Extraction)
    ↓
CLIP编码 (CLIP Encoding)
    ↓
特征聚合 (Feature Aggregation)
    ↓
映射网络 (Mapping Network)
    ↓
语言模型生成 (Language Model Generation)
    ↓
字幕输出 (Caption Output)
```

## 支持的视频格式 (Supported Video Formats)

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- FLV (.flv)
- WMV (.wmv)

## 性能优化建议 (Performance Optimization Tips)

1. **帧数选择**: 对于长视频，建议提取8-16帧即可获得良好效果
2. **GPU加速**: 使用CUDA可以显著提升编码和生成速度
3. **批处理**: 批量处理多个视频时，可以复用模型加载
4. **压缩质量**: 根据需求调整JPEG压缩质量，平衡质量和大小

1. **Frame Selection**: For long videos, extracting 8-16 frames is usually sufficient
2. **GPU Acceleration**: Using CUDA significantly improves encoding and generation speed
3. **Batch Processing**: Reuse model loading when processing multiple videos
4. **Compression Quality**: Adjust JPEG quality based on requirements

## 常见问题 (FAQ)

**Q: 需要什么GPU？**
A: 推荐使用4GB以上显存的GPU，CPU也可以运行但速度较慢。

**Q: 支持实时处理吗？**
A: 当前版本主要用于离线处理，实时处理需要进一步优化。

**Q: 可以用于哪些应用场景？**
A: 视频摘要、视频检索、视频内容理解、辅助字幕生成等。

**Q: What GPU is required?**
A: GPU with 4GB+ VRAM is recommended, but CPU also works (slower).

**Q: Does it support real-time processing?**
A: Current version is mainly for offline processing, real-time requires optimization.

**Q: What are the application scenarios?**
A: Video summarization, video retrieval, content understanding, caption generation, etc.

## 许可证 (License)

MIT License

## 参考文献 (References)

- [CLIP: Connecting Text and Images](https://github.com/openai/CLIP)
- [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/abs/2111.09734)
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://openai.com/blog/better-language-models/)

## 联系方式 (Contact)

如有问题或建议，请提交Issue。

For questions or suggestions, please submit an Issue.
