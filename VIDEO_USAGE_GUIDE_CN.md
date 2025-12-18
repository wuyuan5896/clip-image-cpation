# 生成式视频编解码 - 使用指南

## 概述

这是一个基于CLIP和大语言模型的视频编解码与字幕生成系统。本系统实现了从视频到文本描述的完整pipeline，包括视频帧提取、特征编码、时序聚合和文本生成等功能。

## 主要特点

### 1. 完整的视频处理流程
- **视频帧提取**: 支持多种采样策略（均匀采样、按帧率采样）
- **帧编解码**: 基于JPEG的高效压缩编解码
- **元数据管理**: 自动提取和管理视频元数据

### 2. 深度学习特征提取
- **CLIP编码**: 使用OpenAI的CLIP模型提取视频帧的语义特征
- **时序聚合**: 支持多种特征聚合方法（均值、最大值、注意力机制）
- **GPU加速**: 充分利用GPU加速特征提取过程

### 3. 智能字幕生成
- **多模型支持**: 支持GPT-2、LLaMA等多种语言模型
- **可配置参数**: 支持自定义生成长度、温度、top-p等参数
- **高质量输出**: 生成连贯、准确的视频描述

## 安装步骤

### 1. 环境要求
- Python 3.8+
- CUDA 10.2+ (推荐，用于GPU加速)
- 4GB+ 显存 (推荐)

### 2. 安装依赖

```bash
# 安装核心依赖
pip install -r video_codec_requirements.txt

# 安装CLIP
pip install git+https://github.com/openai/CLIP.git
```

### 3. 验证安装

```bash
# 运行测试脚本
python test_video_codec.py
```

如果所有测试通过，说明安装成功。

## 快速开始

### 运行完整示例

```bash
# 运行所有示例
python video_codec_example.py --all
```

这将依次运行以下5个示例：
1. 视频帧提取
2. 帧编码和解码
3. CLIP特征提取
4. 视频字幕生成
5. 视频重建

### 运行单个示例

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

### 处理自定义视频

```bash
# 为自定义视频生成字幕
python video_codec_example.py --video /path/to/your/video.mp4 --example 4

# 指定输出路径
python video_codec_example.py --video input.mp4 --output output.mp4 --example 5
```

## 代码示例

### 示例1: 视频帧提取

```python
from video_utils import VideoProcessor

# 创建视频处理器
processor = VideoProcessor()

# 获取视频信息
video_info = processor.get_video_info("video.mp4")
print(f"视频总帧数: {video_info['total_frames']}")
print(f"视频帧率: {video_info['fps']}")
print(f"视频时长: {video_info['duration']}秒")

# 提取固定数量的帧
frames, metadata = processor.extract_frames(
    "video.mp4", 
    num_frames=10  # 提取10帧
)

# 按帧率提取
frames, metadata = processor.extract_frames(
    "video.mp4",
    fps=1.0  # 每秒提取1帧
)

# 提取所有帧（限制最大数量）
frames, metadata = processor.extract_frames(
    "video.mp4",
    max_frames=100  # 最多100帧
)
```

### 示例2: 帧编码和解码

```python
from video_utils import VideoFrameEncoder

# 创建编码器（质量参数0-100）
encoder = VideoFrameEncoder(quality=90)

# 编码单个帧
encoded_data = encoder.encode_frame(frame)

# 解码单个帧
decoded_frame = encoder.decode_frame(encoded_data)

# 批量编码
encoded_frames = encoder.encode_frames(frames)

# 批量解码
decoded_frames = encoder.decode_frames(encoded_frames)

# 计算压缩率
original_size = sum(frame.nbytes for frame in frames)
encoded_size = sum(len(data) for data in encoded_frames)
compression_ratio = original_size / encoded_size
print(f"压缩率: {compression_ratio:.2f}x")
```

### 示例3: CLIP特征提取

```python
from video_caption import VideoCLIPEncoder
import torch

# 初始化编码器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_encoder = VideoCLIPEncoder(
    clip_model_type="ViT-B/32",  # 可选: ViT-B/16, RN50等
    device=device
)

# 提取视频帧特征
features = clip_encoder.encode_frames(frames)
print(f"特征维度: {features.shape}")  # (num_frames, 512)

# 聚合特征 - 均值法
aggregated_mean = clip_encoder.aggregate_features(features, method="mean")

# 聚合特征 - 最大值法
aggregated_max = clip_encoder.aggregate_features(features, method="max")

# 聚合特征 - 注意力法
aggregated_attn = clip_encoder.aggregate_features(features, method="attention")
```

### 示例4: 视频字幕生成

```python
from video_caption import VideoCaptionGenerator

# 初始化字幕生成器
generator = VideoCaptionGenerator(
    model_type="gpt2",           # 语言模型类型
    clip_model_type="ViT-B/32",  # CLIP模型类型
    prefix_length=10             # 前缀长度
)

# 从视频生成字幕
caption = generator.generate_caption_from_video(
    video_path="video.mp4",
    num_frames=8,                # 提取8帧
    aggregation_method="mean",   # 特征聚合方法
    max_length=50,               # 最大生成长度
    temperature=0.8,             # 生成温度
    top_p=0.9                    # nucleus sampling
)

print(f"生成的字幕: {caption}")
```

### 示例5: 视频重建

```python
from video_utils import VideoProcessor

processor = VideoProcessor()

# 将帧序列保存为视频
processor.save_frames_as_video(
    frames=frames,
    output_path="output_video.mp4",
    fps=30.0,      # 输出帧率
    codec='mp4v'   # 视频编码器
)
```

## 高级用法

### 自定义特征聚合

```python
# 实现自定义聚合函数
def custom_aggregate(features):
    """
    自定义特征聚合
    例如：加权平均，优先考虑中间帧
    """
    num_frames = features.shape[0]
    weights = torch.exp(-torch.abs(torch.arange(num_frames) - num_frames//2))
    weights = weights / weights.sum()
    weights = weights.to(features.device).unsqueeze(-1)
    return (features * weights).sum(dim=0, keepdim=True)

# 使用自定义聚合
aggregated = custom_aggregate(features)
```

### 批量处理多个视频

```python
import glob
from video_caption import VideoCaptionGenerator

# 初始化生成器（只需一次）
generator = VideoCaptionGenerator(
    model_type="gpt2",
    clip_model_type="ViT-B/32"
)

# 批量处理
video_files = glob.glob("videos/*.mp4")
results = []

for video_path in video_files:
    print(f"处理: {video_path}")
    caption = generator.generate_caption_from_video(
        video_path,
        num_frames=8
    )
    results.append({
        'video': video_path,
        'caption': caption
    })
    print(f"字幕: {caption}\n")

# 保存结果
import json
with open('captions.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### 结合预训练权重

```python
from video_caption import VideoCaptionGenerator

# 加载预训练的映射网络权重
generator = VideoCaptionGenerator(
    model_path="path/to/pretrained_weights.pt",  # 预训练权重路径
    model_type="gpt2",
    clip_model_type="ViT-B/32"
)

# 使用预训练模型生成字幕
caption = generator.generate_caption_from_video("video.mp4")
```

## 性能优化建议

### 1. 帧数选择
- **短视频 (< 30秒)**: 5-10帧
- **中等视频 (30秒-2分钟)**: 10-16帧
- **长视频 (> 2分钟)**: 16-32帧

### 2. GPU优化
```python
# 确保使用GPU
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 批处理帧特征提取（更高效）
batch_size = 8
for i in range(0, len(frames), batch_size):
    batch_frames = frames[i:i+batch_size]
    features = clip_encoder.encode_frames(batch_frames)
```

### 3. 内存管理
```python
# 处理大视频时释放内存
import gc
import torch

# 处理完后清理
del features
gc.collect()
torch.cuda.empty_cache()  # 如果使用CUDA
```

### 4. 压缩质量调整
```python
# 根据需求调整压缩质量
# 高质量（文件更大）
encoder_high = VideoFrameEncoder(quality=95)

# 中等质量（平衡）
encoder_medium = VideoFrameEncoder(quality=85)

# 低质量（文件更小）
encoder_low = VideoFrameEncoder(quality=70)
```

## 常见问题

### Q1: 内存不足怎么办？
A: 
- 减少提取的帧数（num_frames参数）
- 使用较小的CLIP模型（如ViT-B/32而非ViT-L/14）
- 批量处理时减小batch_size
- 处理完及时释放显存

### Q2: 生成的字幕质量不好？
A:
- 增加提取的帧数以获得更多信息
- 尝试不同的特征聚合方法（mean, max, attention）
- 调整temperature参数（0.7-1.0）
- 使用预训练的映射网络权重

### Q3: 支持哪些视频格式？
A: 支持常见格式：MP4, AVI, MOV, MKV, FLV, WMV

### Q4: 可以实时处理吗？
A: 当前版本主要用于离线处理。实时处理需要进一步优化，包括：
- 使用更快的特征提取方法
- 减少帧数
- 使用模型量化和蒸馏

### Q5: 如何训练自己的映射网络？
A: 参考主项目的训练代码（train.py），准备视频-字幕对数据集，然后训练映射网络。

## 应用场景

1. **视频内容理解**: 自动分析视频内容并生成摘要
2. **视频检索**: 通过文本查询相关视频
3. **辅助字幕生成**: 为视频生成初始字幕建议
4. **视频分类**: 基于生成的描述对视频进行分类
5. **视频问答**: 结合视频内容回答问题

## 技术细节

### 架构流程
```
视频文件
  ↓
帧提取 (VideoProcessor)
  ↓
CLIP编码 (VideoCLIPEncoder)
  ↓
特征聚合 (aggregate_features)
  ↓
映射网络 (VideoMappingNetwork)
  ↓
语言模型生成 (GPT-2/LLaMA)
  ↓
字幕输出
```

### 关键组件

1. **VideoProcessor**: 负责视频I/O操作
2. **VideoFrameEncoder**: 实现帧级别的编解码
3. **VideoCLIPEncoder**: 使用CLIP提取视频语义特征
4. **VideoMappingNetwork**: 将CLIP特征映射到语言模型空间
5. **VideoCaptionGenerator**: 整合所有组件的高级API

## 许可证

MIT License

## 致谢

本项目基于以下开源项目：
- [CLIP](https://github.com/openai/CLIP) - OpenAI
- [ClipCap](https://arxiv.org/abs/2111.09734) - Original ClipCap paper
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请在GitHub上提交Issue。
