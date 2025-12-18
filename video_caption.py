"""
Video Captioning with CLIP and GPT-2/LLaMA
基于CLIP和生成模型的视频字幕生成
"""

import torch
import torch.nn as nn
import numpy as np
import clip
from typing import List, Optional, Tuple
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
import PIL.Image
from video_utils import VideoProcessor


class VideoCLIPEncoder:
    """
    视频CLIP编码器
    使用CLIP模型提取视频帧的特征表示
    """
    
    def __init__(
        self, 
        clip_model_type: str = "ViT-B/32",
        device: Optional[torch.device] = None
    ):
        """
        Args:
            clip_model_type: CLIP模型类型
            device: 计算设备
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.preprocess = clip.load(clip_model_type, device=self.device, jit=False)
        self.clip_model.eval()
        print(f"CLIP model loaded on {self.device}")
    
    def encode_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        编码视频帧为CLIP特征
        
        Args:
            frames: 视频帧列表 (numpy arrays)
            
        Returns:
            编码后的特征张量 (num_frames, feature_dim)
        """
        features = []
        batch_size = 8  # Process frames in batches for efficiency
        
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                batch_images = []
                
                for frame in batch_frames:
                    # 转换为PIL Image
                    pil_image = PIL.Image.fromarray(frame)
                    # 预处理
                    image = self.preprocess(pil_image)
                    batch_images.append(image)
                
                # Stack batch and encode
                batch_tensor = torch.stack(batch_images).to(self.device)
                batch_features = self.clip_model.encode_image(batch_tensor)
                features.append(batch_features)
        
        # 拼接所有批次的特征
        features = torch.cat(features, dim=0)
        return features
    
    def aggregate_features(
        self, 
        features: torch.Tensor, 
        method: str = "mean"
    ) -> torch.Tensor:
        """
        聚合多帧特征
        
        Args:
            features: 帧特征张量 (num_frames, feature_dim)
            method: 聚合方法 ("mean", "max", "attention")
            
        Returns:
            聚合后的特征 (1, feature_dim)
        """
        if method == "mean":
            return features.mean(dim=0, keepdim=True)
        elif method == "max":
            return features.max(dim=0, keepdim=True)[0]
        elif method == "attention":
            # 简单的自注意力聚合
            attention_weights = torch.softmax(features.mean(dim=-1), dim=0)
            aggregated = (features * attention_weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
            return aggregated
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


class VideoMappingNetwork(nn.Module):
    """
    视频特征映射网络
    将CLIP特征映射到语言模型的嵌入空间
    """
    
    def __init__(
        self, 
        clip_dim: int = 512,
        gpt_dim: int = 768,
        prefix_length: int = 10,
        num_layers: int = 2
    ):
        """
        Args:
            clip_dim: CLIP特征维度
            gpt_dim: GPT嵌入维度
            prefix_length: 前缀长度
            num_layers: MLP层数
        """
        super().__init__()
        
        self.prefix_length = prefix_length
        
        # 构建MLP
        layers = []
        current_dim = clip_dim
        hidden_dim = (gpt_dim * prefix_length) // 2
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, gpt_dim * prefix_length))
        
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CLIP特征 (batch_size, clip_dim)
            
        Returns:
            映射后的前缀嵌入 (batch_size, prefix_length, gpt_dim)
        """
        x = self.mapping(x)
        batch_size = x.shape[0]
        return x.view(batch_size, self.prefix_length, -1)


class VideoCaptionGenerator:
    """
    视频字幕生成器
    整合视频编码和文本生成
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "gpt2",
        clip_model_type: str = "ViT-B/32",
        prefix_length: int = 10,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model_path: 预训练模型路径
            model_type: 语言模型类型 (currently only "gpt2" is supported)
            clip_model_type: CLIP模型类型
            prefix_length: 前缀长度
            device: 计算设备
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prefix_length = prefix_length
        
        # 初始化视频编码器
        self.video_encoder = VideoCLIPEncoder(clip_model_type, self.device)
        
        # 初始化语言模型和分词器
        if model_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.language_model = GPT2LMHeadModel.from_pretrained("gpt2")
            gpt_dim = self.language_model.transformer.wte.weight.shape[1]
        else:
            raise ValueError(f"Model type {model_type} not supported. Currently only 'gpt2' is supported.")
        
        self.language_model = self.language_model.to(self.device)
        self.language_model.eval()
        
        # 初始化映射网络
        self.mapping_network = VideoMappingNetwork(
            clip_dim=512,
            gpt_dim=gpt_dim,
            prefix_length=prefix_length
        ).to(self.device)
        
        # 加载预训练权重（如果提供）
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """加载预训练的映射网络权重"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.mapping_network.load_state_dict(checkpoint, strict=False)
        print(f"Model loaded from {model_path}")
    
    def generate_caption_from_video(
        self,
        video_path: str,
        num_frames: int = 8,
        aggregation_method: str = "mean",
        max_length: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> str:
        """
        从视频生成字幕
        
        Args:
            video_path: 视频文件路径
            num_frames: 提取的帧数
            aggregation_method: 特征聚合方法
            max_length: 生成的最大长度
            temperature: 生成温度
            top_p: nucleus sampling参数
            
        Returns:
            生成的字幕文本
        """
        # 提取视频帧
        processor = VideoProcessor()
        frames, metadata = processor.extract_frames(video_path, num_frames=num_frames)
        
        print(f"Extracted {len(frames)} frames from video")
        print(f"Video metadata: {metadata}")
        
        # 编码帧
        frame_features = self.video_encoder.encode_frames(frames)
        
        # 聚合特征
        aggregated_features = self.video_encoder.aggregate_features(
            frame_features, 
            method=aggregation_method
        )
        
        # 映射到语言模型空间
        with torch.no_grad():
            prefix_embed = self.mapping_network(aggregated_features)
        
        # 生成文本
        caption = self._generate_text(
            prefix_embed,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        return caption
    
    def _generate_text(
        self,
        prefix_embed: torch.Tensor,
        max_length: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> str:
        """
        使用前缀嵌入生成文本
        
        Args:
            prefix_embed: 前缀嵌入 (1, prefix_length, dim)
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: nucleus sampling参数
            
        Returns:
            生成的文本
        """
        self.language_model.eval()
        
        with torch.no_grad():
            # 使用generate方法
            generated = prefix_embed
            tokens_generated = []
            
            for _ in range(max_length):
                outputs = self.language_model(inputs_embeds=generated)
                logits = outputs.logits[:, -1, :] / temperature
                
                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                tokens_generated.append(next_token.item())
                
                # Get embedding for next token
                next_token_embed = self.language_model.transformer.wte(next_token)
                generated = torch.cat([generated, next_token_embed.unsqueeze(1)], dim=1)
            
            # Decode tokens
            if tokens_generated:
                caption = self.tokenizer.decode(tokens_generated, skip_special_tokens=True)
            else:
                caption = ""
        
        return caption


def demo_video_captioning():
    """演示视频字幕生成"""
    from video_utils import create_sample_video
    import tempfile
    import os
    
    print("=" * 60)
    print("视频字幕生成演示 / Video Captioning Demo")
    print("=" * 60)
    
    # 创建测试视频
    temp_dir = tempfile.gettempdir()
    test_video = os.path.join(temp_dir, "demo_video.mp4")
    print(f"\n1. Creating sample video at {test_video}...")
    create_sample_video(test_video, duration=3, fps=10)
    
    # 初始化字幕生成器
    print("\n2. Initializing video caption generator...")
    generator = VideoCaptionGenerator(
        model_type="gpt2",
        clip_model_type="ViT-B/32",
        prefix_length=10
    )
    
    # 生成字幕
    print("\n3. Generating caption from video...")
    caption = generator.generate_caption_from_video(
        test_video,
        num_frames=5,
        aggregation_method="mean",
        max_length=30
    )
    
    print("\n" + "=" * 60)
    print(f"生成的字幕 / Generated Caption:")
    print(f"  {caption}")
    print("=" * 60)


if __name__ == "__main__":
    demo_video_captioning()
