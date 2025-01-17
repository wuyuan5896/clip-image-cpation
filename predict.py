# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers import AutoModelForCausalLM,AutoTokenizer
import skimage.io as io
import PIL.Image
import time
import cog

# import torch

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

WEIGHTS_PATHS = {
    "coco": "/server24/rsh/clip-image-cpation/data/coco_train_transfromer/coco_prefix-008.pt",
    # "conceptual-captions": "conceptual_weights.pt",
}

D = torch.device
CPU = torch.device("cpu")


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        
        self.device = torch.device("cuda:{}".format('4') if torch.cuda.is_available() else "cpu")
        print("my device is {}".format(torch.cuda.current_device()))
        self.clip_model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device, jit=False
        )
        # local_model_path = "/server24/rsh/clip-image-cpation/gpt2_pretrained"
        # self.tokenizer = GPT2Tokenizer.from_pretrained(local_model_path)
        local_model_path = "/server24/rsh/clip-image-cpation/llama-3.2-1B/"
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        
        self.models = {}
        self.prefix_length = 15
        for key, weights_path in WEIGHTS_PATHS.items():
            model = ClipCaptionModel(self.prefix_length)
            model.load_state_dict(torch.load(weights_path, map_location=CPU))
            model = model.eval()
            model = model.to(self.device)
            self.models[key] = model

    @cog.input("image", type=cog.Path, help="Input image")
    @cog.input(
        "model",
        type=str,
        # options=WEIGHTS_PATHS.keys(),
        default="coco",
        help="Model to use",
    )
    @cog.input(
        "use_beam_search",
        type=bool,
        default=False,
        help="Whether to apply beam search to generate the output text",
    )
    def predict(self, image, model, use_beam_search,prompt=""):
        """Run a single prediction on the model"""
        # print("my device is {}".format(torch.cuda.current_device()))
        image = io.imread(image)
        model = self.models[model]
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(
                self.device, dtype=torch.float32
            )
            prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        if use_beam_search:
            return generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
        else:
            # model_llama = AutoModelForCausalLM.from_pretrained("/server24/rsh/clip-image-cpation/llama-3.2-1B/")
            # model_llama = model_llama.to(self.device)
            return generate2(model, self.tokenizer, embed=prefix_embed, prompt=prompt)


class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


# class ClipCaptionModel(nn.Module):

#     # @functools.lru_cache #FIXME
#     def get_dummy_token(self, batch_size: int, device: D) -> T:
#         return torch.zeros(
#             batch_size, self.prefix_length, dtype=torch.int64, device=device
#         )

#     def forward(
#         self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None
#     ):
#         embedding_text = self.gpt.transformer.wte(tokens)
#         prefix_projections = self.clip_project(prefix).view(
#             -1, self.prefix_length, self.gpt_embedding_size
#         )
#         # print(embedding_text.size()) #torch.Size([5, 67, 768])
#         # print(prefix_projections.size()) #torch.Size([5, 1, 768])
#         embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
#         if labels is not None:
#             dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
#             labels = torch.cat((dummy_token, tokens), dim=1)
#         out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
#         return out

#     def __init__(self, prefix_length: int, prefix_size: int = 512):
#         super(ClipCaptionModel, self).__init__()
#         self.prefix_length = prefix_length
#         local_model_path = "/server24/rsh/clip-image-cpation/gpt2_pretrained"
#         self.gpt = GPT2LMHeadModel.from_pretrained(local_model_path)
#         print('+'*50)
#         self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
#         # self.gpt_embedding_size = 2048
#         if prefix_length > 10:  # not enough memory
#             self.clip_project = nn.Linear(
#                 prefix_size, self.gpt_embedding_size * prefix_length
#             )
#         else:
#             self.clip_project = MLP(
#                 (
#                     prefix_size,
#                     (self.gpt_embedding_size * prefix_length) // 2,
#                     self.gpt_embedding_size * prefix_length,
#                 )
#             )


# class ClipCaptionPrefix(ClipCaptionModel):
#     def parameters(self, recurse: bool = True):
#         return self.clip_project.parameters()

#     def train(self, mode: bool = True):
#         super(ClipCaptionPrefix, self).train(mode)
#         self.gpt.eval()
#         return self
class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        # embedding_text = self.gpt.transformer.wte(tokens)   #这是什么意思？
        embedding_text = self.gpt.model.embed_tokens(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        self.gpt.eval()
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type='mlp'):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        # self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        # print('start to loade the pretrained llama model....')
        local_model_path = "/server24/rsh/clip-image-cpation/llama-3.2-1B"
        self.gpt = AutoModelForCausalLM.from_pretrained(local_model_path)
        self.gpt_embedding_size = self.gpt.model.embed_tokens.embedding_dim
        # local_model_gpt2_path = "/server24/rsh/clip-image-cpation/gpt2_pretrained"
        # self.gpt2 = GPT2LMHeadModel.from_pretrained(local_model_gpt2_path)
        # gpt_embedding_size = self.gpt2.transformer.wte.weight.shape[1]
        # print(f'gpt embedding size is {gpt_embedding_size}')
        
        if mapping_type == 'mlp':
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        # print('this is clip_project parameters!')
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):
    

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts
def generate2(model, tokenizer, embed,prompt=""):
    # 示例实现，您需要根据实际需求调整
    outputs = model.gpt.generate(
        inputs_embeds=embed,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# def generate2(
#     model,
#     tokenizer,
#     tokens=None,
#     prompt=None,
#     embed=None,
#     entry_count=1,
#     entry_length=67,  # maximum number of words
#     top_p=0.8,
#     temperature=1.0,
#     stop_token: str = ".",
# ):
#     model.eval()
#     generated_num = 0
#     generated_list = []
#     stop_token_index = tokenizer.encode(stop_token)[0]
#     filter_value = -float("Inf")
#     device = next(model.parameters()).device
#     # print('my device:',device)
#     with torch.no_grad():

#         for entry_idx in range(entry_count):
#             if embed is not None:
#                 generated = embed
#             else:
#                 if tokens is None:
#                     tokens = torch.tensor(tokenizer.encode(prompt))
#                     tokens = tokens.unsqueeze(0).to(device)

#                 # generated = model.gpt.transformer.wte(tokens)

#             for i in range(entry_length):

#                 # outputs = model(inputs_embeds=generated)
#                 outputs = model.gpt(inputs_embeds=generated)
#                 logits = outputs.logits
#                 logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
#                 sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#                 cumulative_probs = torch.cumsum(
#                     nnf.softmax(sorted_logits, dim=-1), dim=-1
#                 )
#                 sorted_indices_to_remove = cumulative_probs > top_p
#                 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
#                     ..., :-1
#                 ].clone()
#                 sorted_indices_to_remove[..., 0] = 0

#                 indices_to_remove = sorted_indices[sorted_indices_to_remove]
#                 logits[:, indices_to_remove] = filter_value
#                 next_token = torch.argmax(logits, -1).unsqueeze(0)
#                 # next_token_embed = model.gpt.transformer.wte(next_token)
#                 next_token_embed = model.gpt.model.embed_tokens(next_token)
#                 if tokens is None:
#                     tokens = next_token
#                 else:
#                     tokens = torch.cat((tokens, next_token), dim=1)
#                 generated = torch.cat((generated, next_token_embed), dim=1)
#                 if stop_token_index == next_token.item():
#                     break
            

#             output_list = list(tokens.squeeze().cpu().numpy())
#             output_text = tokenizer.decode(output_list)
#             generated_list.append(output_text)

#     return generated_list[0]

if __name__ == "__main__":
    print(torch.cuda.is_available())
    image_path = "/server24/rsh/clip-image-cpation/Images/COCO_val2014_000000579664.jpg"
    model_name = "coco" 
    # prompt = "what is the background of the image?"
    use_beam_search = False 
    # 实例化 Predictor 类
    predictor = Predictor()
    predictor.setup()  # 加载模型

    # 调用预测方法
    start = time.time()
    output = predictor.predict(image=image_path,model=model_name, use_beam_search=use_beam_search)
    gap_time = time.time() - start
    print('输出结果：')
    # 打印输出结果
    print(output)
    print('耗时:{}'.format(gap_time))
    