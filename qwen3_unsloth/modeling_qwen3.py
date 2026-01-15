from typing import Optional

import torch
import torch.nn as nn
from torch.nn.functional import dropout
from transformers import Qwen3Config, ROPE_INIT_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from typing_extensions import Unpack

from qwen3_cache import Cache


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps=1e-6):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_type = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = torch.mean(hidden_states ** 2, dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weights * hidden_states.to(input_type)


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        immediate_size = config.immediate_size
        self.up_proj = nn.Linear(hidden_size, immediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, immediate_size, bias=False)
        self.down_proj = nn.Linear(immediate_size, hidden_size, bias=False)
        self.act_func = config.act_func

    def forward(self, x: torch.Tensor):
        return self.down_proj(self.act_func(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor):
    D = x.shape[-1]
    d = D // 2
    u = x[..., :d]
    v = x[..., d:]
    return torch.concat((-v, u))


def apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    # unsqueeze_dim表示head的注意力的维度
    # cos和sin的维度通常是b,l,d
    # 新增后变为b,h,l,d
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def calculate_cos_sin(hidden_dim, position_ids: torch.Tensor):
    base = 10000
    d = hidden_dim // 2
    B = base ** (1 / d)
    theta_base = 1. / base ** B
    theta_k = theta_base ** torch.arange(d)
    theta = position_ids.outer(theta_k)
    theta = torch.concat((theta, theta), dim=-1)
    cos = theta.cos()
    sin = theta.sin()
    return cos, sin

def repeat_n(hidden_states: torch.Tensor, n: int):
    # 将序列重复n次
    if n == 1:
        return hidden_states
    b, h, l, d = hidden_states.shape
    # 从省内存的角度，这里应该用expand不用repeat，前者仅会创建独立视图，但是底层共享数据，后者是独立的数据副本
    hidden_states = hidden_states[:, :, None, ...].expand(b, h, n, l, d)
    hidden_states = hidden_states.view(b, h*n, l, d)
    return hidden_states

def eager_attention_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        num_head_group: int,
        scaling: float,
        dropout: float = 0.0,
        training: bool = True,
        attention_mask: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    key = repeat_n(key, num_head_group)
    value = repeat_n(value, num_head_group)
    # torch.matmul和@在核心功能上完全一致
    attn = query @ key.transpose(-1, -2) * scaling
    if attention_mask is not None:
        # b, h, l1, l2
        casual_mask = attention_mask[:, :, :, :key.shape[-2]]
        attn = attn + casual_mask
    attn = torch.softmax(attn, dim=-1)
    attn = torch.dropout(attn, p=dropout, train=training)
    output = attn @ value
    # transpose会导致数据在内存不连续
    # 由于在外部需要调用view函数，view要求数据是连续的，因此在此处连续化
    output = output.transpose(1, 2).contiguous()
    # 在外部进行维度变换和投影
    return output, attn



class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_dim: int = config.hidden_dim
        self.num_attention_heads: int = config.num_attention_heads
        self.num_key_value_heads: int = config.num_key_value_heads
        self.head_dim: int = self.hidden_dim // self.num_attention_heads
        self.scaling: float = self.head_dim ** -0.5
        self.attention_dropout: float = config.attention_dropout
        self.num_key_value_groups: int = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_dim, self.head_dim * self.num_attention_heads, bias=config.bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.head_dim * self.num_key_value_heads, bias=config.bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.head_dim * self.num_key_value_groups, bias=config.bias)

        self.o_proj = nn.Linear(self.head_dim * self.num_attention_heads, self.hidden_dim, bias=config.bias)

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embedding: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor]
    ):
        b, l, d = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, l, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = position_embedding
        q, k = apply_rope(q, k, cos, sin, 1)
        forward_func = eager_attention_forward
        output, _ = forward_func(q, k, v, self.num_key_value_groups, self.scaling, self.attention_dropout, self.training, attention_mask)
        output = output.view(b, l, -1)
        output = self.o_proj(output)
        return output

    def forward2(
            self,
            hidden_states: torch.Tensor,
            position_embedding: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[Cache] = None,
            cache_position: torch.LongTensor = None,
            **kwargs: Unpack[FlashAttentionKwargs]
    ):
        b, l, d = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, l, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = position_embedding
        q, k = apply_rope(q, k, cos, sin, 1)
        # 与缓存的k,v cache进行拼接
        if past_key_values is not None:
            cache_kwargs = {"cos": cos, "sin": sin, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        forward_func = eager_attention_forward
        output, _ = forward_func(q, k, v, self.num_key_value_groups, self.scaling, self.attention_dropout, self.training, attention_mask)
        output = output.view(b, l, -1)
        output = self.o_proj(output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.layer_idx = layer_idx
        self.mlp = Qwen3MLP(config)
        self.attention = Qwen3Attention(config, layer_idx)
        self.pre_attention_norm = Qwen3RMSNorm(self.hidden_size, config.rms_eps)
        self.post_attention_norm = Qwen3RMSNorm(self.hidden_size, config.rms_eps)

    def forward(
            self,
            hidden_states,
            position_embedding: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None
    ):
        x = hidden_states
        res = self.pre_attention_norm(hidden_states)
        x = x + self.attention(res, position_embedding, attention_mask, past_key_values, cache_position)

        res = self.post_attention_norm(x)
        x = x + self.mlp(res)
        return x


class Qwen3RotateEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand((position_ids.shape[0], -1, 1)).to(x.device)
        position_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freq = (inv_freq_expanded.float() @ position_expanded.float()).transpose(1, 2)
            emb = torch.concat([freq, freq], dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(x.dtype), sin.to(x.dtype)


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        # 填充
        self.pad_token_id = config.pad_token_id
        # 词典大小
        self.vocab_size = config.vocab_size
        # 词典映射表
        self.vocab = nn.Embedding(config.vocab_size, config.hidden_size, self.pad_token_id)
        # 中间层
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])

        # 后norm层
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # rope层，准备position embedding
        self.rope_emb = Qwen3RotateEmbedding(config)
        # gradient_checkpointing，时间换空间
        self.gradient_checkpointing = False
        #
        self.has_sliding_layers = "sliding_attention" in config.layer_types

        # 初始化
        # self.post_init()


