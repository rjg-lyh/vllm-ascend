from collections.abc import Iterable
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config
from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_reduce_scatter)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              PPMissingLayer, maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_ascend.ops.layernorm import AddRMSNormW8A8Quant
from vllm_ascend.ops.linear import CustomRowParallelLinear, CustomQKVParallelLinear, CustomMergedColumnParallelLinear
import vllm_ascend.envs as ascend_envs
from vllm_ascend.attention.attention_v1 import AscendAttentionState


def all_gather_and_maybe_unpad(
    hidden_states: torch.Tensor,
    pad_size: int,
) -> torch.Tensor:
    hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
    if pad_size > 0:
        return hidden_states[:-pad_size, :]
    return hidden_states


def maybe_pad_and_reduce_scatter(
    hidden_states: torch.Tensor,
    pad_size: int,
) -> torch.Tensor:
    if pad_size > 0:
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
    hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
    return hidden_states


class CustomQwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = CustomMergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = CustomRowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        flashcomm_v1_enabled: bool,
        matmul_rs_enabled: bool,
        ag_matmal_enabled: bool,
        pad_size: int,
    ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x, flashcomm_v1_enabled, ag_matmal_enabled, pad_size)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x, flashcomm_v1_enabled, matmul_rs_enabled, pad_size)
        return x


class CustomQwen3Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 head_dim: Optional[int] = None,
                 rms_norm_eps: float = 1e-06,
                 qkv_bias: bool = False,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = CustomQKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = CustomRowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=attn_type)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        hidden_states: torch.Tensor,
        flashcomm_v1_enabled: bool,
        matmul_rs_enabled: bool,
        ag_matmal_enabled: bool,
        pad_size: int,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states, flashcomm_v1_enabled, ag_matmal_enabled, pad_size)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        if type(self.rotary_emb) is RotaryEmbedding:
            # We optimized RotaryEmbedding by moving index_select of cos & sin outside.
            # if cos & sin are provided, set is_cos_sin_cached to True to skip index_select.
            q, k = self.rotary_emb(positions,
                                   q,
                                   k,
                                   cos=cos,
                                   sin=sin,
                                   is_cos_sin_cached=True)
        else:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output, flashcomm_v1_enabled, matmul_rs_enabled, pad_size)
        return output


class CustomQwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = CustomQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = CustomQwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        if quant_config is None:
            self.input_layernorm = RMSNorm(config.hidden_size,
                                           eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                    eps=config.rms_norm_eps)
        else:
            from vllm_ascend.quantization.quant_config import AscendQuantConfig
            from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod

            assert isinstance(quant_config, AscendQuantConfig), \
                "Expected quant_config to be an instance of AscendQuantConfig"

            if isinstance(self.self_attn.qkv_proj.quant_method.quant_method,
                          AscendW8A8LinearMethod):
                self.input_layernorm = AddRMSNormW8A8Quant(
                    config.hidden_size,
                    layer=self.self_attn.qkv_proj,
                    eps=config.rms_norm_eps)
            else:
                self.input_layernorm = RMSNorm(config.hidden_size,
                                               eps=config.rms_norm_eps)
            if isinstance(self.mlp.gate_up_proj.quant_method.quant_method,
                          AscendW8A8LinearMethod):
                self.post_attention_layernorm = AddRMSNormW8A8Quant(
                    config.hidden_size,
                    layer=self.mlp.gate_up_proj,
                    eps=config.rms_norm_eps)
            else:
                self.post_attention_layernorm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps)
        
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.self_attn.o_proj.reduce_results = False
        self.mlp.down_proj.reduce_results = False

    def forward(
        self,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        flashcomm_v1_enabled: bool,
        matmul_rs_enabled: bool,
        ag_matmal_enabled: bool,
        pad_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            if flashcomm_v1_enabled:
                if pad_size > 0:
                    residual = F.pad(residual, (0, 0, 0, pad_size))
                residual = torch.chunk(residual, self.tp_size,
                                       dim=0)[self.tp_rank]
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions,
            cos,
            sin,
            hidden_states,
            flashcomm_v1_enabled,
            matmul_rs_enabled,
            ag_matmal_enabled,
            pad_size,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states, flashcomm_v1_enabled, matmul_rs_enabled, ag_matmal_enabled, pad_size)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": CustomQwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class CustomQwen3Model(Qwen2Model):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         decoder_layer_type=CustomQwen3DecoderLayer)
        self.cos_sin_cache = None
        first_existing_layer = self.layers[self.start_layer]
        if type(first_existing_layer.self_attn.rotary_emb) is RotaryEmbedding:
            self.cos_sin_cache = first_existing_layer.self_attn.rotary_emb.cos_sin_cache
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        pad_size = 0
        flashcomm_v1_enabled = False
        matmul_rs_enabled = False
        ag_matmal_enabled = False
        attn_metadata = get_forward_context().attn_metadata
        if ascend_envs.VLLM_ASCEND_ENABLE_FLASHCOMM == 1 and \
            attn_metadata is not None and \
            attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            flashcomm_v1_enabled = True
        if flashcomm_v1_enabled and \
            ascend_envs.VLLM_ASCEND_ENABLE_LCOC_MATMUL_RS == 1:
            matmul_rs_enabled = True
        if flashcomm_v1_enabled and \
            ascend_envs.VLLM_ASCEND_ENABLE_LCOC_AG_MATMUL == 1:
            ag_matmal_enabled = True
        if flashcomm_v1_enabled:
            num_tokens = hidden_states.size(0)
            pad_size = (self.tp_size -
                        (num_tokens % self.tp_size)) % self.tp_size
        # Generate cos and sin outside layers to avoid repeated calculation.
        cos, sin = None, None
        if self.cos_sin_cache is not None:
            cos_sin = self.cos_sin_cache.index_select(0, positions)
            last_dim = cos_sin.size()[-1]
            cos, sin = cos_sin.reshape(-1, 2,
                                       last_dim // 2).repeat(1, 1,
                                                             2).chunk(2,
                                                                      dim=-2)
            # BSNH
            cos, sin = cos.view(1, -1, 1, last_dim).contiguous(), sin.view(
                1, -1, 1, last_dim).contiguous()

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                cos,
                sin,
                hidden_states,
                residual,
                flashcomm_v1_enabled,
                matmul_rs_enabled,
                ag_matmal_enabled,
                pad_size,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        if flashcomm_v1_enabled:
            hidden_states = all_gather_and_maybe_unpad(hidden_states, pad_size)
        return hidden_states


class CustomQwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    # add `CustomQwen3Model` to init self.model
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = CustomQwen3Model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)