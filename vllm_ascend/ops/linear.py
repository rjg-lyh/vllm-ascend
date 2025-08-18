import itertools
from abc import abstractmethod
from typing import Any, Literal, Optional, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter
import torch_npu

from vllm import envs
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
# yapf: disable
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           BlockQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           PerTensorScaleParameter,
                                           RowvLLMParameter)
# yapf: enable
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.model_executor.layers.linear import LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.linear import RowParallelLinear, QKVParallelLinear, MergedColumnParallelLinear
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod, quant_per_tensor


class CustomRowParallelLinear(RowParallelLinear):
    # Replace all_reduce with reduce_scatter in flashcomm_v1 situation
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        super().__init__(input_size,
                    output_size,
                    bias,
                    input_is_parallel,
                    skip_bias_add,
                    params_dtype,
                    reduce_results,
                    quant_config,
                    prefix,
                    return_bias=return_bias)

    def forward(
        self,
        input_: torch.Tensor,
        flashcomm_v1_enabled: bool = False,
        matmul_rs_enabled: bool = False,
        pad_size: int = 0,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        if self.tp_size == 1:
            output = self.quant_method.apply(self,
                                            input_parallel,
                                            bias=bias_)
        elif not self.reduce_results:
            output = self.quant_method.apply(self,
                                            input_parallel,
                                            bias=bias_)
        elif not flashcomm_v1_enabled:
            output_parallel = self.quant_method.apply(self,
                                                    input_parallel,
                                                    bias=bias_)
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            if not matmul_rs_enabled:
                output_parallel = self.quant_method.apply(self,
                                                        input_parallel,
                                                        bias=bias_)
                if pad_size > 0:
                    output_parallel = F.pad(output_parallel, (0, 0, 0, pad_size))
                output = tensor_model_parallel_reduce_scatter(output_parallel, 0)
            # 浮点
            elif isinstance(self.quant_method, UnquantizedLinearMethod):
                if pad_size > 0:
                    input_parallel = F.pad(input_parallel, (0, 0, 0, pad_size))
                output_parallel = torch.empty(input_parallel.shape[0] // self.tp_size, self.weight.shape[0], dtype=input_parallel.dtype, device=input_parallel.device)
                current_rank = torch.npu.current_device()
                commDomain = str(current_rank // self.tp_size)
                torch_npu.atb._npu_matmul_reduce_scatter(input_parallel, self.weight, output_parallel, None, None,  rank=self.tp_rank, rankSize=self.tp_size, commDomain=commDomain, outdata_type=27)
                output = output_parallel
            # w8a8
            elif isinstance(self.quant_method.quant_method, AscendW8A8LinearMethod):
                if pad_size > 0:
                    input_parallel = F.pad(input_parallel, (0, 0, 0, pad_size))
                input_parallel_quant = quant_per_tensor(input_parallel, self.aclnn_input_scale_reciprocal, self.aclnn_input_offset)
                output_parallel = torch.empty(input_parallel_quant.shape[0] // self.tp_size, self.weight.shape[1], dtype=input_parallel.dtype, device=input_parallel.device)
                bias_ = torch.zeros(self.weight.shape[1], dtype=torch.int).unsqueeze(0)
                current_rank = torch.npu.current_device()
                commDomain = str(current_rank // self.tp_size)
                deq_scale = self.deq_scale.unsqueeze(0)
                torch_npu.atb._npu_matmul_reduce_scatter(input_parallel_quant, self.weight, output_parallel, bias_, deqScale=deq_scale, rank=self.tp_rank, rankSize=self.tp_size, commDomain=commDomain, outdata_type=27)     # outdata_type参数，bf16=27, fp16=1
                output = output_parallel
            # 其他量化方式
            else:
                output_parallel = self.quant_method.apply(self,
                                                        input_parallel,
                                                        bias=bias_)
                if pad_size > 0:
                    output_parallel = F.pad(output_parallel, (0, 0, 0, pad_size))
                output = tensor_model_parallel_reduce_scatter(output_parallel, 0)

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias


class CustomQKVParallelLinear(QKVParallelLinear):
    # Replace all_reduce with all_gather in flashcomm_v1 situation
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        super().__init__(hidden_size=hidden_size,
                         head_size=head_size,
                         total_num_heads=total_num_heads,
                         total_num_kv_heads=total_num_kv_heads,
                         bias=bias,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix,
                         return_bias=return_bias)

    def forward(
        self,
        input_: torch.Tensor,
        flashcomm_v1_enabled: bool = False,
        ag_matmal_enabled: bool = False,
        pad_size: int = 0,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        if not flashcomm_v1_enabled:
            output_parallel = self.quant_method.apply(self, input_, bias)
        elif not ag_matmal_enabled:
            output_parallel = tensor_model_parallel_all_gather(input_, 0)
            if pad_size > 0:
                output_parallel = output_parallel[:-pad_size, :]
        # 浮点
        elif isinstance(self.quant_method, UnquantizedLinearMethod):
            pass
        # w8a8
        elif isinstance(self.quant_method.quant_method, AscendW8A8LinearMethod):
            output_parallel = torch.empty(input_.shape[0]*self.tp_size, self.weight.shape[1], dtype=input_.dtype, device=input_.device)
            input_quant = quant_per_tensor(input_, self.aclnn_input_scale_reciprocal, self.aclnn_input_offset)
            current_rank = torch.npu.current_device()
            commDomain = str(current_rank // self.tp_size)
            bias_ = torch.zeros(self.weight.shape[1], dtype=torch.int).unsqueeze(0)
            deq_scale = self.deq_scale.unsqueeze(0)
            torch_npu.atb._npu_all_gather_matmul(input_quant, self.weight, output_parallel, bias_, deqScale=deq_scale, rank=self.tp_rank, rankSize=self.tp_size, commDomain=commDomain, outdata_type=27)
            if bias is not None:
                bias = bias.reshape(1, -1)
                output_parallel = torch.add(output_parallel, bias)
            if pad_size > 0:
                output_parallel = output_parallel[:-pad_size, :]
        # 其他量化方式
        else:
            output_parallel = tensor_model_parallel_all_gather(input_, 0)
            if pad_size > 0:
                output_parallel = output_parallel[:-pad_size, :]

        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias