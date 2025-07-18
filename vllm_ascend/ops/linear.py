import itertools
from abc import abstractmethod
from typing import Any, Literal, Optional, Union

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter
import torch_npu
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

from vllm.model_executor.layers.linear import RowParallelLinear, QKVParallelLinear, MergedColumnParallelLinear
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
logger = init_logger(__name__)


class MaybeScatterRowParallelLinear(RowParallelLinear):
    # Replace all_reduce with reduce_scatter in flashcomm_v1 situation
    def __init__(self,
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
        return_bias: bool = True,):
        super().__init__(input_size,
                    output_size,
                    bias,
                    input_is_parallel,
                    skip_bias_add,
                    params_dtype,
                    reduce_results,
                    quant_config,
                    prefix)
        self.quant_method = CostomUnquantizedLinearMethod()

    def forward(
        self, input_, use_scatter, pad_size,
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
        # print(f'zyl_before = {input_parallel.shape}')
        if self.reduce_results and self.tp_size > 1:
            if use_scatter:
                output_parallel = self.quant_method.apply_prefill(self,
                                            input_parallel,
                                            pad_size,
                                            bias=bias_)
                output = output_parallel                       
            else:
                output_parallel = self.quant_method.apply(self,
                                            input_parallel,
                                            bias=bias_)
                output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel
        

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias


class CostomUnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply_prefill(self,
              layer: torch.nn.Module,   # [5120, 1280]
              x: torch.Tensor,          # [8192,1280]
              pad_size: int,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        origin_device = x.device
        origin_dtype = x.dtype
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        current_rank = torch.npu.current_device()
        commDomain = str(current_rank // tp_size)
        output = torch.empty(x.shape[0] // tp_size, layer.weight.shape[0], dtype=origin_dtype, device=origin_device)
        torch_npu.atb._npu_matmul_reduce_scatter(x, layer.weight, output, bias, rank=tp_rank, rankSize=tp_size, commDomain=commDomain)
        # print(f'zyl_module = {output.shape}')
        return output
    
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return dispatch_unquantized_gemm()(x, layer.weight, bias)


class MaybeGatherQKVParallelLinear(QKVParallelLinear):

    def forward(
        self,
        input_, 
        use_gather: bool = False,
        pad_size: int = -1,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None
        
        # Matrix multiply.
        assert self.quant_method is not None

        if use_gather:
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            origin_device = input_.device
            origin_dtype = input_.dtype
            output = torch.empty(input_.shape[0]*tp_size, self.weight.shape[0], dtype=origin_dtype, device=origin_device)
            currnet_rank = torch.npu.current_device()
            commDomain = str(currnet_rank // tp_size)


            torch_npu.atb._npu_all_gather_matmul(input_, self.weight, output, None, rank=tp_rank, rankSize=tp_size, commDomain=commDomain)
            if bias is not None:
                bias = bias.reshape(1, -1)
                output = torch.add(output, bias)

            if pad_size > 0:
                output = output[:-pad_size, :]

        else:
            output_parallel = self.quant_method.apply(self, input_, bias)
            if self.gather_output:
                # All-gather across the partitions.
                output = tensor_model_parallel_all_gather(output_parallel)
            else:
                output = output_parallel
        
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

class MaybeGatherMergedColumnParallelLinear(MergedColumnParallelLinear):

    def forward(
        self,
        input_, 
        use_gather: bool = False,
        pad_size: int = -1,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None

        
        if use_gather:
            
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            origin_device = input_.device
            origin_dtype = input_.dtype
            
            output = torch.empty(input_.shape[0]*tp_size, self.weight.shape[0], dtype=origin_dtype, device=origin_device)
            currnet_rank = torch.npu.current_device()
            commDomain = str(currnet_rank // tp_size)

            torch_npu.atb._npu_all_gather_matmul(input_, self.weight, output, bias, rank=tp_rank, rankSize=tp_size, commDomain=commDomain)

            if pad_size > 0:
                output = output[:-pad_size, :]
            
        else:
            output_parallel = self.quant_method.apply(self, input_, bias)
            if self.gather_output:
                # All-gather across the partitions.
                output = tensor_model_parallel_all_gather(output_parallel)
            else:
                output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias