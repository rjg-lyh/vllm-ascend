from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch_npu

from vllm.distributed import (get_tensor_model_parallel_rank,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.linear import RowParallelLinear, ColumnParallelLinear
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod, quant_per_tensor

def ascend_row_forward(
    self,
    input_: torch.Tensor,
    flashcomm_v1_enabled: bool = False,
    matmul_rs_enabled: bool = False,
    pad_size: int = 0,
) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
    tp_rank = get_tensor_model_parallel_rank()
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
    if self.tp_size == 1 or not self.reduce_results:
        output = self.quant_method.apply(self,
                                        input_parallel,
                                        bias=bias_)
    elif not flashcomm_v1_enabled:
        output_parallel = self.quant_method.apply(self,
                                                input_parallel,
                                                bias=bias_)
        output = tensor_model_parallel_all_reduce(output_parallel)
    # 浮点
    elif matmul_rs_enabled and isinstance(self.quant_method, UnquantizedLinearMethod):
        if pad_size > 0:
            input_parallel = F.pad(input_parallel, (0, 0, 0, pad_size))
        output_parallel = torch.empty(input_parallel.shape[0] // self.tp_size, self.weight.shape[0], dtype=self.params_dtype, device=input_parallel.device)
        current_rank = torch.npu.current_device()
        commDomain = str(current_rank // self.tp_size)
        torch_npu.atb._npu_matmul_reduce_scatter(input_parallel, self.weight, output_parallel, None, None,  rank=self.tp_rank, rankSize=self.tp_size, commDomain=commDomain, outdata_type=27)
        output = output_parallel
    # w8a8
    elif matmul_rs_enabled and isinstance(self.quant_method.quant_method, AscendW8A8LinearMethod):
        if pad_size > 0:
            input_parallel = F.pad(input_parallel, (0, 0, 0, pad_size))
        if input_parallel.dtype != torch.int8:
            input_parallel_quant = quant_per_tensor(input_parallel, self.aclnn_input_scale_reciprocal, self.aclnn_input_offset)
        else:
            input_parallel_quant = input_parallel
        output_parallel = torch.empty(input_parallel_quant.shape[0] // self.tp_size, self.weight.shape[1], dtype=self.params_dtype, device=input_parallel.device)
        bias_ = self.quant_bias if tp_rank == 0 else torch.zeros(self.weight.shape[1], dtype=torch.int).unsqueeze(0)
        current_rank = torch.npu.current_device()
        commDomain = str(current_rank // self.tp_size)
        deq_scale = self.deq_scale.unsqueeze(0)
        torch_npu.atb._npu_matmul_reduce_scatter(input_parallel_quant, self.weight, output_parallel, bias_, deqScale=deq_scale, rank=self.tp_rank, rankSize=self.tp_size, commDomain=commDomain, outdata_type=27)     # outdata_type参数，bf16=27, fp16=1
        output = output_parallel
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


def ascend_column_forward(
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
    # 浮点
    elif ag_matmal_enabled and isinstance(self.quant_method, UnquantizedLinearMethod):
        pass
    # w8a8
    elif ag_matmal_enabled and isinstance(self.quant_method.quant_method, AscendW8A8LinearMethod):
        output_parallel = torch.empty(input_.shape[0]*self.tp_size, self.weight.shape[1], dtype=self.params_dtype, device=input_.device)
        if input_.dtype != torch.int8:
            input_quant = quant_per_tensor(input_, self.aclnn_input_scale_reciprocal, self.aclnn_input_offset)
        else:
            input_quant = input_
        current_rank = torch.npu.current_device()
        commDomain = str(current_rank // self.tp_size)
        bias_ = self.quant_bias
        deq_scale = self.deq_scale.unsqueeze(0)
        tp_rank = get_tensor_model_parallel_rank()
        torch_npu.atb._npu_all_gather_matmul(input_quant, self.weight, output_parallel, bias_, deqScale=deq_scale, rank=tp_rank, rankSize=self.tp_size, commDomain=commDomain, outdata_type=27)
        if bias is not None:
            bias = bias.reshape(1, -1)
            output_parallel = torch.add(output_parallel, bias)
        if pad_size > 0:
            output_parallel = output_parallel[:-pad_size, :]
    else:
        input_ = tensor_model_parallel_all_gather(input_, 0)
        if pad_size > 0:
            input_ = input_[:-pad_size, :]
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


RowParallelLinear.forward = ascend_row_forward
ColumnParallelLinear.forward = ascend_column_forward
