"""
Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
This file is a part of the vllm-ascend project.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch_npu
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter,
                              get_tp_group)
from vllm.model_executor.layers.linear import (WEIGHT_LOADER_V2_SUPPORTED,
                                               ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.forward_context import get_forward_context

from vllm_ascend.distributed.parallel_state import (
    get_mlp_tensor_model_parallel_rank,
    get_mlp_tensor_model_parallel_world_size, get_mlp_tp_group)
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod, quant_per_tensor


class AscendMlpColumnParallelLinear(ColumnParallelLinear):
    """Linear layer with column parallelism.

    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        output_sizes: Optional[list[int]] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        # Divide the weight matrix along the last dimension.
        if prefix.find("gate_up_proj") != -1:
            self.tp_size = get_mlp_tensor_model_parallel_world_size()
            self.tp_rank = get_mlp_tensor_model_parallel_rank()
            self.enable_mlp_optimze = True
        else:
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_rank = get_tensor_model_parallel_rank()
            self.enable_mlp_optimze = False
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]
        LinearBase.__init__(self,
                            input_size,
                            output_size,
                            skip_bias_add,
                            params_dtype,
                            quant_config,
                            prefix,
                            return_bias=return_bias)

        self.gather_output = gather_output

        if output_sizes is None:
            output_sizes = [output_size]

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)


class AscendMlpRowParallelLinear(RowParallelLinear):
    """Linear layer with row parallelism.
    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

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
        if prefix.find("down_proj") != -1:
            self.tp_size = get_mlp_tensor_model_parallel_world_size()
            self.tp_rank = get_mlp_tensor_model_parallel_rank()
            self.enable_mlp_optimze = True
        else:
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_rank = get_tensor_model_parallel_rank()
            self.enable_mlp_optimze = False
        # Divide the weight matrix along the first dimension.
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        LinearBase.__init__(self,
                            input_size,
                            output_size,
                            skip_bias_add,
                            params_dtype,
                            quant_config,
                            prefix,
                            return_bias=return_bias)

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.enable_mlp_optimze:
            tp_rank = get_mlp_tensor_model_parallel_rank()
            if self.input_is_parallel:
                input_parallel = input_
            else:
                tp_rank = get_mlp_tensor_model_parallel_rank()
                splitted_input = split_tensor_along_last_dim(
                    input_, num_partitions=self.tp_size)
                input_parallel = splitted_input[tp_rank].contiguous()
            # Matrix multiply.
            assert self.quant_method is not None
            # Only fuse bias add into GEMM for rank 0 (this ensures that
            # bias will not get added more than once in TP>1 case)
            bias_ = None if (self.tp_rank > 0
                             or self.skip_bias_add) else self.bias
            output_parallel = self.quant_method.apply(self,
                                                      input_parallel,
                                                      bias=bias_)
            output = get_mlp_tp_group().reduce_scatter(output_parallel, 0)
            # output = output[:num_tokens,:]
            # dispose_tensor(output_parallel)
        else:
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
            bias_ = None if (self.tp_rank > 0
                             or self.skip_bias_add) else self.bias
            output_parallel = self.quant_method.apply(self,
                                                      input_parallel,
                                                      bias=bias_)
            if self.reduce_results and self.tp_size > 1:
                output = tensor_model_parallel_all_reduce(output_parallel)
            else:
                output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias


class AscendMlpMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        self.output_sizes = output_sizes
        if prefix.find("gate_up_proj") != -1:
            self.tp_size = get_mlp_tensor_model_parallel_world_size()
            self.tp_rank = get_mlp_tensor_model_parallel_rank()
            self.enable_mlp_optimze = True
        else:
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_rank = get_tensor_model_parallel_rank()
            self.enable_mlp_optimze = False
        assert all(output_size % self.tp_size == 0
                   for output_size in output_sizes)
        AscendMlpColumnParallelLinear.__init__(self,
                                               input_size=input_size,
                                               output_size=sum(output_sizes),
                                               bias=bias,
                                               gather_output=gather_output,
                                               skip_bias_add=skip_bias_add,
                                               params_dtype=params_dtype,
                                               quant_config=quant_config,
                                               prefix=prefix,
                                               return_bias=return_bias)

    def forward(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None
        # self.global_batch_size = vllm_config.scheduler_config.max_num_seqs
        # Matrix multiply.
        assert self.quant_method is not None
        if self.enable_mlp_optimze:
            input2_ = get_mlp_tp_group().all_gather(input_, 0)
            output = self.quant_method.apply(self, input2_, bias)
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


class AscendDenseMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Linear layer with column parallelism.

    Implemented multiple optimization projects for dense models, such as FlashComm and
    communication-computation fusion.
    """

    def forward(
        self,
        input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        forward_context = get_forward_context()
        flashcomm_v1_enabled = forward_context.flashcomm_v1_enabled
        ag_matmal_enabled = forward_context.ag_matmal_enabled
        pad_size = forward_context.pad_size
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


class AscendDenseQKVParallelLinear(QKVParallelLinear):
    """Linear layer with column parallelism.

    Implemented multiple optimization projects for dense models, such as FlashComm and
    communication-computation fusion.
    """

    def forward(
        self,
        input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        forward_context = get_forward_context()
        layer_num = self.prefix.split('.')[2]
        if layer_num == '0':
            flashcomm_v1_enabled = False
        else:
            flashcomm_v1_enabled = forward_context.flashcomm_v1_enabled
        ag_matmal_enabled = forward_context.ag_matmal_enabled
        pad_size = forward_context.pad_size
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


class AscendDenseRowParallelLinear(RowParallelLinear):
    """Linear layer with row parallelism.

    Implemented multiple optimization projects for dense models, such as FlashComm and
    communication-computation fusion.
    """

    def forward(
        self,
        input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        tp_rank = get_tensor_model_parallel_rank()
        forward_context = get_forward_context()
        flashcomm_v1_enabled = forward_context.flashcomm_v1_enabled
        matmul_rs_enabled = forward_context.matmul_rs_enabled
        pad_size = forward_context.pad_size
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
            hcom_name = get_tp_group().device_group._get_backend(torch.device('npu')).get_hccl_comm_name(self.tp_rank)
            world_size = self.tp_size
            output_dtype = torch.bfloat16
            comm_mode = "aiv"
            output_parallel = torch_npu.npu_mm_reduce_scatter_base(input_parallel, 
                                                                   self.weight.t(), 
                                                                   hcom_name, 
                                                                   world_size, 
                                                                   reduce_op="sum", 
                                                                   bias=None, 
                                                                   comm_turn=0, 
                                                                   comm_mode=comm_mode)
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
            bias_ = self.quant_bias
            
            hcom_name = get_tp_group().device_group._get_backend(torch.device('npu')).get_hccl_comm_name(self.tp_rank)
            world_size = self.tp_size
            deq_scale = self.deq_scale
            output_dtype = torch.bfloat16
            comm_mode = "aiv"
            output_parallel = torch_npu.npu_mm_reduce_scatter_base(input_parallel_quant, 
                                                                   self.weight, 
                                                                   hcom_name, 
                                                                   world_size, 
                                                                   reduce_op="sum", 
                                                                   bias=None, 
                                                                   comm_turn=0, 
                                                                   x2_scale=deq_scale, 
                                                                   output_dtype=output_dtype, 
                                                                   comm_mode=comm_mode)
            output = torch.add(output_parallel, torch.mul(bias_, deq_scale).to(torch.bfloat16))

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