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
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.forward_context import get_forward_context

from vllm_ascend.distributed.parallel_state import (
    get_mlp_tensor_model_parallel_rank,
    get_mlp_tensor_model_parallel_world_size, get_mlp_tp_group)
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
from vllm_ascend.utils import (all_gather_and_maybe_unpad,
                               maybe_pad_and_reduce_scatter)

from vllm.model_executor.layers.linear import (  # isort: skip
    WEIGHT_LOADER_V2_SUPPORTED, ColumnParallelLinear, LinearBase,
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear,
    UnquantizedLinearMethod)


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
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        # forward_context = get_forward_context()
        # flashcomm_v1_enabled = forward_context.flashcomm_v1_enabled
        # ag_matmal_enabled = forward_context.ag_matmal_enabled
        # pad_size = forward_context.pad_size
        # if not flashcomm_v1_enabled:
        #     output_parallel = self.quant_method.apply(self, input_, bias)
        # # fp or bf
        # elif ag_matmal_enabled and isinstance(self.quant_method,
        #                                       UnquantizedLinearMethod):
        #     raise NotImplementedError(
        #         "Unquantized AllGather+MatMul fusion is not implemented yet.")
        # # w8a8 quant
        # elif ag_matmal_enabled and isinstance(self.quant_method.quant_method,
        #                                       AscendW8A8LinearMethod):
        #     raise NotImplementedError(
        #         "W8A8 quantized AllGather+MatMul fusion is not implemented yet."
        #     )
        # else:
        #     input_ = all_gather_and_maybe_unpad(input_, pad_size, 0)
        #     output_parallel = self.quant_method.apply(self, input_, bias)
            
        input_ = torch.ops.vllm.flashcomm_all_gather(input_)
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
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        # forward_context = get_forward_context()
        layer_num = self.prefix.split('.')[2]
        # if layer_num == '0':
        #     flashcomm_v1_enabled = False
        # else:
        #     flashcomm_v1_enabled = forward_context.flashcomm_v1_enabled
        # ag_matmal_enabled = forward_context.ag_matmal_enabled
        # pad_size = forward_context.pad_size
        # if not flashcomm_v1_enabled:
        #     output_parallel = self.quant_method.apply(self, input_, bias)
        # # fp or bf
        # elif ag_matmal_enabled and isinstance(self.quant_method,
        #                                       UnquantizedLinearMethod):
        #     raise NotImplementedError(
        #         "Unquantized AllGather+MatMul fusion is not implemented yet.")
        # # w8a8 quant
        # elif ag_matmal_enabled and isinstance(self.quant_method.quant_method,
        #                                       AscendW8A8LinearMethod):
        #     raise NotImplementedError(
        #         "W8A8 quantized AllGather+MatMul fusion is not implemented yet."
        #     )
        # else:
        #     input_ = all_gather_and_maybe_unpad(input_, pad_size, 0)
        #     output_parallel = self.quant_method.apply(self, input_, bias)
        input_ = torch.ops.vllm.flashcomm_all_gather_with_condition(
            input_, layer_num != '0')
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

    def prefetch_gate_up_proj(self,
                              dependency: torch.Tensor):
        # get prefetch model
        forward_context = get_forward_context()
        layer_num = int(self.prefix.split('.')[2])
        prefetch_model = forward_context.prefetch_model
        prefetch_stream = forward_context.prefetch_stream

        # start point of weight prefetch
        forward_context.prefetch_mlp_up = True if self.prefix.split('.')[-2] == 'self_attn' else False
        if forward_context.prefetch_mlp_up:
            prefetch_stream.wait_stream(torch.npu.current_stream())

            with torch.npu.stream(prefetch_stream):
                # For Qwen3-32B
                MLP_GATE_UP_PREFETCH_SIZE = 50 * 1024 * 1024
                torch_npu.npu_prefetch(prefetch_model.model.layers[layer_num].mlp.gate_up_proj.weight, \
                                    dependency, MLP_GATE_UP_PREFETCH_SIZE)


    def wait_prefetch_done(self):
        forward_context = get_forward_context()
        if forward_context.prefetch_mlp_up:
            prefetch_stream = forward_context.prefetch_stream
            # wait until reduce-scatter is done
            torch.npu.current_stream().wait_stream(prefetch_stream)

    def forward(
        self, input_: torch.Tensor
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
            output = self.quant_method.apply(self, input_parallel, bias=bias_)
        # elif not flashcomm_v1_enabled:
        #     output_parallel = self.quant_method.apply(self,
        #                                               input_parallel,
        #                                               bias=bias_)
        #     output = tensor_model_parallel_all_reduce(output_parallel)
        # # fp or bf
        # elif matmul_rs_enabled and isinstance(self.quant_method,
        #                                       UnquantizedLinearMethod):
        #     raise NotImplementedError(
        #         "Unquantized MatMul+ReduceScatter fusion is not implemented yet."
        #     )
        # # w8a8 quant
        # elif matmul_rs_enabled and isinstance(self.quant_method.quant_method,
        #                                       AscendW8A8LinearMethod):
        #     raise NotImplementedError(
        #         "W8A8 quantized MatMul+ReduceScatter fusion is not implemented yet."
        #     )
        # else:
        #     output_parallel = self.quant_method.apply(self,
        #                                               input_parallel,
        #                                               bias=bias_)
        #     output = maybe_pad_and_reduce_scatter(output_parallel, pad_size, 0)
        else:
            output_parallel = self.quant_method.apply(self,
                                                      input_parallel,
                                                      bias=bias_)

            dependency = output_parallel
            self.prefetch_gate_up_proj(dependency)

            output = torch.ops.vllm.flashcomm_reduce(output_parallel)

            # self.wait_prefetch_done()

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias
