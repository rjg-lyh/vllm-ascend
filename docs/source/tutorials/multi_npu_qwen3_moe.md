# Multi-NPU (Qwen3-30B-A3B)

## Run vllm-ascend on Multi-NPU with Qwen3 MoE

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```

Setup environment variables:

```bash
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

### Online Inference on Multi-NPU

Run the following script to start the vLLM server on Multi-NPU:

For an Atlas A2 with 64GB of NPU card memory, tensor-parallel-size should be at least 2, and for 32GB of memory, tensor-parallel-size should be at least 4.

```bash
vllm serve Qwen/Qwen3-30B-A3B --tensor-parallel-size 4 --enable_expert_parallel
```

Once your server is started, you can query the model with input prompts

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-30B-A3B",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 4096
}'
```

### Offline Inference on Multi-NPU

Run the following script to execute offline inference on multi-NPU:

```python
import gc
import torch

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)
llm = LLM(model="Qwen/Qwen3-30B-A3B",
          tensor_parallel_size=4,
          distributed_executor_backend="mp",
          max_model_len=4096,
          enable_expert_parallel=True)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

del llm
clean_up()
```

If you run this script successfully, you can see the info shown below:

```bash
Prompt: 'Hello, my name is', Generated text: " Lucy. I'm from the UK and I'm 11 years old."
Prompt: 'The future of AI is', Generated text: ' a topic that has captured the imagination of scientists, philosophers, and the general public'
```
