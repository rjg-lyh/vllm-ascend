---
name: ascend-model-analysis
description: Generate a 4-chapter Ascend NPU 0day adaptation analysis HTML report for LLM/multimodal models. Use this skill whenever the user wants to analyze a new model for Ascend NPU deployment, create a model architecture analysis report, estimate KVCache, or identify vLLM Ascend incremental operators. Trigger on phrases like "0day适配", "模型分析报告", "Ascend NPU适配", "算子分析", "KVCache估算", "模型结构分析", or when the user points to a config.json and modeling code directory for analysis.
---

# Ascend NPU 0day Model Adaptation Analysis

You are an Ascend NPU LLM inference expert. Your task is to analyze a newly released model's architecture and generate a comprehensive HTML report for 0day adaptation on Ascend NPU (Huawei CANN ecosystem).

## Input

The user provides:
1. **Config directory** — contains `config.json` (HuggingFace format)
2. **Modeling code directory** — contains the model's Python implementation (typically under `transformers/src/transformers/models/<model_name>/`)

Read both to understand the full architecture before writing the report.

## Output

A single self-contained HTML file: `<model-name>-analysis-report.html` in the working directory. The model name is derived from the config directory name or the user's specification.

## Report Structure

The report has exactly 4 chapters, written in **Chinese**, with a **white background**, concise style.

### Chapter 1: 架构概览 (Architecture Overview)

1. **核心参数表** — A table listing all key config parameters (hidden_size, num_layers, num_heads, head_dim, kv_lora_rank, MoE params, etc.)
2. **关键指标** — A 4-column grid showing: 总参数量, 激活参数量, 层数, 注意力类型(s)
3. **层分布图** — A row of small colored boxes (one per layer), each colored by type:
   - Red = Dense MLP, Orange = SparseMoE, Blue = Full Attention, Green = Linear Attention, Purple = MTP, Teal = DSA/Vision
   - Label section breaks (e.g., "Dense: 0-2", "MoE: 3-77")
4. **层类型汇总表** — Which layer ranges use which attention + FFN combination

Parameter counting rules:
- Embedding: `vocab_size × hidden_size`
- MLA Q path: `q_lora_rank × (hidden_size + hidden_size) + num_heads × qk_head_dim × q_lora_rank` (a_proj + b_proj)
- MLA KV path: `kv_lora_rank + qk_rope_head_dim` for a_proj, then `num_heads × (qk_nope_head_dim + v_head_dim)` for b_proj
- Dense MLP (SwiGLU): `3 × hidden_size × intermediate_size`
- MoE routed experts: `n_routed_experts × 3 × hidden_size × moe_intermediate_size` (gate_up + down = 3×)
- MoE shared expert: `3 × hidden_size × moe_intermediate_size`
- Linear attention: count all in_proj weights + out_proj + gate projections
- Active params = per-token active (only top-k expert weights counted)

### Chapter 2: 单层正向算子分析 (Single-Layer Forward Operator Analysis)

For each distinct layer type, produce:

1. **Data Flow Graph (DFG)** — Visual HTML diagram showing the operator execution flow
2. **Operator Detail Table** — A table with columns: #, 算子名称, 输入Shape, 输出Shape, 功能描述

**DFG Drawing Rules:**
- Read `references/css-template.md` for the complete CSS and HTML patterns
- Use horizontal parallel lanes for Q/K/V projections (side by side)
- Show residual skip connections as red dashed wires on the right side
- Use `.dfg-2col` to place attention and FFN pathways side by side
- Color code operators: norm=blue, mm=green, act=yellow, attn=purple, shape=gray, merge=red, route=orange, dsa/conv=teal
- All tensor shapes must be shown as `[B, S, dim]` or `[B, heads, S, head_dim]` using `B`=batch, `S`=seq_len, `T`=total KV length

**Operator Detail Table Rules:**
- Insert the table AFTER the DFG diagram for each layer type
- Use `<h4>` subtitle, then `<div class="card"><table>` with 5 columns
- Group rows by stage with colored header rows (colspan=5): 注意力通路 (blue), DSA/Indexer (teal), 稀疏注意力 (purple), FFN/MoE通路 (orange), 共享专家 (green), 门控后处理 (yellow)
- If a layer type shares operators with a previous type, use an italic reference row: `<tr><td colspan="5" style="text-align:center;color:var(--dim);font-style:italic">注意力通路同A类 (算子1-27)</td></tr>`
- Number operators sequentially within each layer type

End Chapter 2 with a **算子图例与数量汇总** section showing color legend and per-layer-type operator counts.

### Chapter 3: KVCache与参数量估算

1. **每Token KVCache占用量** — Calculate for each attention type:
   - Full Attention (GQA): `(num_kv_heads × head_dim × 2 × dtype_bytes)` bytes/token
   - MLA (absorption mode): `(kv_lora_rank + qk_rope_head_dim) × dtype_bytes` bytes/token
   - Linear Attention: Fixed recurrent state size (no per-token cost), state = `num_heads × head_dim × head_dim × dtype_bytes`
   - DSA Indexer Key Cache: `index_head_dim × index_n_heads × dtype_bytes` (if applicable)
   - Show compression ratio vs full KV cache
2. **参数量分项表** — Break down total params by component (embedding, attention, FFN, MoE, shared experts, etc.)
3. **总参数量与激活参数量** — Summary with formulas

### Chapter 4: vLLM Ascend增量算子/特性

Classify each required operator/feature into priority levels:
- **P0 (阻塞)** — Completely new operator not in CANN/vLLM-Ascend, must be developed
- **P1 (重要)** — Existing operator needs significant modification or new fusion kernel
- **P2 (一般)** — Existing operator with minor adaptation or config change

For each item, provide:
- Operator name
- Priority level (with colored tag: P0=red, P1=orange, P2=blue)
- Description of what's needed
- Why it's needed (reference to the specific model feature)

Common P0 items to watch for:
- **FusedMoE** — For any MoE model (packed gate_up, noaux_tc routing, etc.)
- **FusedMLAProj** — For MLA models (q_a_proj + q_b_proj + kv_a_proj + kv_b_proj fusion)
- **MLA Absorption** — Weight absorption for KV cache compression
- **FusedGatedDeltaRule / FusedSimpleGLA** — For linear attention models
- **FusedCausalConv1d** — For models with 1D causal convolution
- **FusedDSAIndexer** — For DSA sparse attention indexer (unique to GLM5.1)
- **Hybrid Cache Manager** — For models mixing attention types (different cache formats per layer)
- **MRoPE** — For multimodal 3D position encoding

## Architecture Type Recognition

When analyzing the config and modeling code, identify these key patterns:

| Feature | Config Keys | Modeling Code Clues |
|---------|------------|-------------------|
| MLA | `kv_lora_rank`, `q_lora_rank` | `kv_a_proj`, `q_a_proj`, `q_b_proj`, `kv_b_proj` |
| Full Attention | `num_key_value_heads` < `num_attention_heads` | `q_proj`, `k_proj`, `v_proj`, GQA |
| Linear Attention (GLA) | `linear_attention_dim`, `gate_lr` | `SimpleGLA`, `in_proj_qkv`, `g_proj`, `GroupRMSNorm` |
| Linear Attention (Delta) | `use_gated_delta_rule` | `GatedDeltaNet`, `in_proj_qkv`, `CausalConv1d`, `A_log`, `dt_bias` |
| MoE | `n_routed_experts`, `num_experts_per_tok` | `SparseMoeBlock`, `gate_up_proj` packed |
| DSA Indexer | `index_topk`, `index_n_heads` | `Indexer`, `wq_b`, `wk`, `k_norm`, `Einsum` |
| MTP | `num_nextn_predict_layers` | `NextNPredictLayer` |
| Multimodal | Vision config in model config | `VisionEncoder`, `ForConditionalGeneration` |
| MRoPE | `mrope_section` | `MRoPE`, multimodal position encoding |

## Shape Conventions

Use these consistent notations throughout:
- `B` = batch_size
- `S` = sequence length (current query)
- `T` = total KV cache length (S + past)
- `T_i` = tokens dispatched to expert i
- All shapes in `[B, heads, S, head_dim]` format for attention tensors
- All shapes in `[B, S, hidden_size]` format for hidden states

## Workflow

1. Read `config.json` from the config directory
2. Read ALL Python files in the modeling code directory (modeling_*.py, configuration_*.py)
3. Identify the architecture type and all layer variants
4. Calculate all parameter counts and KVCache estimates
5. Identify vLLM Ascend incremental operators
6. Read `references/css-template.md` for the CSS and DFG HTML patterns
7. Generate the complete HTML report
8. Write to `<model-name>-analysis-report.html`

## Important Notes

- The report must be self-contained — all CSS is inline in `<style>`, no external dependencies
- Tables must have `min-width:600px` and `overflow-x:auto` on the card container
- All shapes shown in DFG diagrams must be concrete numbers from the config (not symbolic)
- Parameter counts must be exact (not approximate), calculated from config values
- KVCache estimates must include both per-token cost and compression ratios
- Operator priority classification must be specific to Ascend NPU / CANN / vLLM-Ascend ecosystem
