# Anonymous Artifact

> Anonymous code repository for the submitted paper.  
> All author information, affiliations, and personal filesystem paths have been removed for double-blind review.

This repository provides the implementation and scripts for a position-based KV cache eviction strategy for long-context LLM inference.

The method controls KV cache memory usage by selectively retaining tokens according to their positions. Tokens at the beginning of the context are treated as sink tokens and always retained, while recent tokens near the end of the sequence are also preserved. Tokens in the middle region are either uniformly sampled or removed depending on the selected eviction strategy.

The implementation is designed to support long-context evaluation, perplexity computation, greedy decoding, cache visualization, and RoPE-aware cache compaction.

## Installation

Create a Python environment and install the required packages.

```bash
conda create -n kv-eviction python=3.10
conda activate kv-eviction
pip install -r requirements.txt
```

FlashAttention-2 is used when available. If it is not available in the environment, the model loader falls back to standard attention.

## Repository Structure

```text
kv_eviction/
├── __init__.py          # Package entry point and public API re-export
├── eviction.py          # Core position-based eviction logic
├── streaming.py         # Chunked prefill, perplexity evaluation, and greedy decoding
├── rope_patch.py        # RoPE correction after cache compaction
├── model_utils.py       # HuggingFace model/tokenizer loading utilities
└── visualization.py     # Eviction-pattern heatmap generation

scripts/
├── run_longbench.py     # Long-context benchmark evaluation
├── run_pg19.py          # Perplexity evaluation
└── run_booksum.py       # Summarization evaluation

configs/
├── longbench.yaml       # Configuration for long-context evaluation
├── pg19.yaml            # Configuration for perplexity evaluation
└── booksum.yaml         # Configuration for summarization evaluation
```

## Module Overview

### `eviction.py`

This module contains the core eviction logic. It defines the eviction configuration and computes the token indices that should be retained in the KV cache.

Main components include:

- `SimpleEvictConfig`: configuration object for eviction strategies.
- `build_simple_keep_token_idx()`: computes retained-token indices.
- `evict_dynamic_cache_inplace()`: slices HuggingFace `DynamicCache` tensors in place to release memory.

The core eviction logic depends only on PyTorch.

### `streaming.py`

This module supports chunk-based long-sequence processing. It feeds long inputs into the model by chunks and triggers eviction when the cache exceeds the target budget.

Main functions include:

- `streaming_prefill()`: performs chunked prefill with eviction.
- `streaming_ppl()`: computes perplexity under streaming prefill and eviction.
- `greedy_decode()`: performs greedy decoding after streaming prefill.

### `rope_patch.py`

This module handles position correction after cache compaction.

When tokens are evicted from the KV cache, the remaining tokens may have discontinuous positions. The `raw_rel` mode addresses this issue by storing raw keys and reapplying RoPE using contiguous slot positions at attention time.

This module supports RoPE-aware patching for Llama-style and Qwen2-style attention layers.

### `model_utils.py`

This module provides HuggingFace model and tokenizer loading utilities.

It attempts to load models with FlashAttention-2 when available and falls back to standard attention otherwise.

### `visualization.py`

This module visualizes eviction patterns as heatmaps. The visualization shows which token positions are retained or removed after eviction.

## Eviction Strategies

The cache is partitioned into three regions:

```text
[0 ............. sink_end)       -> SINK   (always retained)
[sink_end ... recent_start)      -> MIDDLE (strategy-dependent)
[recent_start ...... total_len)  -> RECENT (always retained)
```

### `sink_recent`

This strategy keeps only sink tokens and recent tokens. The entire middle region is removed.

```python
from kv_eviction.eviction import build_eviction_config

config = build_eviction_config(
    strategy="sink_recent",
    sink_tokens=256,
    recent_tokens=512,
)
```

### `sink_recent_uniform`

This strategy keeps sink tokens, recent tokens, and a uniform sample of blocks from the middle region.

The amount of retained middle-context tokens can be controlled by either `middle_budget` or `uniform_stride`.

#### Budget mode

```python
from kv_eviction.eviction import build_eviction_config

config = build_eviction_config(
    strategy="sink_recent_uniform",
    sink_tokens=256,
    recent_tokens=1024,
    middle_budget=2816,
)
```

#### Stride mode

```python
from kv_eviction.eviction import build_eviction_config

config = build_eviction_config(
    strategy="sink_recent_uniform",
    sink_tokens=256,
    recent_tokens=1024,
    uniform_stride=4,
)
```

If both `middle_budget` and `uniform_stride` are specified, `uniform_stride` takes priority.

## RoPE Modes

The position-encoding behavior after eviction can be selected with the following modes.

### `abs`

This mode keeps the original absolute positions after eviction. It does not require additional patching, but position gaps remain after cache compaction.

### `raw_rel`

This mode stores raw keys in the cache and reapplies RoPE using contiguous slot positions at attention time.

To use this mode, patch the model before running streaming prefill or evaluation:

```python
from kv_eviction.rope_patch import patch_model_raw_kv

patch_model_raw_kv(model)
```

## Library Usage

Example usage for streaming prefill and greedy decoding:

```python
from kv_eviction.eviction import build_eviction_config
from kv_eviction.rope_patch import patch_model_raw_kv
from kv_eviction.streaming import streaming_prefill, greedy_decode

config = build_eviction_config(
    strategy="sink_recent_uniform",
    sink_tokens=256,
    recent_tokens=1024,
    middle_budget=2816,
    rope_mode="raw_rel",
)

patch_model_raw_kv(model)

past_key_values = streaming_prefill(
    model=model,
    tokenizer=tokenizer,
    input_ids=input_ids,
    config=config,
    chunk_size=2048,
    target_cache_size=4096,
)

output_ids = greedy_decode(
    model=model,
    tokenizer=tokenizer,
    input_ids=input_ids,
    past_key_values=past_key_values,
    max_new_tokens=256,
)
```

Example usage for perplexity evaluation:

```python
from kv_eviction.streaming import streaming_ppl

ppl = streaming_ppl(
    model=model,
    tokenizer=tokenizer,
    text=text,
    config=config,
    chunk_size=2048,
    target_cache_size=4096,
)

print(ppl)
```

## Running Experiments

All commands assume that the repository root is the working directory.

### Long-context evaluation

```bash
python scripts/run_longbench.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --sink 256 \
    --recent 1024 \
    --middle 2816 \
    --block-size 128 \
    --rope-mode raw_rel \
    --output results/longbench/
```

### Perplexity evaluation

```bash
python scripts/run_pg19.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --sink 64 \
    --cache-size 16384 \
    --block-size 128 \
    --rope-mode raw_rel \
    --middle-ratio 0.25
```

To run a cache-size or middle-ratio sweep, change the values of `--cache-size` and `--middle-ratio`.

Example:

```bash
for cache_size in 16384 32768 65536; do
    for middle_ratio in 0.25 0.75; do
        python scripts/run_pg19.py \
            --model meta-llama/Meta-Llama-3.1-8B \
            --sink 64 \
            --cache-size ${cache_size} \
            --block-size 128 \
            --rope-mode raw_rel \
            --middle-ratio ${middle_ratio} \
            --output results/pg19_cache_${cache_size}_middle_${middle_ratio}/
    done
done
```

### Summarization evaluation

```bash
python scripts/run_booksum.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --sink 64 \
    --block-size 128 \
    --rope-mode raw_rel \
    --dynamic-cache
```

Cross-model validation can be performed by changing the model argument:

```bash
python scripts/run_booksum.py \
    --model Qwen/Qwen2-7B-Instruct \
    --sink 64 \
    --block-size 128 \
    --rope-mode raw_rel \
    --dynamic-cache
```

### RoPE mode comparison

Run the same evaluation with different RoPE modes:

```bash
python scripts/run_longbench.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --sink 256 \
    --recent 1024 \
    --middle 2816 \
    --block-size 128 \
    --rope-mode raw_rel \
    --output results/longbench_raw_rel/
```

```bash
python scripts/run_longbench.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --sink 256 \
    --recent 1024 \
    --middle 2816 \
    --block-size 128 \
    --rope-mode abs \
    --output results/longbench_abs/
```

### Block-size sensitivity

```bash
for b in 64 128 192; do
    python scripts/run_longbench.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --sink 256 \
        --recent 1024 \
        --middle 2816 \
        --block-size ${b} \
        --rope-mode raw_rel \
        --output results/blocksize_${b}/
done
```

## Configuration Reference

| Evaluation | Sink | Recent | Middle Budget | Total Cache | Block Size | RoPE Mode |
|---|---:|---:|---:|---:|---:|---|
| Long-context evaluation | 256 | 1,024 | 2,816 | 4,096 | 128 | `raw_rel` |
| Perplexity evaluation | 64 | varies | varies | 16,384 | 128 | `raw_rel` |
| Summarization evaluation | 64 | dynamic | dynamic | dynamic | 128 | `raw_rel` |

YAML versions of the experiment configurations are provided in `configs/`.

## Visualization

Eviction patterns can be visualized as heatmaps.

```python
from kv_eviction.visualization import visualize_eviction_pattern

visualize_eviction_pattern(
    keep_indices=keep_indices,
    total_len=total_len,
    save_path="eviction_pattern.png",
)
```

The generated heatmap shows which token positions are retained or removed after eviction.

## Notes

- This repository is anonymized for double-blind review.
- No author names, affiliations, personal emails, or personal filesystem paths are included.
- Model checkpoints and datasets must be obtained separately according to their original licenses.
- The repository provides implementation and scripts for reproducing the main experimental behavior of the submitted paper.
