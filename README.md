# Anonymous Artifact

This repository contains the code and scripts for the submitted paper.

## Overview

This project implements a KV cache eviction strategy for controlling memory usage during long-context LLM inference.

When processing long sequences, the method selectively evicts cached tokens so that KV cache memory does not exceed a fixed budget. Sink tokens at the beginning of the context and recent tokens near the end are always preserved, while tokens in the middle region are either uniformly sampled or fully removed depending on the selected strategy.

## Architecture

```text
kv_eviction/
├── __init__.py          # Package entry point and public API re-export
├── eviction.py          # Core eviction logic (torch only, no transformers dependency)
├── streaming.py         # Chunk-based streaming prefill and perplexity evaluation
├── rope_patch.py        # RoPE monkey-patch for position correction after eviction
├── model_utils.py       # Model/tokenizer loading utilities
└── visualization.py     # Eviction-pattern heatmap generation
```

## Module Descriptions

### eviction.py

Core eviction module. `SimpleEvictConfig` defines the eviction strategy, and `build_simple_keep_token_idx()` computes the token indices to preserve. `evict_dynamic_cache_inplace()` directly slices the K/V tensors in the HuggingFace `DynamicCache` to release memory. This module only depends on PyTorch.

### streaming.py

Provides chunk-based long-sequence processing. It performs eviction when the cache exceeds the target budget. It includes `streaming_prefill()` for generation tasks, `streaming_ppl()` for perplexity evaluation, and `greedy_decode()` for greedy decoding.

### rope_patch.py

Addresses the discontinuous-position issue after eviction. The attention layer is monkey-patched so that raw, unrotated keys are stored in the cache, and RoPE is reapplied at attention time using continuous slot positions `[0..kv_len-1]`. This module supports FlashAttention2 layers in Llama and Qwen2-style models.

### model_utils.py

Loads HuggingFace models and tokenizers. It first attempts to use FlashAttention2 and falls back to standard attention when FlashAttention2 is unavailable.

### visualization.py

Generates heatmaps for eviction patterns. The heatmap shows which tokens are preserved or evicted across layers and positions.

## Eviction Strategies

The cache is divided into three regions:

```text
[0 ............. sink_end)       -> SINK   (always preserved)
[sink_end ... recent_start)      -> MIDDLE (strategy-dependent)
[recent_start ...... total_len)  -> RECENT (always preserved)
```

### sink_recent

Removes the entire middle region. This is the most aggressive strategy and maximizes memory reduction, but it discards all middle-context information.

```python
build_eviction_config(
    strategy="sink_recent",
    sink_tokens=256,
    recent_tokens=512,
)
```

### sink_recent_uniform

Uniformly samples blocks from the middle region. The block size is aligned with the FlashAttention block size, 128 tokens. The amount of preserved middle-region tokens can be controlled in two ways.

#### Budget mode

Directly specifies the number of middle tokens to preserve. The budget is converted into block units, and blocks are selected uniformly using `torch.linspace`.

```python
build_eviction_config(
    strategy="sink_recent_uniform",
    sink_tokens=256,
    recent_tokens=512,
    middle_budget=256,
)
```

#### Stride mode

Preserves every N-th block in the middle region. This option takes priority over the budget mode.

```python
build_eviction_config(
    strategy="sink_recent_uniform",
    sink_tokens=256,
    recent_tokens=512,
    uniform_stride=4,
)
```

## RoPE Modes

The position-encoding behavior after eviction can be selected with the following modes:

- `abs`: Keeps the original absolute positions. This mode works without additional patching, but position gaps remain after eviction.
- `raw_rel`: Uses `patch_model_raw_kv()` to store raw keys in the cache and reapply RoPE with continuous positions at attention time. This mode reduces the position-gap issue after eviction.

## Installation

Create a Python environment and install the required packages.

```bash
conda create -n kv-eviction python=3.10
conda activate kv-eviction
pip install -r requirements.txt
```

If FlashAttention2 is available in the environment, the model loader will try to use it. Otherwise, it falls back to standard attention.

## Basic Usage

The eviction configuration can be created as follows:

```python
from kv_eviction.eviction import build_eviction_config

config = build_eviction_config(
    strategy="sink_recent_uniform",
    sink_tokens=256,
    recent_tokens=512,
    middle_budget=256,
    rope_mode="abs",
)
```

For RoPE correction mode:

```python
from kv_eviction.eviction import build_eviction_config
from kv_eviction.rope_patch import patch_model_raw_kv

config = build_eviction_config(
    strategy="sink_recent_uniform",
    sink_tokens=256,
    recent_tokens=512,
    middle_budget=256,
    rope_mode="raw_rel",
)

patch_model_raw_kv(model)
```

## Running Long-Context Evaluation

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

Example usage for streaming prefill and greedy decoding:

```python
from kv_eviction.streaming import streaming_prefill, greedy_decode

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

The generated heatmap shows which token positions are preserved or removed after eviction.

## Notes

- This repository is anonymized for double-blind review.
- No author names, affiliations, or personal paths are included.
- The repository is intended to provide the implementation and scripts necessary to reproduce the main behavior of the proposed KV cache eviction method.
- Model checkpoints and datasets should be downloaded separately according to their original licenses.
