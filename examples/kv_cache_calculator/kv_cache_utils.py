#!/usr/bin/env python3
"""
Shared utilities for KV cache calculations
Provides common functions and constants used across visualization and table generation
"""

import json

# Table formatting constants
TABLE_WIDTH = 140
TABLE_SEPARATOR = "=" * TABLE_WIDTH
ROW_SEPARATOR = "-" * TABLE_WIDTH
MODEL_COLUMN_WIDTH = 40
VALUE_COLUMN_WIDTH = 12

# GPU memory thresholds (in GiB)
GPU_CONSUMER_MEMORY = 24
GPU_A100_40_MEMORY = 40
GPU_A100_80_MEMORY = 80
GPU_H100_MEMORY = 80
GPU_MULTI_160_MEMORY = 160
GPU_MULTI_320_MEMORY = 320
GPU_MULTI_640_MEMORY = 640

# Color scheme for feasibility visualization
FEASIBILITY_COLORS = {
    'fits_consumer': '#2ecc71',  # Green - fits in consumer GPU
    'fits_datacenter': '#f39c12',  # Yellow - needs datacenter GPU
    'needs_multi': '#e67e22',     # Orange - needs multiple GPUs
    'challenging': '#e74c3c'      # Red - very challenging
}

# Model family colors for visualization - expanded palette
MODEL_FAMILY_COLORS = {
    "llama": "#1f77b4",
    "mistral": "#ff7f0e",
    "mixtral": "#ff7f0e",  # Same family as mistral
    "qwen": "#2ca02c",
    "qwen2": "#17becf",
    "qwen3": "#bcbd22",
    "deepseek": "#d62728",
    "gemma": "#9467bd",
    "phi": "#e377c2",
    "default": "#7f7f7f"
}


def load_model_configs(filepath="modelconfig.json"):
    """
    Load model configurations from JSON file

    Args:
        filepath: Path to the modelconfig.json file

    Returns:
        Dictionary of model configurations
    """
    with open(filepath, "r") as f:
        return json.load(f)


def get_model_architecture(model_name, config):
    """
    Determine model architecture from configuration

    Future improvement: This should be replaced with an explicit
    'architecture' field in modelconfig.json for robustness

    Args:
        model_name: Full model name (e.g., "meta-llama/Llama-3.1-8B")
        config: Model configuration dictionary

    Returns:
        Architecture type: 'deepseek', 'qwen3', 'moe', 'qwen3-next', 'qwen3-omni', or 'standard'
    """
    # Check for explicit architecture field first
    if 'architecture' in config:
        return config['architecture']

    # Check for architecture-specific fields (more reliable than name)
    if 'kv_lora_rank' in config and 'qk_rope_head_dim' in config:
        return 'deepseek'

    # MOE models have num_local_experts field
    if 'num_local_experts' in config:
        return 'moe'
    # Qwen3 models have explicit head_dim field
    if 'head_dim' in config and 'qwen3' in model_name.lower():
        return 'qwen3'

    # Fallback to name-based detection for backward compatibility
    # TODO: Replace with explicit architecture field in modelconfig.json
    model_lower = model_name.lower()
    if 'deepseek' in model_lower:
        return 'deepseek'
    elif 'qwen/qwen3-' in model_lower:
        return 'qwen3'
    elif 'mixtral' in model_lower or 'moe' in model_lower:
        return 'moe'
    elif 'qwen3-next' in model_lower:
        return 'qwen3-next'
    elif 'qwen3-omni' in model_lower:
        return 'qwen3-omni'

    return 'standard'


def calculate_kv_cache_size(config, tokens, dtype="float16", model_name=""):
    """
    Calculate KV cache size in GiB for a given model configuration

    Uses model-specific formulas:
    - Standard: 2 × layers × tokens × kv_heads × (hidden_size/attention_heads)
    - DeepSeek: layers × tokens × (kv_lora_rank + qk_rope_head_dim)
    - Qwen3: 2 × layers × tokens × kv_heads × head_dim
    - MOE: Same as standard (KV cache doesn't change with expert count)
    - Qwen3-Next: Uses hybrid attention, reduced KV cache for linear layers
    - Qwen3-Omni: Similar to standard but optimized for multimodal

    Args:
        config: Model configuration dictionary
        tokens: Number of tokens
        dtype: Data type ('float32', 'float16', 'bfloat16', 'int8')
        model_name: Model name for architecture detection

    Returns:
        KV cache size in GiB
    """
    # Determine dtype size in bytes
    dtype_sizes = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}
    dtype_size = dtype_sizes.get(dtype, 2)

    # Determine architecture
    architecture = get_model_architecture(model_name, config)

    if architecture == 'deepseek':
        # DeepSeek models use KV-LoRA optimization
        kv_lora_rank = config.get("kv_lora_rank", 512)
        qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
        num_hidden_layers = config["num_hidden_layers"]
        total_elements = num_hidden_layers * tokens * (kv_lora_rank + qk_rope_head_dim)

    elif architecture == 'qwen3':
        # Qwen3 models use explicit head_dim
        num_hidden_layers = config["num_hidden_layers"]
        num_key_value_heads = config["num_key_value_heads"]
        head_dim = config.get("head_dim", 128)
        total_elements = 2 * num_hidden_layers * tokens * num_key_value_heads * head_dim

    elif architecture == 'qwen3-next':
        # Qwen3-Next uses hybrid attention: 1/4 layers use GQA, rest use linear attention
        # Linear attention doesn't need KV cache, so only 1/4 of layers contribute
        num_hidden_layers = config["num_hidden_layers"]
        num_key_value_heads = config["num_key_value_heads"]
        hidden_size = config["hidden_size"]
        num_attention_heads = config["num_attention_heads"]
        head_size = hidden_size / num_attention_heads
        # Only 1/4 of layers (every 4th layer) use traditional attention with KV cache
        gqa_layers = num_hidden_layers // 4
        total_elements = 2 * gqa_layers * tokens * num_key_value_heads * head_size

    elif architecture in ['moe', 'qwen3-omni']:
        # MOE and Qwen3-Omni models use standard KV cache calculation
        # Expert count doesn't affect KV cache size
        hidden_size = config["hidden_size"]
        num_attention_heads = config["num_attention_heads"]
        num_hidden_layers = config["num_hidden_layers"]
        num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
        head_size = hidden_size / num_attention_heads
        total_elements = 2 * num_hidden_layers * tokens * num_key_value_heads * head_size
    else:
        # Standard transformer calculation
        hidden_size = config["hidden_size"]
        num_attention_heads = config["num_attention_heads"]
        num_hidden_layers = config["num_hidden_layers"]
        num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
        head_size = hidden_size / num_attention_heads
        total_elements = 2 * num_hidden_layers * tokens * num_key_value_heads * head_size

    total_bytes = total_elements * dtype_size
    return total_bytes / (1024**3)  # Convert to GiB


def select_representative_models(configs):
    """
    Select a representative set of models for visualization

    Args:
        configs: Full model configurations dictionary

    Returns:
        Dictionary of selected models with their configurations
    """
    # Define representative models across different size categories
    selected = {
        # Small models (< 5B)
        "meta-llama/Llama-3.2-1B-Instruct": configs["meta-llama/Llama-3.2-1B-Instruct"],
        "meta-llama/Llama-3.2-3B-Instruct": configs["meta-llama/Llama-3.2-3B-Instruct"],
        "Qwen/Qwen2.5-3B-Instruct": configs["Qwen/Qwen2.5-3B-Instruct"],

        # Medium models (5B-10B)
        "mistralai/Mistral-7B-Instruct-v0.2": configs["mistralai/Mistral-7B-Instruct-v0.2"],
        "meta-llama/Llama-3.1-8B-Instruct": configs["meta-llama/Llama-3.1-8B-Instruct"],
        "Qwen/Qwen2.5-7B-Instruct": configs["Qwen/Qwen2.5-7B-Instruct"],
        "google/gemma-2-9b": configs["google/gemma-2-9b"],

        # Large models (10B-50B)
        "Qwen/Qwen2.5-14B-Instruct": configs["Qwen/Qwen2.5-14B-Instruct"],
        "google/gemma-2-27b": configs["google/gemma-2-27b"],
        "Qwen/Qwen2.5-32B-Instruct": configs["Qwen/Qwen2.5-32B-Instruct"],
        "Qwen/Qwen3-32B": configs["Qwen/Qwen3-32B"],

        # Very large models (50B+)
        "meta-llama/Llama-3.1-70B-Instruct": configs["meta-llama/Llama-3.1-70B-Instruct"],
        "Qwen/Qwen2.5-72B-Instruct": configs["Qwen/Qwen2.5-72B-Instruct"],
        "mistralai/Mistral-Large-Instruct-2407": configs["mistralai/Mistral-Large-Instruct-2407"],

        # Ultra-large models
        "meta-llama/Llama-3.1-405B": configs["meta-llama/Llama-3.1-405B"],
        "deepseek-ai/DeepSeek-V3": configs["deepseek-ai/DeepSeek-V3"],

        # MOE models (showing different expert counts and architectures)
        "mistralai/Mixtral-8x7B-Instruct-v0.1": configs["mistralai/Mixtral-8x7B-Instruct-v0.1"],
        "mistralai/Mixtral-8x22B-v0.1": configs["mistralai/Mixtral-8x22B-v0.1"],
        "microsoft/Phi-3.5-MoE-instruct": configs["microsoft/Phi-3.5-MoE-instruct"],

        # Advanced architectures
        "Qwen/Qwen3-Next-80B-A3B-Instruct": configs["Qwen/Qwen3-Next-80B-A3B-Instruct"],
        "Qwen/Qwen3-Omni-30B-A3B-Instruct": configs["Qwen/Qwen3-Omni-30B-A3B-Instruct"],
    }
    return selected


def format_memory_size(size_gib):
    """
    Format memory size in appropriate units (MiB, GiB, TiB, PiB)

    Args:
        size_gib: Size in GiB

    Returns:
        Formatted string with appropriate unit
    """
    if size_gib < 1:
        return f"{size_gib*1024:.1f} MiB"
    elif size_gib < 1024:
        return f"{size_gib:.1f} GiB"
    elif size_gib < 1024 * 1024:
        return f"{size_gib/1024:.2f} TiB"
    else:
        return f"{size_gib/(1024*1024):.2f} PiB"


def format_yaxis_memory(value, _):
    """
    Format y-axis tick labels with human-readable memory units
    For use with matplotlib FuncFormatter

    Args:
        value: Numeric value in GiB
        _: Unused tick position

    Returns:
        Formatted string for axis label
    """
    if value < 1:
        return f"{value*1024:.0f}M"
    elif value < 10:
        return f"{value:.1f}G"
    elif value < 1024:
        return f"{value:.0f}G"
    else:
        return f"{value/1024:.1f}T"


def get_model_color(model_name):
    """
    Get visualization color for a model based on its family

    Args:
        model_name: Full model name

    Returns:
        Hex color code
    """
    name_lower = model_name.lower()

    if "llama" in name_lower:
        return MODEL_FAMILY_COLORS["llama"]
    elif "mixtral" in name_lower:
        return MODEL_FAMILY_COLORS["mixtral"]
    elif "mistral" in name_lower:
        return MODEL_FAMILY_COLORS["mistral"]
    elif "qwen3-next" in name_lower or "qwen3-omni" in name_lower:
        return MODEL_FAMILY_COLORS["qwen3"]
    elif "qwen2.5" in name_lower:
        return MODEL_FAMILY_COLORS["qwen2"]
    elif "qwen" in name_lower:
        return MODEL_FAMILY_COLORS["qwen"]
    elif "deepseek" in name_lower:
        return MODEL_FAMILY_COLORS["deepseek"]
    elif "gemma" in name_lower:
        return MODEL_FAMILY_COLORS["gemma"]
    elif "phi" in name_lower:
        return MODEL_FAMILY_COLORS["phi"]

    return MODEL_FAMILY_COLORS["default"]