#!/usr/bin/env python3
"""
Generate KV Cache Memory Requirements Table
Produces a formatted text table showing memory requirements across different models and context lengths
"""

import os

# Import shared utilities
from kv_cache_utils import (
    load_model_configs,
    calculate_kv_cache_size,
    select_representative_models,
    format_memory_size,
    TABLE_WIDTH,
    TABLE_SEPARATOR,
    ROW_SEPARATOR,
    MODEL_COLUMN_WIDTH,
    VALUE_COLUMN_WIDTH,
)




def generate_memory_table(output_file="kv_cache_visualizations/kv_cache_table.txt"):
    """
    Generate a detailed table with KV cache memory requirements
    Uses the KV cache calculator logic to compute accurate memory sizes
    """
    # Load configurations
    configs = load_model_configs()
    models = select_representative_models(configs)

    # Define context lengths to include in table
    context_lengths = [
        4_000,
        8_000,
        16_000,
        32_000,
        64_000,
        128_000,
        256_000,
        512_000,
        1_000_000,
    ]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        # Write header
        f.write("KV Cache Memory Requirements - float16 precision\n")
        f.write(TABLE_SEPARATOR + "\n\n")

        # Create header row with context lengths
        header = "Model".ljust(MODEL_COLUMN_WIDTH)
        for ctx in context_lengths:
            if ctx >= 1_000_000:
                header += f"{ctx//1_000_000}M".rjust(12)
            else:
                header += f"{ctx//1000}K".rjust(VALUE_COLUMN_WIDTH)
        f.write(header + "\n")
        f.write(ROW_SEPARATOR + "\n")

        # Generate rows for each model
        for model_name, config in models.items():
            short_name = model_name.split("/")[-1][: MODEL_COLUMN_WIDTH - 2]
            row = short_name.ljust(MODEL_COLUMN_WIDTH)

            for ctx in context_lengths:
                # Calculate KV cache size using the calculator logic
                size_gib = calculate_kv_cache_size(config, ctx, "float16", model_name)
                # Format with appropriate units
                formatted_size = format_memory_size(size_gib)
                row += formatted_size.rjust(VALUE_COLUMN_WIDTH)

            f.write(row + "\n")

        # Write footer with legend
        f.write("\n" + TABLE_SEPARATOR + "\n")
        f.write("\nLegend:\n")
        f.write(
            "- Memory sizes shown in MiB (Mebibytes), GiB (Gibibytes), and TiB (Tebibytes)\n"
        )
        f.write("- 1 GiB = 1,024 MiB; 1 TiB = 1,024 GiB\n")
        f.write("- Calculations based on float16 precision (2 bytes per parameter)\n")
        f.write(
            "- DeepSeek models use KV-LoRA optimization (reduced memory footprint)\n"
        )
        f.write("\nGenerated using KV cache calculator formulas:\n")
        f.write(
            "- Standard models: 2 × layers × tokens × kv_heads × (hidden_size/attention_heads)\n"
        )
        f.write(
            "- DeepSeek models: layers × tokens × (kv_lora_rank + qk_rope_head_dim)\n"
        )
        f.write("- Qwen3 models: 2 × layers × tokens × kv_heads × head_dim\n")

    return output_file


def main():
    """Main function to generate the memory requirements table"""
    print("Generating KV Cache Memory Requirements Table...")

    output_file = generate_memory_table()
    print(f"✓ Created: {output_file}")

    # Display summary statistics (reuse loaded data from generate_memory_table)
    configs = load_model_configs()
    models = select_representative_models(configs)

    print("\nMemory requirements at 1M context (float16):")
    for model_name, config in models.items():
        size_gib = calculate_kv_cache_size(config, 1_000_000, "float16", model_name)
        short_name = model_name.split("/")[-1]
        formatted = format_memory_size(size_gib)
        print(f"  {short_name:35} {formatted:>10}")

    print("\nTable generated successfully using KV cache calculator formulas")


if __name__ == "__main__":
    main()
