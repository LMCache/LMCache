# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import json

# Third Party
from transformers import AutoConfig


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Fetch model configuration using AutoConfig."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the model to fetch configuration for.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Load model configuration using AutoConfig
    try:
        config = AutoConfig.from_pretrained(args.model)
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            config = get_text_config(decoder=True)

        # Prepare configuration data in a dictionary format
        config_data = {
            "hidden_size": getattr(config, "hidden_size", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        }

        if args.model == "deepseek-ai/DeepSeek-V3":
            config_data["kv_lora_rank"] = getattr(config, "kv_lora_rank", None)
            config_data["qk_rope_head_dim"] = getattr(config, "qk_rope_head_dim", None)

        # Check for GLM4 models
        if ("zai-org/glm-4." in args.model.lower()):
            config_data["head_dim"] = getattr(config, "head_dim", None)

        # Check for Qwen3 / Qwen3.5 / Qwen3.6 models (fuzzy matching)
        if args.model.lower().startswith("qwen/qwen3"):
            config_data["head_dim"] = getattr(config, "head_dim", None)
            full_attention_interval = getattr(config, "full_attention_interval", None)
            if full_attention_interval is not None:
                config_data["full_attention_interval"] = full_attention_interval

        # Convert to JSON and print
        string = json.dumps(config_data, indent=4)

        print("\033[32m" + "Model configuration for " + args.model + ":\n" + "\033[0m")

        print(f'"{args.model}": {string}\n')

        print(
            "\033[32mPlease copy the above JSON to the 'modelconfig.json'"
            "and create a new PR\033[0m"
        )

    except Exception as e:
        # Print error message in JSON format
        error_data = {"error": str(e)}
        print(json.dumps(error_data, indent=4))


if __name__ == "__main__":
    main()
