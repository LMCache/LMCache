import argparse
import json

from transformers import AutoConfig


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Fetch model configuration using AutoConfig.")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="The name of the model to fetch configuration for.")

    # Parse arguments
    args = parser.parse_args()

    # Load model configuration using AutoConfig
    try:
        config = AutoConfig.from_pretrained(args.model)

        # Prepare configuration data in a dictionary format
        config_data = {
            "hidden_size": getattr(config, 'hidden_size', None),
            "num_attention_heads": getattr(config, 'num_attention_heads',
                                           None),
            "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
            "num_key_value_heads": getattr(config, 'num_key_value_heads', None)
        }

        # Convert to JSON and print
        string = json.dumps(config_data, indent=4)

        print("\033[32m" + "Model configuration for " + \
                args.model + ":\n" + "\033[0m")

        print(f'"{args.model}": {string}\n')

        print("\033[32mPlease copy the above JSON to the 'modelconfig.json'"
              "and create a new PR\033[0m")

    except Exception as e:
        # Print error message in JSON format
        error_data = {"error": str(e)}
        print(json.dumps(error_data, indent=4))


if __name__ == "__main__":
    main()
