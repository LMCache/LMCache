# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[2]
CALCULATOR_DIR = REPO_ROOT / "examples" / "kv_cache_calculator"
DOCS_STATIC_DIR = REPO_ROOT / "docs" / "source" / "_static"


def test_docs_calculator_assets_use_the_canonical_example() -> None:
    """The published docs and standalone example must share the same assets."""
    for filename in ("kv_cache_calculator.html", "modelconfig.json"):
        docs_asset = DOCS_STATIC_DIR / filename
        canonical_asset = CALCULATOR_DIR / filename

        assert docs_asset.is_symlink()
        assert docs_asset.resolve() == canonical_asset.resolve()


def test_calculator_exposes_all_merged_feature_families() -> None:
    """The canonical assets retain every feature family from both old copies."""
    model_configs = json.loads(
        (CALCULATOR_DIR / "modelconfig.json").read_text(encoding="utf-8")
    )
    calculator = (CALCULATOR_DIR / "kv_cache_calculator.html").read_text(
        encoding="utf-8"
    )

    assert model_configs["tencent/Hunyuan-Large"]["cla_share_factor"] == 2
    assert "compress_ratios" in model_configs["deepseek-ai/DeepSeek-V4-Pro"]
    assert "sliding_attention_layers" in model_configs["google/gemma-4-31B-it"]
    assert "linear_attention_layers" in model_configs["Qwen/Qwen3.5-397B-A17B-FP8"]

    for required_ui_contract in (
        "function setLanguage(language)",
        "function calculateKVCache()",
        "function calculateMaxTokens()",
        "config.cla_share_factor",
        "config.compress_ratios",
        "config.sliding_attention_layers",
        "config.linear_attention_layers",
    ):
        assert required_ui_contract in calculator
