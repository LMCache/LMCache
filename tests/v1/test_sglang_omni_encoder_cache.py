# SPDX-License-Identifier: Apache-2.0
"""Public encoder-cache contract tests; no vLLM/SGLang/GPU import required."""

# Standard
import unittest

# Third Party
import torch

# First Party
from lmcache.integration.sglang_omni.encoder_cache import SGLangOmniEncoderCache


class MemoryEC:
    """Test double for the EC tensor interface, with observable allocation failure."""

    def __init__(self) -> None:
        self.data: dict[str, torch.Tensor] = {}
        self.accept = True
        self.closes = 0

    def put(self, key: str, tensor: torch.Tensor) -> bool:
        if not self.accept:
            return False
        self.data[key] = tensor.clone()
        return True

    def get(self, key: str, device: str) -> torch.Tensor | None:
        tensor = self.data.get(key)
        return None if tensor is None else tensor.clone().to(device)

    def close(self) -> None:
        self.closes += 1


class EncoderCacheTests(unittest.TestCase):
    def test_audio_roundtrip_and_no_alias(self) -> None:
        cache = SGLangOmniEncoderCache(MemoryEC(), "model@rev/audio/bf16/processor-v1")
        data = {
            "audio_embeds": torch.arange(24, dtype=torch.bfloat16).reshape(6, 4),
            "audio_feature_lengths": torch.tensor([20, 12]),
            "audio_output_lengths": torch.tensor([4, 2]),
        }
        self.assertIsNone(cache.get("input"))
        self.assertTrue(cache.put("input", data))
        result = cache.get("input")
        for key, tensor in data.items():
            self.assertEqual(result[key].dtype, tensor.dtype)
            self.assertTrue(torch.equal(result[key], tensor))
        result["audio_embeds"].zero_()
        self.assertTrue(
            torch.equal(cache.get("input")["audio_embeds"], data["audio_embeds"])
        )

    def test_image_deepstack_and_metadata_are_one_entry(self) -> None:
        ec = MemoryEC()
        cache = SGLangOmniEncoderCache(ec, "model@rev/image")
        x = torch.arange(48, dtype=torch.float16).reshape(6, 8)[:, ::2]
        data = {
            "image_embeds": x,
            "deepstack_visual_embeds_image": [x + 1, x + 2],
            "image_grid_thw": torch.tensor([[1, 4, 6]]),
            "metadata": (None, "image", True, 3, 0.5),
            "empty": torch.empty(0, 4),
        }
        cache.put("same", data)
        result = cache.get("same")
        self.assertEqual(len(ec.data), 1)
        self.assertIsInstance(result["metadata"], tuple)
        self.assertEqual(result["metadata"], data["metadata"])
        self.assertTrue(torch.equal(result["deepstack_visual_embeds_image"][1], x + 2))
        self.assertEqual(result["empty"].shape, (0, 4))
        ec.data.clear()
        self.assertIsNone(cache.get("same"))

    def test_stage_revision_dtype_and_processor_isolation(self) -> None:
        ec = MemoryEC()
        a = SGLangOmniEncoderCache(ec, "rev1/image/bf16/processor1")
        a.put("same-media", torch.ones(2, 4))
        for namespace in [
            "rev1/audio/bf16/processor1",
            "rev2/image/bf16/processor1",
            "rev1/image/fp16/processor1",
            "rev1/image/bf16/processor2",
        ]:
            self.assertIsNone(SGLangOmniEncoderCache(ec, namespace).get("same-media"))

    def test_prepared_inputs_determine_key(self) -> None:
        cache = SGLangOmniEncoderCache(MemoryEC(), "revision")
        x = torch.arange(4).reshape(2, 2)
        a = cache.key_for_inputs({"pixels": x, "grid": [1, 2, 2]})
        self.assertEqual(
            a, cache.key_for_inputs({"grid": [1, 2, 2], "pixels": x.clone()})
        )
        for data in [
            {"pixels": x + 1, "grid": [1, 2, 2]},
            {"pixels": x.float(), "grid": [1, 2, 2]},
            {"pixels": x.reshape(4), "grid": [1, 2, 2]},
            {"pixels": x, "grid": [2, 1, 2]},
        ]:
            self.assertNotEqual(a, cache.key_for_inputs(data))

    def test_bypass_failed_store_and_shutdown(self) -> None:
        ec = MemoryEC()
        cache = SGLangOmniEncoderCache(ec, "revision")
        self.assertFalse(cache.put(None, object()))
        self.assertIsNone(cache.get(None))
        ec.accept = False
        self.assertFalse(cache.put("key", torch.ones(1, 1)))
        self.assertIsNone(cache.get("key"))
        cache.close()
        cache.close()
        self.assertEqual(ec.closes, 1)
        with self.assertRaises(RuntimeError):
            cache.get("key")
        with self.assertRaises(RuntimeError):
            cache.put("key", torch.ones(1, 1))

    def test_unsupported_objects_and_nonfinite_metadata(self) -> None:
        cache = SGLangOmniEncoderCache(MemoryEC(), "revision")
        with self.assertRaises(TypeError):
            cache.put("key", {"object": object()})
        with self.assertRaises(ValueError):
            cache.put("key", {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
