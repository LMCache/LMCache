# SPDX-License-Identifier: Apache-2.0
"""
Bigtable key encoder for translating ObjectKeys to row keys.
"""

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.key_codec import format_storage_key


class BigtableL2KeyEncoder:
    """Encoder to translate ObjectKeys into Bigtable row keys using a template."""

    def __init__(
        self,
        template: str = "{hash_prefix}@{model}@{rank}@{group}@{hash}@{salt}",
        layer_group_size: int = 0,
    ):
        self.template = template
        self.layer_group_size = layer_group_size

    def encode_row_key(self, key: ObjectKey) -> bytes:
        """Translates a logical ObjectKey into a physical Bigtable row key.

        Formats the row key using the configured template, injecting the model
        name, rank, group, chunk hash, and optional cache salt.

        Args:
            key: The ObjectKey to encode.

        Returns:
            The formatted row key as bytes.
        """
        suffix = f"@lg{self.layer_group_size}" if self.layer_group_size > 0 else ""
        row_key = format_storage_key(key, self.template, model_suffix=suffix)
        if hasattr(key, "layer_id"):
            row_key += f"@layer_{key.layer_id}"
        return row_key.encode("utf-8")
