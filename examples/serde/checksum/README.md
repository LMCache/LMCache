# Checksum Serde Benchmark

This example measures the in-memory transform cost of the checksum serde.
It compares the shipped xxh3_64 implementation with a byte-copy baseline and
AES-GCM through the same Serializer/Deserializer interfaces used by the
production path.

The results are hardware-dependent and are intended for relative comparison,
not as fixed performance guarantees.

## Run

Run the benchmark with the default 128 KB, 1 MB, and 16 MB payload sizes:

~~~bash
python examples/serde/checksum/bench_checksum_vs_aesgcm.py
~~~

Select payload sizes explicitly; values are bytes:

~~~bash
python examples/serde/checksum/bench_checksum_vs_aesgcm.py \
    --chunk-sizes 131072 1048576 16777216
~~~

The output reports serialization, deserialization, and combined round-trip
latency for each case.

## Verification

The serde unit and filesystem integration tests can be run with:

~~~bash
python -m pytest tests/v1/distributed/serde/test_checksum.py \
    tests/v1/distributed/serde/test_checksum_fs_e2e.py -q
~~~

The design and performance discussion is in
[docs/design/v1/distributed/serde/checksum.md](../../../docs/design/v1/distributed/serde/checksum.md).

No GPU is required for the benchmark.
