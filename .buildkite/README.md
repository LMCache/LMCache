# Comprehensive Tests

An end-to-end integration suite for LMCache & vLLM latest branch.

## Layout

- **Scripts**: `scripts/vllm-integration-tests.sh`
- **LMCache configs**: `lmcache_configs/`
- **Workload configs**: `workload_configs/`
- **Pipeline**: `pipelines/comprehensive-tests.yml`

## Prepare

1. Add your LMCache YAMLs to `lmcache_configs/` (e.g. `local_cpu.yaml`, `local_disk.yaml`).

2. Add matching workload YAMLs to `workload_configs/` **using the same filenames** (e.g. `local_cpu.yaml`, `local_disk.yaml`).

3. Add the filenames to `cases/comprehensive-cases.txt`.
