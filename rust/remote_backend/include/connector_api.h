/* SPDX-License-Identifier: Apache-2.0 */

/*
 * LMCache Remote Connector C ABI.
 *
 * Third-party connectors (C, C++, or Rust cdylib) must
 * export every function declared below.  The remote backend
 * loads the shared library at runtime via dlopen and
 * resolves these symbols.
 *
 * Lifecycle:
 *   1. connector_create(json_cfg, json_cfg_len) -> handle
 *   2. connector_exists / put / get / remove / list_keys
 *   3. connector_destroy(handle)
 */

#ifndef LMCACHE_CONNECTOR_API_H
#define LMCACHE_CONNECTOR_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle returned by connector_create. */
typedef void* ConnectorHandle;

/*
 * Create a new connector instance.
 *
 * `config_json` is a UTF-8 JSON string (not NUL-terminated)
 * of length `config_json_len` containing all configuration
 * key-value pairs from extra_config.
 *
 * Returns an opaque handle, or NULL on failure.
 */
ConnectorHandle connector_create(const char* config_json,
                                 size_t config_json_len);

/* Destroy a connector instance. */
void connector_destroy(ConnectorHandle handle);

/*
 * Check whether `key` (UTF-8, NUL-terminated) exists.
 * Returns 1 if exists, 0 otherwise.
 */
int32_t connector_exists(ConnectorHandle handle, const char* key);

/*
 * Write `data_len` bytes from `data` for `key`.
 * Returns 0 on success, non-zero on failure.
 */
int32_t connector_put(ConnectorHandle handle, const char* key,
                      const uint8_t* data, size_t data_len);

/*
 * Read data for `key` into `out_buf` of capacity
 * `out_cap`.  On success, writes the number of bytes
 * actually read into `*out_len` and returns 0.
 *
 * Returns  1 if the key does not exist (not-found).
 * Returns -1 on I/O error.
 */
int32_t connector_get(ConnectorHandle handle, const char* key, uint8_t* out_buf,
                      size_t out_cap, size_t* out_len);

/*
 * Remove the entry for `key`.
 * Returns 1 if removed, 0 if not found, -1 on error.
 */
int32_t connector_remove(ConnectorHandle handle, const char* key);

/*
 * Get the data size in bytes for `key`.
 * On success writes to `*out_size` and returns 0.
 * Returns 1 if not found, -1 on error.
 */
int32_t connector_file_size(ConnectorHandle handle, const char* key,
                            uint64_t* out_size);

/*
 * List all keys.  The connector writes keys as a single block of
 * newline-separated UTF-8 strings into `out_buf`.
 * The total capacity of `out_buf` is `out_cap`.
 * It writes the total number of bytes used into `*out_len`.
 *
 * Returns 0 on success, -1 on error.
 */
int32_t connector_list_keys(ConnectorHandle handle, char* out_buf,
                            size_t out_cap, size_t* out_len);

#ifdef __cplusplus
}
#endif

#endif /* LMCACHE_CONNECTOR_API_H */
