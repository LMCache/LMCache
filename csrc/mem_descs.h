#pragma once

#include <cstddef>

struct PageBufferShapeDesc {
  size_t kv_dim;  // 2 for normal and 1 for mla
  size_t num_pages;
  size_t page_size;
  size_t hidden_dim;
};

struct ObjBufferShapeDesc {
  size_t kv_dim;  // 2 for normal and 1 for mla
  size_t num_layers;
  size_t chunk_size;
  size_t hidden_dim;
};
