// SPDX-License-Identifier: Apache-2.0

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::{hash_map::DefaultHasher, BTreeMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

#[derive(PartialEq, Eq, Hash)]
struct KeyContext {
    model_name: String,
    kv_rank: i64,
    object_group_id: i64,
    cache_salt: String,
    tags: BTreeMap<String, String>,
}

impl KeyContext {
    fn validate(&self) -> PyResult<()> {
        if self.model_name.contains('@') {
            return Err(PyValueError::new_err("model_name must not contain '@'"));
        }
        if self.object_group_id < 0 {
            return Err(PyValueError::new_err("object_group_id must be >= 0"));
        }
        if self.cache_salt.contains(['@', '/', '\\', '\0']) {
            return Err(PyValueError::new_err(
                "cache_salt contains a reserved character",
            ));
        }
        if self.cache_salt.chars().count() > 128 {
            return Err(PyValueError::new_err("cache_salt exceeds max length 128"));
        }
        if self.tags.keys().any(String::is_empty) {
            return Err(PyValueError::new_err("tag names must not be empty"));
        }
        Ok(())
    }
}

struct KeyBatchData {
    context: KeyContext,
    packed_hashes: Vec<u8>,
    hash_size: usize,
    hashes: Vec<u64>,
}

impl KeyBatchData {
    fn new(context: KeyContext, packed_hashes: Vec<u8>, hash_size: usize, count: usize) -> Self {
        let mut hasher = DefaultHasher::new();
        context.hash(&mut hasher);
        let context_hash = hasher.finish();
        let hashes = (0..count)
            .map(|index| {
                let mut hasher = DefaultHasher::new();
                context_hash.hash(&mut hasher);
                packed_hashes[index * hash_size..(index + 1) * hash_size].hash(&mut hasher);
                hasher.finish()
            })
            .collect();
        Self {
            context,
            packed_hashes,
            hash_size,
            hashes,
        }
    }

    fn chunk_hash(&self, index: usize) -> &[u8] {
        &self.packed_hashes[index * self.hash_size..(index + 1) * self.hash_size]
    }
}

#[pyclass(frozen, module = "lmcache_rust_core")]
#[derive(Clone)]
struct ObjectKey {
    data: Arc<KeyBatchData>,
    index: usize,
}

impl PartialEq for ObjectKey {
    fn eq(&self, other: &Self) -> bool {
        (Arc::ptr_eq(&self.data, &other.data) || self.data.context == other.data.context)
            && self.data.chunk_hash(self.index) == other.data.chunk_hash(other.index)
    }
}

impl Eq for ObjectKey {}

impl Hash for ObjectKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.data.hashes[self.index]);
    }
}

#[pymethods]
impl ObjectKey {
    #[new]
    #[pyo3(signature = (chunk_hash, model_name, kv_rank, object_group_id=0, cache_salt=String::new(), tags=None))]
    fn new(
        chunk_hash: Vec<u8>,
        model_name: String,
        kv_rank: i64,
        object_group_id: i64,
        cache_salt: String,
        tags: Option<BTreeMap<String, String>>,
    ) -> PyResult<Self> {
        let context = KeyContext {
            model_name,
            kv_rank,
            object_group_id,
            cache_salt,
            tags: tags.unwrap_or_default(),
        };
        context.validate()?;
        let hash_size = chunk_hash.len();
        Ok(Self {
            data: Arc::new(KeyBatchData::new(context, chunk_hash, hash_size, 1)),
            index: 0,
        })
    }

    #[getter]
    fn chunk_hash<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.data.chunk_hash(self.index))
    }

    #[getter]
    fn model_name(&self) -> &str {
        &self.data.context.model_name
    }

    #[getter]
    fn kv_rank(&self) -> i64 {
        self.data.context.kv_rank
    }

    #[getter]
    fn object_group_id(&self) -> i64 {
        self.data.context.object_group_id
    }

    #[getter]
    fn cache_salt(&self) -> &str {
        &self.data.context.cache_salt
    }

    #[getter]
    fn tags(&self) -> BTreeMap<String, String> {
        self.data.context.tags.clone()
    }

    fn __hash__(&self) -> u64 {
        self.data.hashes[self.index]
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

#[pyclass(frozen, sequence, module = "lmcache_rust_core")]
struct ObjectKeyBatch {
    data: Arc<KeyBatchData>,
}

#[pymethods]
impl ObjectKeyBatch {
    #[new]
    #[pyo3(signature = (packed_hashes, hash_size, model_name, kv_rank, object_group_id=0, cache_salt=String::new(), tags=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        packed_hashes: &[u8],
        hash_size: usize,
        model_name: String,
        kv_rank: i64,
        object_group_id: i64,
        cache_salt: String,
        tags: Option<BTreeMap<String, String>>,
    ) -> PyResult<Self> {
        if hash_size == 0 || packed_hashes.len() % hash_size != 0 {
            return Err(PyValueError::new_err(
                "packed_hashes must contain whole, nonzero-width hashes",
            ));
        }
        let context = KeyContext {
            model_name,
            kv_rank,
            object_group_id,
            cache_salt,
            tags: tags.unwrap_or_default(),
        };
        context.validate()?;
        let count = packed_hashes.len() / hash_size;
        let packed = packed_hashes.to_vec();
        let data = py.allow_threads(move || KeyBatchData::new(context, packed, hash_size, count));
        Ok(Self {
            data: Arc::new(data),
        })
    }

    fn __len__(&self) -> usize {
        self.data.hashes.len()
    }

    fn __getitem__(&self, mut index: isize) -> PyResult<ObjectKey> {
        if index < 0 {
            index += self.data.hashes.len() as isize;
        }
        if index < 0 || index as usize >= self.data.hashes.len() {
            return Err(PyIndexError::new_err("key index out of range"));
        }
        Ok(ObjectKey {
            data: Arc::clone(&self.data),
            index: index as usize,
        })
    }

    #[getter]
    fn packed_hashes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.data.packed_hashes)
    }
}

#[pyclass(frozen, module = "lmcache_rust_core")]
#[derive(Default)]
struct ObjectKeySet {
    keys: RwLock<HashSet<ObjectKey>>,
}

#[pymethods]
impl ObjectKeySet {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn insert(&self, py: Python<'_>, batch: &ObjectKeyBatch) -> usize {
        let data = Arc::clone(&batch.data);
        py.allow_threads(move || {
            let mut keys = self.keys.write().unwrap();
            let before = keys.len();
            keys.extend((0..data.hashes.len()).map(|index| ObjectKey {
                data: Arc::clone(&data),
                index,
            }));
            keys.len() - before
        })
    }

    fn contains(&self, py: Python<'_>, batch: &ObjectKeyBatch) -> Vec<bool> {
        let data = Arc::clone(&batch.data);
        py.allow_threads(move || {
            let keys = self.keys.read().unwrap();
            (0..data.hashes.len())
                .map(|index| {
                    keys.contains(&ObjectKey {
                        data: Arc::clone(&data),
                        index,
                    })
                })
                .collect()
        })
    }

    fn remove(&self, py: Python<'_>, batch: &ObjectKeyBatch) -> usize {
        let data = Arc::clone(&batch.data);
        py.allow_threads(move || {
            let mut keys = self.keys.write().unwrap();
            (0..data.hashes.len())
                .filter(|&index| {
                    keys.remove(&ObjectKey {
                        data: Arc::clone(&data),
                        index,
                    })
                })
                .count()
        })
    }

    fn __len__(&self) -> usize {
        self.keys.read().unwrap().len()
    }
}

#[pymodule]
fn lmcache_rust_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ObjectKey>()?;
    module.add_class::<ObjectKeyBatch>()?;
    module.add_class::<ObjectKeySet>()?;
    Ok(())
}
