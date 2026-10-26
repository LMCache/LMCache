Producer-declared KV tensor layouts
===================================

This integration targets the registration hook proposed in
`vLLM #63699 <https://github.com/vllm-project/vllm/pull/63699>`_.
Use a vLLM build containing that API to exercise it; older versions keep the
existing tensor-only registration and format detection. No LMCache
configuration switch enables producer declarations on an older engine.

What the engine provides
------------------------

A producer supplies one axis declaration for each described layer tensor to
``register_kv_caches_with_layouts(kv_caches, layouts)``. For example, RBLN can
register either of these contiguous layouts:

.. code-block:: text

   [BLOCK, KV, HEAD, SINGLETON, STATE, HEAD_DIM]   # blocks first
   [KV, BLOCK, HEAD, SINGLETON, STATE, HEAD_DIM]   # K/V first

With two blocks, both shapes start with ``[2, 2]``. Shape-based detection cannot
distinguish them; the declared roles can. ``KV`` is an explicit K-at-0/V-at-1
selector of size 2. MLA's single opaque plane has no such selector.
``HEAD`` may have size 1, and ``SINGLETON`` is a structural axis of size 1.
``CONTENT`` is opaque; its packing, quantization and recurrent-state granularity
still require the cache spec or the existing encoding-specific support.

How LMCache uses it
-------------------

.. code-block:: text

   engine tensor + axes
       -> connector registration
       -> per-layer layout hints
          -> in-process GPU connector
          -> MP worker -> registration wire -> server cache context
       -> zero-copy normalization of tensor and axes together
       -> existing transfer format and kernels

Declared axes take precedence over ambient ``NHD``/``HND`` hints. Each layer
keeps its own declaration, including layers with identical shapes but different
roles. Unspecified layers keep legacy detection. The MP path retains declarations
for re-registration and preserves the described tensor view across IPC.

The supported declared forms are the existing single-tensor split K/V layouts,
RBLN's structural-singleton layouts, and existing MLA/opaque CONTENT planes.
This does not enable arbitrary strided tensors or tuple component layouts.
Opaque Mamba/subpage re-views that replace a declared tensor fail explicitly;
they need a spec-aware conversion before their new views can be described.
Unsupported declarations also fail instead of falling back to a shape guess.
Handle-based MP transfer of declared views currently supports CPU SHM and
CUDA wrappers. Other handle implementations fail explicitly; use engine-driven
transfer when supported by the device. Hardware inference validation is
separate from CPU tests of tensor views.
