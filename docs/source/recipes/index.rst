.. _recipes:

Recipes
=======

This section lists model architectures that have been validated end-to-end with
LMCache, with a recipe page per architecture covering only the LMCache-specific
configuration that diverges from defaults.

Engine-side documentation (how to serve the model itself) lives with the
serving engine. Recipe pages link out rather than duplicate.

Recipe page contents
--------------------

Each recipe page is intentionally minimal:

- **Validated models** -- exact HF repo IDs that have been tested.
- **Engine tabs** -- one tab per serving engine (vLLM, SGLang, TRT-LLM). Each
  tab links to the engine's own documentation for the model and shows the
  exact ``lmcache server`` and engine launch commands. Tabs for engines that
  are not yet validated state so explicitly.
- **CacheBlend support** -- validation status (may be empty).
- **Compression support** -- table of compression methods (CacheGen, etc.)
  with per-method validation status. Extensible: new methods get a row.
- **Caveats** -- known limitations, if any.

For the generic LMCache + engine wiring (ports, remote hosts, in-process mode,
sending a first request), see :doc:`../getting_started/quickstart` and
:doc:`../mp/quickstart`. Recipes assume those pages as a prerequisite.

Supported architectures
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 30 10 10 10 12

   * - Architecture
     - Example HF model
     - vLLM
     - SGLang
     - TRT-LLM
     - Recipe
   * - ``MiniMaxM2ForCausalLM``
     - ``MiniMaxAI/MiniMax-M2``
     - ✓
     - —
     - —
     - :doc:`minimax_m2`
   * - ``Gemma4ForConditionalGeneration``
     - ``google/gemma-4-31B-it``
     - ✓
     - —
     - —
     - :doc:`gemma4`
   * - ``MistralForCausalLM``
     - ``mistralai/Devstral-2-123B-Instruct-2512``
     - ✓
     - —
     - —
     - :doc:`devstral`
   * - ``GptOssForCausalLM``
     - ``openai/gpt-oss-120b``
     - ✓
     - —
     - —
     - :doc:`gpt_oss`
   * - ``Qwen3MoeForCausalLM``
     - ``Qwen/Qwen3-235B-A22B``
     - ✓
     - —
     - —
     - :doc:`qwen3`

Legend: ``✓`` validated, ``—`` not validated.

Contributing a recipe
---------------------

To add a new architecture:

1. Copy an existing page (e.g. ``minimax_m2.rst``) to
   ``recipes/<architecture_snake_case>.rst``.
2. Fill in **Validated models**, **Engines**, **LMCache configuration**, and
   **Caveats**. Keep each section terse -- if a field has nothing to say, say
   so in one line rather than padding it.
3. Add a row to the table above and an entry to the hidden toctree below.

.. toctree::
   :hidden:
   :maxdepth: 1

   minimax_m2
   gemma4
   devstral
   gpt_oss
   qwen3
