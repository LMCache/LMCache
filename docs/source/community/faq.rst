Frequently asked questions
==========================

This page collects questions the LMCache community asks most often, together
with the answers given by project maintainers. The community maintains this
page: questions arrive through Slack, GitHub, and community channels such as
Xiaohongshu, and are collected here.

The English page is the source of this content. Translations are maintained as
gettext catalogs under ``docs/source/locale/`` and are published alongside the
English page. See :doc:`../developer_guide/contributing` for how to contribute
a question, an answer, or a translation.

.. note::
   Some answers below are marked as needing maintainer confirmation. They
   record what was said in a community discussion, but they are incomplete,
   undated, or not yet checked against the current code. Treat these answers as
   provisional. Where a reference link is given, use the reference instead.

Project and roadmap
-------------------

Why was Python chosen as LMCache's implementation language, and is there a plan to switch to another language?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The AI ecosystem is still mostly Python. vLLM and SGLang are both written in
Python. Python is also quick to learn and friendly to newcomers, so LMCache
chose Python as well.

Switching is possible, and some people have already rewritten the LMCache
server side in C++ or Rust. However, the core of LMCache will remain mostly
Python, to stay consistent with the surrounding ecosystem.

*Answered by* `@maobaolong <https://github.com/maobaolong>`__. *First asked on Xiaohongshu.*

Which models does LMCache support?
++++++++++++++++++++++++++++++++++

Officially supported models are listed in the
`LMCache recipes <https://docs.lmcache.ai/recipes/index.html>`__. That page is
updated as support for each new model is added.

*Answered by* `@maobaolong <https://github.com/maobaolong>`__.

Deploying on Kubernetes
-----------------------

Why does LMCache need its own Kubernetes operator when projects such as llm-d exist?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

LMCache MP (multiprocess) mode runs as a DaemonSet on Kubernetes, which needs
extra management logic. LMCache also has a coordinator, plugins, and other
components that all need an extra control plane to operate them. For these
reasons, the project currently maintains its own separate Kubernetes operator.

*Answered by* `@ApostaC <https://github.com/ApostaC>`__.

Does deploying LMCache on Kubernetes require special permissions?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Yes. This is known behavior today, and these permissions exist mainly to
support the cross-GPU KV Cache sharing feature. If you do not grant these
permissions, deployment fails. The project plans to offer an approach that does
not require special permissions in a future release.

Storage and hardware acceleration
---------------------------------

Does LMCache support GDS and GDR?
+++++++++++++++++++++++++++++++++

LMCache supports GDS (GPUDirect Storage) today, and the community recently
merged a pull request that adds uGDS support. GDR (GPUDirect RDMA) is on the
roadmap.

What performance gains come from enabling GDS?
++++++++++++++++++++++++++++++++++++++++++++++

.. note::
   **Needs maintainer confirmation.** The community discussion pointed to the
   uGDS paper as a source of measurements, because those tests were run against
   LMCache. No figures or citation were recorded. This answer needs a specific
   reference and specific numbers before it is useful.

Does multi-hardware support require hardware features such as IPC events?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This depends on the mode. EngineDriven mode does not require IPC
(inter-process communication) events. LMCacheDriven mode does require them.
Moore Threads MUSA currently supports LMCacheDriven mode.

CacheBlend
----------

Is CacheBlend only in a commercial version, and when will it be open-sourced?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The core team is currently focused on support for the latest vLLM. The code is
not yet ready to be open-sourced.

.. note::
   **Needs maintainer confirmation.** The original answer said that an
   open-source version was expected "this quarter", but that statement was not
   dated, so the timeline cannot be relied on. A current target date is needed.

*Answered by* `@ApostaC <https://github.com/ApostaC>`__.

Does CacheBlend support linear attention models?
++++++++++++++++++++++++++++++++++++++++++++++++

.. note::
   **Needs maintainer confirmation.** The community raised this question, but
   no answer was recorded, apart from a reference to an article on Xiaohongshu.
   That article was not identified.

KV Cache behavior and control
-----------------------------

A conversation still has its context a week later. Does that mean KV Cache is being used?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Not necessarily. The agent client stores the context text and sends it with
each request. That behavior is unrelated to KV Cache.

LMCache can provide features such as pin, which forces a KV Cache entry to stay
in the system. Pinning is what guarantees that a given conversation hits the
KV Cache.

*Answered by* `@ApostaC <https://github.com/ApostaC>`__.

Can LMCache control KV Cache based on the semantics of a request?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This question came from a user who wanted sub-agent conversations excluded from
the KV Cache, keeping only the main conversation's KV Cache.

inProcess mode previously supported reading an extra parameter on a request
that indicated whether caching should be disabled. MP mode has not yet ported
this feature, but it could be added with little effort.

In MP mode, the MP coordinator module already supports lookup, prefetch, pin,
and delete. It also supports re-synchronization after a restart, checkpointing,
and related operations.
