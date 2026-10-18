.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: kubernetes
   question: How does LMCache run on Kubernetes?
   aliases: Kubernetes, k8s, cloud native, LMCache Kubernetes, KV cache Kubernetes
   related: cache-aware-routing, prefix-caching, cache-hit-rate, lmcache
   tier: 5
   publish: yes
   wp_post_type: page
   wp_status: draft
   canonical: https://lmcache.ai/glossary/kubernetes
   schema_type: DefinedTermSet
   seo_title: LMCache on Kubernetes: KV Caching in a Cloud-Native Deployment
   meta_description: How a KV cache layer behaves on Kubernetes -- pods, replicas, routing, Operators, and probes -- and why replica count is a cache-hit-rate problem.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-kubernetes:

LMCache on Kubernetes
=====================

.. meta::
   :description: How a KV cache layer behaves on Kubernetes -- pods, replicas, routing, Operators, and probes -- and why replica count is a cache-hit-rate problem.
   :keywords: LMCache Kubernetes, KV cache Kubernetes, LLM inference Kubernetes, cloud native inference, cache-aware routing, vLLM Kubernetes

**Running LMCache on Kubernetes changes one thing fundamentally: a cache that was
local to a process becomes state spread across a fleet, and Kubernetes has no
native concept of which replica holds which cached data.** Everything else on this
page follows from that.

This page defines the cloud-native vocabulary that shows up in LMCache deployment
discussions, but it defines each term *in the context of operating a cache layer*
rather than in the abstract. If you want the inference-side vocabulary instead --
prefill, decode, TTFT, hit rate -- start at the :doc:`glossary index <index>`.

The short version
-----------------

A single-machine LMCache deployment is already several processes. A Kubernetes
deployment is several *replicas* of that, behind a load balancer that does not
know your cache exists. Cache hit rate, the variable every other performance
number routes through, becomes partly a routing problem the moment you scale past
one replica.

The pieces, smallest to largest
-------------------------------

.. mermaid::

   flowchart TB
       subgraph Cluster["Cluster"]
           subgraph Node1["Node (one physical or virtual machine)"]
               Pod1["Pod<br/>vLLM + LMCache"]
               Pod2["Pod<br/>vLLM + LMCache"]
           end
           subgraph Node2["Node"]
               Pod3["Pod<br/>LMCache MP server"]
           end
       end

**Container** -- a lightweight, self-contained package of one process plus
everything it needs to run: code, dependencies, system libraries, built so it
behaves identically regardless of which physical machine runs it. Lighter than a
virtual machine, because a container shares the host's operating system kernel
rather than emulating a whole separate OS. The standard analogy is a shipping
container: the contents stay consistent no matter which ship or truck carries it.

**Node** -- one physical or virtual machine that is part of the cluster, capable
of running some number of pods. For inference workloads a node is usually where
the GPUs are, which makes node boundaries the same boundaries your KV data has to
cross.

**Pod** -- Kubernetes' smallest deployable unit: usually one container,
occasionally a small group of tightly coupled containers, scheduled onto a node
together. When people casually say "an instance" or "a replica" of a service, they
usually mean one pod.

**Cluster** -- the whole collection of nodes managed together as one system.
"Fleet" is the informal version of the same idea: the group of running replicas
considered together, without implying the specific Kubernetes machinery
underneath.

Why replica count is a cache problem
------------------------------------

This is the section that matters, and it is the one most Kubernetes glossaries
have no reason to contain.

**Replica (Deployment / ReplicaSet)** -- a replica is one running copy of a
service. A Kubernetes Deployment or ReplicaSet is the object that declares "keep N
copies of this pod running at all times," and it is the literal mechanism that
creates and maintains them. Autoscaling works by having something adjust that N up
or down based on load.

**Service / load balancer** -- a stable network entry point that distributes
incoming requests across whichever pods are currently running for a deployment. It
is the routing layer: the thing deciding which replica a given request actually
reaches.

Here is the problem. Plain Kubernetes load balancing has no concept of cache
locality. It does not know, and has no way to learn, which replica might already
hold the KV data relevant to an incoming request. So in a deployment with N
replicas, a request carrying a prefix that *was* cached has roughly a 1-in-N
chance of landing where that cache lives -- and a request routed to a replica that
does not hold the matching cache is a guaranteed miss, regardless of how well
tuned that replica's own configuration is.

Scaling out therefore does something counterintuitive: it can *lower* your
effective :doc:`cache hit rate <cache-hit-rate>` even as it raises total capacity.

There are two answers, and production deployments usually want both:

- :doc:`Cache-aware routing <cache-aware-routing>` -- send requests likely to
  share a prefix to the same replica, so the cache is reachable by the request
  that needs it.
- Cross-replica sharing -- let replicas reach each other's cached data, over
  :doc:`RDMA or another fast interconnect <interconnects>`, or through a shared
  cache tier that all replicas can read. This is LMCache reaching across the pod
  boundary that Kubernetes drew.

Keeping a cache fleet healthy
-----------------------------

**Kubernetes Operator** -- a custom controller that knows how to manage one
particular piece of software's lifecycle. Where plain Kubernetes generically keeps
N pods running, an Operator understands application-specific concerns: how to
safely scale down a cache server without losing in-flight lookups, or how to roll
out a new engine version without dropping requests. It is an optional layer of
smarter automation, not something every deployment has.

**Liveness and readiness probes** -- how Kubernetes periodically asks a running
pod whether it is healthy, usually by hitting a URL like ``/healthz`` and checking
the response. A failing *liveness* probe means Kubernetes assumes the pod is stuck
and restarts it. A failing *readiness* probe means "do not route new traffic here
yet" without necessarily restarting anything -- for example while a pod is still
loading model weights, which for a large model is a meaningful window.

The readiness distinction matters more for inference than for a typical web
service. A pod that is alive but still warming has no useful cache and a cold
model; routing to it produces exactly the slow requests your
:doc:`tail latency <tail-latency>` numbers will remember.

LMCache's ``mp_coordinator`` component exposes a liveness endpoint, which is
concrete evidence that its fleet-coordination piece is built to be managed *by*
Kubernetes rather than sitting adjacent to it.

What is actually running inside a pod
-------------------------------------

Worth being precise about, because "one pod" is not "one thing."

**Process** -- one running instance of a program, with its own private memory,
sharing nothing with other processes unless explicitly wired together through a
network call, shared memory, or a message queue. A single vLLM plus LMCache
deployment typically runs several at once: under ``--tensor-parallel-size 4``,
that is four separate GPU worker processes.

**Thread** -- the thing that actually executes instructions inside a process.
Every process starts with one, and can ask the operating system for more; they all
share the same process memory. LMCache uses several deliberately: one thread
accepts incoming lookup and store requests, another waits for GPU-to-CPU
copy-completion signals, another periodically evicts old entries. On a single
thread, one slow step -- waiting on a copy to finish -- would freeze all the
others.

If a process is an office building with one shared set of filing cabinets, threads
are the employees: each doing different work, all reading and writing the same
files. That sharing is why they are fast, and why they need coordination.

**REST** -- a common convention for how two programs talk over plain HTTP:
standardized verbs (``GET``, ``POST``) against standardized URLs, so any two
programs that both speak it can exchange data without custom communication code.
Think of it as a standardized survey form -- as long as both sides agree on the
format, neither has to invent a bespoke conversation.

Frequently asked questions
--------------------------

**Does LMCache require Kubernetes?**

No. LMCache runs as a standalone daemon process and works on a single machine.
Kubernetes is one deployment target among several, and the terms on this page only
become relevant once you are running more than one replica.

**Does scaling to more replicas improve inference performance?**

It raises total capacity, but it can reduce cache hit rate at the same time,
because each additional replica is another place a cached prefix might be and only
one of them receives the next request. Cache-aware routing or cross-replica
sharing is what keeps hit rate from degrading as you scale out.

**Do I need the Operator?**

Not to run LMCache. An Operator earns its place when you need lifecycle behavior
plain Kubernetes cannot express -- draining a cache server without losing in-flight
lookups, or coordinated rollouts across a cache fleet.

**Where does the KV cache actually live in a Kubernetes deployment?**

Wherever you configure the tiers to put it: GPU memory first, then CPU memory on
the same node, then local disk, then a remote backend reachable across the
cluster. See :doc:`cache tiering <cache-tiering>` for the hierarchy and the
tradeoffs at each level.

.. seealso::

   :doc:`cache-aware-routing` | :doc:`cache-hit-rate` | :doc:`cache-tiering` | :doc:`interconnects` | :doc:`lmcache` | :doc:`index`
