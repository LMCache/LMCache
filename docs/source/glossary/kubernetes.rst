.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable; they are the
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
   meta_description: How a KV cache layer behaves on Kubernetes (pods, replicas, routing, Operators, and probes), and why replica count is a cache hit rate problem.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-kubernetes:

LMCache on Kubernetes
=====================

.. meta::
   :description: How a KV cache layer behaves on Kubernetes (pods, replicas, routing, Operators, and probes), and why replica count is a cache hit rate problem.
   :keywords: LMCache Kubernetes, KV cache Kubernetes, LLM inference Kubernetes, cloud native inference, cache-aware routing, vLLM Kubernetes

**Running LMCache on Kubernetes changes one thing fundamentally: a cache that was
local to a process becomes state spread across a fleet, and Kubernetes has no
native concept of which replica holds which cached data.** Everything else on this
page follows from that.

This page defines the cloud-native vocabulary that shows up in LMCache deployment
discussions, but it defines each term *in the context of operating a cache layer*
rather than in the abstract. If you want the inference-side vocabulary instead (prefill,
decode, TTFT, hit rate), start at the :doc:`glossary index <index>`.

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

**Container**: an isolated process packaged with everything its application needs
to run (code, dependencies, system libraries), so it behaves consistently on
different machines. A container is lighter than a virtual machine because it
shares the host's operating system kernel instead of running a whole separate
operating system. The shared kernel also means a container still depends on the
host: a GPU container, for example, uses the GPU driver installed on the host. The
standard analogy is a shipping container, whose contents stay the same no matter
which ship or truck carries it.

**Node**: one physical or virtual machine that is part of the cluster, capable
of running some number of pods. For inference workloads a node is usually where
the GPUs are, which makes node boundaries the same boundaries your KV data has to
cross.

**Pod**: Kubernetes' smallest deployable unit: usually one container,
occasionally a small group of tightly coupled containers, scheduled onto a node
together. When people casually say "an instance" or "a replica" of a service, they
usually mean one pod.

**Cluster**: a set of nodes plus the control plane that manages them as one
system. *Fleet* is the informal version of the same idea: the group of running
replicas considered together, without implying the specific Kubernetes machinery
underneath.

Why replica count is a cache problem
------------------------------------

This is the section that matters, and it is the one most Kubernetes glossaries
have no reason to contain.

**Replica (Deployment / ReplicaSet)**: one running copy of a service. In
Kubernetes, a ReplicaSet keeps a specified number of identical pods running, and a
Deployment manages ReplicaSets; the Deployment is usually where you declare "keep
N copies of this pod running." Autoscaling means adjusting that N automatically as
load changes. The Kubernetes HorizontalPodAutoscaler, for example, updates a
Deployment's replica count to match demand.

**Service / load balancer**: a stable network entry point that spreads incoming
connections across the ready pods behind a deployment. It is the routing layer,
which decides which replica a given request actually reaches.

Here is the problem. Plain Kubernetes load balancing has no concept of cache
locality. It does not know, and has no way to learn, which replica might already
hold the KV data relevant to an incoming request. So in a deployment with N replicas, a request carrying a prefix that *was*
cached has roughly a 1-in-N chance of landing where that cache lives. A request
routed to a replica that does not hold the matching cache misses on that
replica, however well tuned its configuration is, unless the replicas share
cached data (the second answer below).

Scaling out therefore does something counterintuitive: it can *lower* your
effective :doc:`cache hit rate <cache-hit-rate>` even as it raises total capacity.

There are two answers, and production deployments usually want both:

- :doc:`Cache-aware routing <cache-aware-routing>`: send requests likely to
  share a prefix to the same replica, so the cache is reachable by the request
  that needs it.
- Cross-replica sharing: let replicas reach each other's cached data, over
  :doc:`RDMA or another fast interconnect <interconnects>`, or through a shared
  cache tier that all replicas can read. This is LMCache reaching across the pod
  boundary that Kubernetes drew.

Keeping a cache fleet healthy
-----------------------------

**Kubernetes Operator**: a custom controller that knows how to manage one
particular piece of software's lifecycle. Where plain Kubernetes generically keeps
N pods running, an Operator understands application-specific concerns: how to
safely scale down a cache server without losing in-flight lookups, or how to roll
out a new engine version without dropping requests. Operators are optional, and many deployments
do not use one. LMCache ships its own Operator, which deploys and manages
multiprocess servers and the coordinator; see :doc:`/mp/operator`.

**Liveness and readiness probes**: how Kubernetes periodically asks a running
container whether it is healthy, usually by requesting a URL like ``/healthz`` and
checking the response. When a *liveness* probe keeps failing, Kubernetes assumes
the container is stuck and restarts it. When a *readiness* probe fails, Kubernetes
stops routing new traffic to the pod but does not restart it; for example, a pod
can report not ready while it is still loading model weights, which for a large
model takes a long time.

The readiness distinction matters more for inference than for a typical web
service. A pod that is alive but still warming has no useful cache and a cold
model; routing to it produces exactly the slow requests your
:doc:`tail latency <tail-latency>` numbers will remember.

LMCache's multiprocess coordinator (``mp_coordinator``) serves a ``/healthz``
endpoint, and the LMCache Operator uses that path for the coordinator's liveness
and readiness probes.

What is actually running inside a pod
-------------------------------------

This is worth being precise about, because one pod is not one thing.

**Process**: one running instance of a program, with its own private memory.
Processes exchange data only through explicit mechanisms such as network
connections, shared memory, or message queues. A single vLLM plus LMCache
deployment on one machine typically runs several at once. For example, vLLM starts
one worker process per GPU, so ``--tensor-parallel-size 4`` means four separate
GPU worker processes.

**Thread**: the thing that actually executes instructions inside a process. Every
process starts with one; a process can ask the operating system for more, and they
all share the same process memory. LMCache uses several deliberately: one thread
accepts incoming lookup and store requests, another handles the completion signals
from GPU memory copies, and another periodically evicts old entries. On a single
thread, one slow step, such as waiting on a copy to finish, would freeze all the
others.

If a process is an office building with one shared set of filing cabinets, threads
are the employees: each doing different work, all reading and writing the same
files. That sharing is why they are fast, and why they need coordination.

**REST**: a common style for building APIs over HTTP. Clients use standard HTTP
methods (``GET``, ``POST``) on resources that each have their own URL, so programs
can exchange data with ordinary HTTP tools instead of a custom protocol. Both
sides still have to agree on which URLs exist and what data format each one uses.

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
plain Kubernetes cannot express, such as draining a cache server without losing in-flight
lookups, or coordinated rollouts across a cache fleet.

**Where does the KV cache actually live in a Kubernetes deployment?**

Wherever you configure the tiers to put it: GPU memory first, then L1 (CPU memory
on the same node, sized with ``--l1-size-gb``), then any L2 backends configured
with ``--l2-adapter``, such as a local file system or an object store reachable
across the cluster. See :doc:`cache tiering <cache-tiering>` for the hierarchy and the
tradeoffs at each level.

.. seealso::

   :doc:`cache-aware-routing` | :doc:`cache-hit-rate` | :doc:`cache-tiering` | :doc:`interconnects` | :doc:`lmcache` | :doc:`index`
