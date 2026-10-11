.. _orchestration:

Orchestration
=============

This section groups LMCache deployment substrates. Each subsection documents
its own launch topology, networking, connector, and storage requirements.

- **Kubernetes** -- long-lived pods, managed by the Deployment manifests or the
  LMCache Operator.
- **NVIDIA Dynamo** -- multi-engine coordination.
- **HPC / Slurm** -- batch jobs on a shared-filesystem supercomputing cluster
  with a rootless container runtime and offline compute nodes.

.. toctree::
   :maxdepth: 2

   /production/kubernetes_deployment
   /mp/operator
   /production/dynamo_coordination
   slurm_hpc/index
