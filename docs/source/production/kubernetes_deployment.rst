Kubernetes Deployment
=====================

For multiprocess deployments, use the :doc:`LMCache operator </mp/operator>`
or the :doc:`manual DaemonSet deployment </mp/deployment>`. Both run a
standalone LMCache server and connect inference workers through MP connectors.
For Dynamo deployments, see :doc:`dynamo_coordination`.

The Production Stack example that enables an embedded CPU offload buffer
is kept in :doc:`/legacy/production/kubernetes_deployment`; its YAML does not
create an MP server.
