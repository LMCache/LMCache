# Configure storage backend
This folder is about how to write configuration yamls(for storage backends).  
Just replace example.yaml in other folders with yamls you want in this folder, and run with steps mentioned in those folders.  

Local: CUDA, CPU, Disk.  
Remote: lmcache, redis, redis-sentinel.  
```
chunk_size: an integer (e.g., 256)
local_device: "cuda", "cpu" or "an arbitrary path (e.g., file://local_disk/)"
remote_url: "remote shared cache server url" (e.g., "lm://localhost:65432")
remote_serde: "torch", "safetensor", "cachegen" or "fast"
piplined_backend: True/False
```
## storage backend types
Configuration yaml decides storage backend.  
The table shows which storage backend is used when some fields are present or not present in the yaml.  
```
---------------------------------------------------
| storage backend |  local_device  |  remote_url  |
---------------------------------------------------
|  local backend  |    present     |  not present |
---------------------------------------------------
|  remote backend |   not present  |    present   |
---------------------------------------------------
|  hybrid backend |    present     |  not present |
---------------------------------------------------
```
### local backend
In configuration yaml, local_device tells local backend.   
```
-----------------------------------------------
| local backend |     local_device value      |
-----------------------------------------------
|     cuda      |           "cuda"            |
-----------------------------------------------
|     cpu       |           "cpu"             |
-----------------------------------------------
|     disk      | "file://an_arbitrary_path/" |
-----------------------------------------------
```
### remote backend
In configuration yaml, remote_url tells remote backend.   
Remember to start remote server before starting vllm.  
```
lmcache_server host port
redis-server --bind host --port port
```
```
-------------------------------------------------------------------------
| remote backend |                 remote_url value                     |
-------------------------------------------------------------------------
|     lmcache    | "lm://host:port"                                     |
-------------------------------------------------------------------------
|     redis      | "redis://host:port"                                  |
-------------------------------------------------------------------------
| redis-sentinel | "redis-sentinel://<host>:<port>,<host2>:<port2>,..." |
-------------------------------------------------------------------------  
```
And decide which serde(serializer and deserializer) to use.  
```
remote serde name x, then write.  
remote_serde: "x"  
x can be torch, safetensor, cachegen, fast  
```
## pipelined_backend
Use pipelined remote backend or not.  
```
pipelined_backend: True/False
```
