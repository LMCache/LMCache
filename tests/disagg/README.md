# Test disaggregated prefill related components

## NIXL Pipe

```bash
# Terminal 1 (sender)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=0 python3 test_nixl_pipe.py --role sender
 
# Terminal 2 (receiver)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=1 python3 test_nixl_pipe.py --role receiver
```

## NIXL Channel

```bash
# Terminal 1 (Sender)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=0 python3 test_nixl_channel.py --role sender --num-objs 500

# Terminal 2 (Receiver)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=1 python3 test_nixl_channel.py --role receiver --num-objs 500
```

NOTE: why 500 objects? -- Becase we the pipe only has 4GB buffer, but 500 objects are 16GB in total. This is to test when the data to transfer is larger than the buffer size, how the sender/receiver behaves.
