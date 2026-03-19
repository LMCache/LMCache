CUDA_VISIBLE_DEVICES=0 python -m lmcache.tools.kernel_harness \
    --mode benchmark \
    --use-reference \
    --format normal \
    --dtype bf16 \
    --num-bench-iters 5 \
    --num-warmup-iters 2 
