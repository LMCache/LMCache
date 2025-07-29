#!/bin/bash

unxz data.tar.xz
tar xf data.tar

# see README.md for pre-configuring your CI runner
source /home/slshen/correctness_venv/bin/activate
cd /home/slshen/correctness_repositories/LMCache
git pull origin dev
cd /home/slshen/correctness_repositories/vllm
git pull origin main
