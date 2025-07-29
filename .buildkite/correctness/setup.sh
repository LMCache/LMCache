#!/bin/bash

unxz data.tar.xz
tar xf data.tar

git config --global --add safe.directory /home/slshen/correctness_repositories/LMCache
git config --global --add safe.directory /home/slshen/correctness_repositories/vllm

# see README.md for pre-configuring your CI runner
cd /home/slshen/correctness_repositories/LMCache
git pull origin dev
cd /home/slshen/correctness_repositories/vllm
git pull origin main
