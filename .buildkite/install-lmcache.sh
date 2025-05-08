#!/bin/bash

set -xe 

python3 -m pip install uv
uv pip install -e . --no-build-isolation
