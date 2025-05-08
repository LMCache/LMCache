#!/bin/bash

set -xe 

pipx install uv
uv pip install -e . --no-build-isolation
