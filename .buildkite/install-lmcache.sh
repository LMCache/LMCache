#!/bin/bash

set -xe 

VENV_DIR=".venv"
PYTHON_BIN="/usr/bin/python3.11"
if [[ -d "$VENV_DIR" ]]; then
  echo "⟳ Using existing venv: $(pwd)/$VENV_DIR"
else
  echo "⚙️  Creating venv with Python 3.11 at: $(pwd)/$VENV_DIR"
  # use uv for fast venv creation
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

uv pip install -e .

# List installed packages for debugging
echo "📦 Installed packages in venv:"
uv pip freeze