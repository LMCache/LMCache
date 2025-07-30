#!/bin/bash

set -xe 

VENV_DIR=".venv"
PYTHON_BIN="/usr/bin/python3.10"
if [[ -d "$VENV_DIR" ]]; then
  echo "⟳ Using existing venv: $(pwd)/$VENV_DIR"
else
  echo "⚙️  Creating venv with Python 3.10 at: $(pwd)/$VENV_DIR"
  # use uv for fast venv creation
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
uv pip install -e .

# List installed packages for debugging
echo "📦 Installed packages in venv:"
uv pip freeze