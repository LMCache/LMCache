#!/bin/bash

set -xe 

VENV_DIR=".venv"

# Try to find python3.10, otherwise fall back to python3
PYTHON_BIN=$(command -v python3.10 || command -v python3)

if [[ -z "$PYTHON_BIN" ]]; then
  echo "❌ No python3 interpreter found on PATH; please install Python 3.10 or 3.x"
  exit 1
fi

if [[ -d "$VENV_DIR" ]]; then
  echo "⟳ Using existing venv: $(pwd)/$VENV_DIR"
else
  echo "⚙️  Creating venv with Python at: $PYTHON_BIN → $(pwd)/$VENV_DIR"
  # use uv for fast venv creation
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

uv pip install -e .

# List installed packages for debugging
echo "📦 Installed packages in venv:"
uv pip freeze