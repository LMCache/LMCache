export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
cd ..
python3 setup.py bdist_wheel --dist-dir=dist
uv pip install dist/*.whl