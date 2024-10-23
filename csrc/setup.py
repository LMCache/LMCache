from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='lmc_ops',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'lmc_ops', 
            [
                'pybind.cpp',
                'lmc_kernels.cu',
            ],
            #extra_compile_args={'cxx': ['-g'],
            #                    'nvcc': ['-G', '-g']},
            #include_dirs=['./']#['./include']
            ),
        
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)