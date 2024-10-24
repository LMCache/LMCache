from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='lmc_ops',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'lmc_ops',
            [
                'pybind.cpp',
                'mem_kernels.cu',
                'cal_cdf.cu',
                'ac_enc.cu',
                'ac_dec.cu',
            ],
            #extra_compile_args={'cxx': ['-g'],
            #                    'nvcc': ['-G', '-g']},
            #include_dirs=['./']#['./include']
        ),
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
