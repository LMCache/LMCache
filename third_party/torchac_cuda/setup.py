from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='torchac_cuda',
    ext_modules=[
        cpp_extension.CUDAExtension('torchac_cuda', 
                                    ['torchac_kernel.cu']),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)

#setup(
#    name='torchac_cuda_layer',
#    ext_modules=[
#        cpp_extension.CUDAExtension('torchac_cuda_layer', 
#                                    ['torchac_kernel_layer.cu']),
#    ],
#    cmdclass={
#        'build_ext': cpp_extension.BuildExtension
#    }
#)
