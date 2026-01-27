from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os

setup(
    name='custom_backend', # Python에서 import 할 이름
    ext_modules=[
        # CUDA 코드가 없더라도 나중에 GPU 텐서를 다룰 것이므로 CUDAExtension 사용
        # 순수 C++만 있을 땐 CppExtension을 써도 무방
        CUDAExtension(
            name='custom_backend',
            sources=[
                'custom_kernel/bindings.cpp',
                'custom_kernel/custom_add.cu',
                'custom_kernel/custom_matmul.cu',
            ],
            libraries=['cublas'],  # cuBLAS 라이브러리 링크
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)