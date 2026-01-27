from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os

setup(
    name='simple_extension', # Python에서 import 할 이름
    ext_modules=[
        # CUDA 코드가 없더라도 나중에 GPU 텐서를 다룰 것이므로 CUDAExtension 사용 권장
        # 순수 C++만 있을 땐 CppExtension을 써도 무방
        CUDAExtension(
            name='simple_extension',
            sources=['kernel/simple_add.cpp'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)