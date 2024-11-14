from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    name='gaussian_lidar_renderer',
    packages=['gaussian_lidar_renderer'],
    ext_modules=[
        CUDAExtension(
            name='gaussian_lidar_renderer._C',
            sources=[
                'lidar_render.cu',
                'ext.cpp'
            ],
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique", "-I" + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diff-gaussian-rasterization/third_party/glm/")]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
