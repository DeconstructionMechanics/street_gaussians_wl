import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name='gaussian_lidar_renderer',
    packages=['gaussian_lidar_renderer'],
    ext_modules=[
        CUDAExtension(
            name='gaussian_lidar_renderer._C',
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'bvh.cu',
                'trace.cu',
                'construct.cu',
                'backward.cu',
                'bindings.cpp',
            ]],
            include_dirs=[
                os.path.join(_src_path, 'include'),
            ],
            extra_compile_args={
                "nvcc": ["-O3", "--expt-extended-lambda"],
                "cxx": ["-O3"]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
