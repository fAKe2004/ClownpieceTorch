from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'clownpiece',
        sources=[
            "tensor/meta.cc",
            "tensor/tensor.cc",
            "tensor/tensor_pybind.cc"
        ],
        include_dirs=[
            pybind11.get_include(),
            "tensor/",
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-Wall'],
    ),
]

setup(
    name='clownpiece',
    version='0.0.1',
    ext_modules=ext_modules,
)