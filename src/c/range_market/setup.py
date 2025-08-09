from setuptools import setup, Extension
import numpy

setup(
    name="range_market",
    version="0.1.0",
    author="ytee",
    packages=["range_market"],
    ext_modules=[
        Extension(
            "range_market._core",  # package.module
            sources=["range_market/main.c"],
            include_dirs=[numpy.get_include()],
        )
    ],
    zip_safe=False,
)