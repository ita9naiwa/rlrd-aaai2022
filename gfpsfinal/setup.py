from setuptools import setup
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
import numpy

#python setup.py build_ext --inplace
# 하면 이 디렉토리에 설치가 됨;
extensions = [
    Extension("cy_heuristics",
              sources=["cy_heuristics.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-O3", "-std=c++14"],
              extra_link_args=["-std=c++14"],
              language="c++")

]

setup(ext_modules=cythonize(extensions))
