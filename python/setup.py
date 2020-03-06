from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


setup(
    ext_modules = cythonize([
    Extension("cvm", ["cvm.pyx"], libraries=["cvm_runtime_cuda"])
    ]) 
)
