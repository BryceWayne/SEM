from distutils.core import setup
from Cython.Build import cythonize

setup(name='fastlepoly', ext_modules = cythonize('fastlepoly.pyx', annotate=True))