from distutils.core import setup, Extension

cpp_args = ['-std=c++11', '-fopenmp']

ext_modules = [ Extension('phase_ret_algs',
                         sources =['funcs.cpp', 'phase_ret_algs.cpp'],
                         libraries = ['fftw3', 'fftw3_omp'],
                         include_dirs=['pybind11/include'],
                         extra_compile_args = cpp_args)]

setup(name='phase_ret_algs',
      version='0.0.1',
      description='C++ implementation of phase retrieval algorithms',
      ext_modules=ext_modules)
