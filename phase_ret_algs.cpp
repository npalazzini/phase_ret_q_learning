#include <pybind11/pybind11.h>
#include <omp.h>

#include "funcs.hpp"

namespace py = pybind11;

PYBIND11_MODULE(phase_ret_algs,m) {
    m.def("init", [](int n_threads){ if(n_threads<=0) n_threads=omp_get_max_threads();
                                     fftw_init_threads();
                                     fftw_plan_with_nthreads(n_threads);
                                     omp_set_num_threads(n_threads);
                                   }, py::arg("n_threads")=-1);
    m.def("ER", &ER, "ER algorithm");
    m.def("HIO", &HIO, "HIO algorithm");
    m.def("get_error", &get_error, "Get error");
    m.def("ShrinkWrap", &ShrinkWrap, "Shrink-Wrap algorithm");
};
