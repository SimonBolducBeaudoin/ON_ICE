#pragma once

#include "../../includes/header_common.h"
// #include "../../SM-Scoped_timer/includes/scoped_timer.h"
#include "../../SM-Omp_extra/includes/omp_extra.h"
#include "../../SM-Multi_array/Multi_array.h"

// Numpy compatible functions
template < typename DataType , typename ComplexType >
py::array_t<ComplexType,py::array::c_style> bispectre_fft_parallel_py(const py::array_t<DataType,py::array::c_style>& data , uint l_fft = (1<<10), int n_threads = 4 );

void init_Bispectrum(py::module &m);
