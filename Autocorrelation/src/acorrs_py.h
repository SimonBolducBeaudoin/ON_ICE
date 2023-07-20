#pragma once

#include <Multi_array.h>
#include <fftw_extra.h>

typedef py::array_t<double,py::array::c_style> np_double ; 
typedef py::array_t<complex_d,py::array::c_style> np_complex_d ; 

// np_complex_d r2c_basic_py( np_double& in );
// np_complex_d r2c_advanced_py( np_double& in , int l_fft , int howmany );
// np_double hilbert_py( np_double& in );
// np_complex_d analytic_py( np_double& in );

void init_autocorrelation(py::module &m);
