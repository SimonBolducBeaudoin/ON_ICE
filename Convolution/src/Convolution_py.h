#pragma once

#include <Convolution.h>

#include <pybind11/pybind11.h>
// #include <pybind11/complex.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace pybind11::literals;

// Numpy compatible functions
template < typename KernelType, typename DataType , typename OutputType >
py::array_t<OutputType> Convolution_direct_py(py::array_t< KernelType> f, py::array_t<DataType> g );

// template < class KernelType, class DataType , class OutputType >
// py::array_t<OutputType> Convolution_fft_parallel_py( py::array_t<OutputType> f, py::array_t<DataType> g , uint L_FFT = 1024 , int n_threads = 1 );

void init_Convolution(py::module &m);

#include "Convolution_py.tpp"

PYBIND11_MODULE(convolution, m)
{
    m.doc() = " C++ implemented direct convolution and fft convolution";
	init_Convolution(m);
}