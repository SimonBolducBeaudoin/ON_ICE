#pragma once

#include <omp_extra.h>
#include <Multi_array.h>
// #include <complex>
// #include <fftw3.h>

typedef unsigned int uint ;
typedef std::complex<double> complex_d;

// Core functions
template <class KernelType, class DataType, class OutputType>
void Convolution_direct( KernelType* f , DataType* g , OutputType* h , uint L_data , uint L_kernel);

// template < class KernelType, class DataType , class OutputType>
// void Convolution_fft_parallel( KernelType* f , DataType* g , OutputType* h , uint L_data , uint L_kernel , uint L_FFT = (1<<10), int n_threads = 4);

#include "../src/Convolution.tpp"