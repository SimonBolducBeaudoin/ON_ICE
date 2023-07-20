#include "../includes/Convolution.h"

// Explicit template instanciation
// See: https://docs.microsoft.com/en-us/cpp/cpp/explicit-instantiation?view=vs-2019

template void Convolution_direct( double* f , double* g , double* h , uint L_data , uint L_kernel);
template void Convolution_direct( double* f , int16_t* g , double* h , uint L_data , uint L_kernel);

// template void Convolution_fft_parallel( double* f , double* g , double* h , uint L_data , uint L_kernel , uint L_FFT , int n_threads);
// template void Convolution_fft_parallel( double* f , int16_t* g , double* h , uint L_data , uint L_kernel , uint L_FFT , int n_threads);