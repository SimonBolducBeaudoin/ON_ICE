// Numpy compatible functions
template < typename KernelType, typename DataType , typename OutputType >
py::array_t<OutputType> Convolution_direct_py(py::array_t< KernelType> f, py::array_t<DataType> g )
{
    py::buffer_info buf_f = f.request(),
					buf_g = g.request();

    if (buf_f.ndim != 1 || buf_g.ndim != 1 )
    {
		throw std::runtime_error("Number of dimensions must be one");
	}
	
	uint L_data = buf_g.size,
		 L_kernel = buf_f.size,
		 L_output = L_data + L_kernel - 1 ;
		 
	py::array_t<OutputType> h = py::array_t<OutputType>(L_output);
	py::buffer_info buf_h = h.request();
	
    KernelType* ptr_f = (KernelType *) buf_f.ptr;
    DataType* ptr_g = (DataType *) buf_g.ptr;
	OutputType* ptr_h = (OutputType*) buf_h.ptr; 

    Convolution_direct( ptr_f , ptr_g , ptr_h , L_data, L_kernel);
	
	return h ; 
}

// template < class KernelType, class DataType , class OutputType >
// py::array_t<OutputType> Convolution_fft_parallel_py( py::array_t<OutputType> f, py::array_t<DataType> g , uint L_FFT , int n_threads )
// {
    // py::buffer_info buf_f = f.request(),
					// buf_g = g.request();

    // if (buf_f.ndim != 1 || buf_g.ndim != 1 )
    // {
		// throw std::runtime_error("Number of dimensions must be one");
	// }
	
	// uint L_data = buf_g.size,
		 // L_kernel = buf_f.size,
		 // L_output = L_data + L_kernel - 1 ;
		 
	// py::array_t<OutputType> h = py::array_t<OutputType>(L_output);
	// py::buffer_info buf_h = h.request();
	
    // KernelType* ptr_f = (KernelType *) buf_f.ptr;
    // DataType* ptr_g = (DataType *) buf_g.ptr;
	// OutputType* ptr_h = (OutputType*) buf_h.ptr; 

    // Convolution_fft_parallel( ptr_f , ptr_g , ptr_h , L_data , L_kernel , L_FFT , n_threads);
	
	// return h ;
// }

#define CONVOLUTION_DIRECTE(KernelType, DataType, OutputType)\
	m.def("Convolution_direct", &Convolution_direct_py<KernelType , DataType, OutputType> );
	
#define CONVOLUTION_RAPIDE(KernelType, DataType, OutputType)\
	m.def("Convolution_rapide", &Convolution_fft_parallel_py<KernelType , DataType, OutputType> , py::arg("kernel") = py::array_t<KernelType>(1) , py::arg("data")= py::array_t<DataType>(1) , py::arg("l_fft") = 1024 , py::arg("n_threads") = 2 );
    
void init_Convolution(py::module &m)
{
	CONVOLUTION_DIRECTE(double,double,double);
	CONVOLUTION_DIRECTE(double,int16_t,double);
	// CONVOLUTION_RAPIDE(double,double,double);
	// CONVOLUTION_RAPIDE(double,int16_t,double);
}

#undef CONVOLUTION_DIRECTE
#undef CONVOLUTION_RAPPIDE

