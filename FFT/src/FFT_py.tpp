template< class DataType >
DataType FFT_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*n);

	FFT(n, reinterpret_cast<fftw_complex*>(ptr_py_in), reinterpret_cast<fftw_complex*>(result));

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n},
		{sizeof(dbl_complex)},
		result,
		free_when_done	
	);

}

template< class DataType >
DataType iFFT_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*n);

	iFFT(n, reinterpret_cast<fftw_complex*>(ptr_py_in), reinterpret_cast<fftw_complex*>(result));

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n},
		{sizeof(dbl_complex)},
		result,
		free_when_done	
	);

}


template< class DataType >
np_complex rFFT_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	double* ptr_py_in = (double*) buf_in.ptr;
	dbl_complex* result = (dbl_complex*) fftw_malloc(sizeof(dbl_complex)*(n/2+1));

	rFFT(n, ptr_py_in, reinterpret_cast<fftw_complex*>(result));

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<dbl_complex, py::array::c_style> 
	(
		{n/2+1},
		{sizeof(dbl_complex)},
		result,
		free_when_done	
	);

}

template< class DataType >
np_double irFFT_py(DataType py_in)
{
	py::buffer_info buf_in = py_in.request();

	if (buf_in.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}	

	int n = buf_in.size;

	dbl_complex* ptr_py_in = (dbl_complex*) buf_in.ptr;
	double* result = (double*) fftw_malloc(sizeof(double)*2*(n-1));

	irFFT(2*(n-1), reinterpret_cast<fftw_complex*>(ptr_py_in), result);

	py::capsule free_when_done( result, fftw_free );
	return py::array_t<double, py::array::c_style> 
	(
		{2*(n-1)}, //Shape
		{sizeof(double)}, //Stride
		result, //Pointer
		free_when_done //mem clear
	);

}


