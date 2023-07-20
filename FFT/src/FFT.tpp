template<class DataType>
void FFT(int n, DataType* in, DataType* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<class DataType>
void iFFT(int n, DataType* in, DataType* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<class DataTypeIn, class DataTypeOut>
void rFFT(int n, DataTypeIn* in, DataTypeOut* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

template<class DataTypeIn, class DataTypeOut>
void irFFT(int n, DataTypeIn* in, DataTypeOut* out)
{
	fftw_plan plan;
	plan = fftw_plan_dft_c2r_1d(n, in, out, FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
}


