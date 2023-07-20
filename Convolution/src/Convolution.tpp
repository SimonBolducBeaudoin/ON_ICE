// Cores functions
template < class KernelType, class DataType, class OutputType>
inline void Convolution_direct( KernelType* f , DataType* g , OutputType* h, uint L_data, uint L_kernel)
{
	uint L_full = L_kernel + L_data - 1 ;
	// Begenning
	
    uint64_t i = 0 ;
    uint64_t j = 0 ;
    
    for(; i < L_kernel - 1 ; i++)
	{
		for(j = 0; j <= i  ; j++)
		{	
			h[i] += (OutputType) ( f[j] * g[(i - j)] ) ;
		}
	}
	// Middle (This is also the valid part)
	uint64_t translation =  L_kernel - 1;

    for(j=0; j < L_kernel ; j++)
    {
        for(i = 0 ; i < L_data - L_kernel + 1 ; i++)
        {
			h[i+translation] += (OutputType)( f[j] * g[ (L_kernel - 1) - j + i]  ) ;
		}
	}
	// end
	uint64_t end_full = L_full -1;
	
	for( i=0; i < L_kernel - 1; i++)
	{
		for( j=0; j <= i ; j++)
		{
			h[end_full-i] += (OutputType)(f[(L_kernel-1)-j]  * g[(L_data - 1)-i+j] ) ;
		}
	}
}

/*
template < class KernelType, class DataType, class OutputType >
inline void Convolution_fft_parallel( KernelType* kernel , DataType* data , OutputType* output , uint L_data, uint L_kernel , uint L_FFT, int n_threads)
{
	// uint L_full = L_kernel + L_data - 1 ,
	// L_valid = L_data - L_kernel + 1 ,
	uint L_chunk = L_FFT - L_kernel + 1 , 
		 N_chunks = L_data/L_chunk ,
		 L_reste = L_data%L_chunk ;
	
	// if (L_kernel %2 != 0)
	// {
		// throw std::runtime_error(" L_kernel is not even dont expect this to work... ");
	// }
	if (L_kernel > L_FFT)
	{
		throw std::runtime_error(" L_kernel > L_FFT dont expect this to work... ");
	}
	if (L_kernel > L_data)
	{
		throw std::runtime_error(" L_kernel > L_data dont expect this to work... ");
	}
	if (L_FFT > L_data)
	{
		throw std::runtime_error(" L_FFT > L_data dont expect this to work... ");
	}
	if (n_threads > omp_get_thread_limit())
	{
		throw std::runtime_error(" n_threads is bigger than omp_get_thread_limit() ");
	}
	
	omp_set_num_threads(n_threads);
	
	
	// PREPARE_HEAP
	complex_d* kernel_complex = (complex_d*)fftw_malloc( 2*(L_FFT/2+1) * sizeof(double) ); // For memory alignment
	// For in-place transforms g as 2( n/2 + 1 )  elements. The elements beyond the first n are unused padding.
	complex_d** gs = (complex_d **) malloc(n_threads* sizeof(double*));
	complex_d** hs = (complex_d **) malloc(n_threads* sizeof(double*));  
	#pragma omp parallel
	{
		gs[omp_get_thread_num()] = (complex_d*)fftw_malloc( 2 * (L_FFT/2+1) * sizeof(double) ); // fftw_malloc for memory alignment;
		hs[omp_get_thread_num()] = (complex_d*)fftw_malloc( 2 * (L_FFT/2+1) * sizeof(double) ); // fftw_malloc for memory alignment;
	}
	
	
	// PREPARE_PLANS
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	
	// Prepare plan for real to half complex transform
	fftw_plan kernel_plan = fftw_plan_dft_r2c_1d( L_FFT , (double*)kernel_complex , reinterpret_cast<fftw_complex*>(kernel_complex) , FFTW_EXHAUSTIVE); // in-place fft
	// Prepare plans : r2c and c2r
	fftw_plan g_plan = fftw_plan_dft_r2c_1d( L_FFT , (double*)gs[0] , reinterpret_cast<fftw_complex*>(hs[0]) , FFTW_EXHAUSTIVE);
	//  in-place transform of h.
	//The c2r transform destroys its input array even for out-of-place transforms
	fftw_plan h_plan = fftw_plan_dft_c2r_1d( L_FFT, reinterpret_cast<fftw_complex*>(hs[0]) , (double*)hs[0] , FFTW_EXHAUSTIVE); 
	
	fftw_export_wisdom_to_filename("FFTW_Wisdom.dat");
	
	// OPTIONNALS TIMER
	{
	// Timer Timer(& this->Time_execution); 
	
		// THE CORE OF THE ALGORITHM 
		
		///////////////////////////////////////////////
		//// KERNEL FFT
		// Value assignment and zero padding
		for(uint64_t i = 0 ; i < L_kernel ; i++)
		{
			( (double*)kernel_complex )[i] = (double)kernel[i] ; // Convert IntType to double
		}
		for(uint64_t i = L_kernel ; i < L_FFT ; i++)
		{
			( (double*)kernel_complex )[i] = 0 ; //zero padding
		}

		fftw_execute(kernel_plan); // f now contains the results of the fft
		///////////////////////////////////////////////

		#pragma omp parallel
		{
			manage_thread_affinity();
			
			int this_thread = omp_get_thread_num();
			
			// For zero padding
			for(uint64_t j=0; j < (L_FFT/2+1) ; j++)
			{
				gs[this_thread][j] = 0;
			}


		// loop on chunks and add the fft result to Out
		///// The chunks ---->
			#pragma omp for
			for(uint64_t i=0; i < N_chunks ; i++)
			{
				uint64_t j;
				// Copie-paste the right part of Data into g
				for(j=0; j < L_chunk ; j++)
				{
					( (double*)gs[this_thread])[j] = ( (double)data[i*L_chunk + j] );
				}
				// execute the transform of g 
				fftw_execute_dft_r2c(g_plan, (double*)gs[this_thread] , reinterpret_cast<fftw_complex*>(hs[this_thread]) ); // h now contains the results of the DFT of g
				
				//multiplication de kernel_complex et h
				for(j=0; j < (L_FFT/2+1) ; j++)
				{
					hs[this_thread][j] = kernel_complex[j] * hs[this_thread][j];
				}
				
				fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(hs[this_thread]) , (double*)hs[this_thread] );  // h now contains the results of the chunk's contribution to the convolution product
				
				for(j=0; j < L_chunk ; j++)
				{
					output[i*L_chunk+j] += (OutputType)( ( ( (double*)hs[this_thread])[j] )/L_FFT );
				}
				for(j=L_chunk; j < L_FFT ; j++)
				{
					#pragma omp atomic update
					output[i*L_chunk+j] += (OutputType)( ( ( (double*)hs[this_thread])[j] )/L_FFT );
				}
				//////////////////
			}
		}
		/////
		///// The rest ---->
			if (L_reste != 0)
			{	
				uint64_t j;
				// make sure g only contains zeros
				for(j=0; j < L_FFT ; j++)
				{
					( (double*)gs[0] )[j] = 0 ;
				}
				// add the rest
				for(j=0; j < L_reste ; j++)
				{
					( (double*)gs[0] )[j] = ( (double) data[N_chunks*L_chunk + j] );
				}
				// execute the transform of g 
				fftw_execute(g_plan); // h now contains the results of the DFT of g
				// in-place multiplication de kernel_complex et h (aussi half complexe)
				for(j=0; j < (L_FFT/2+1) ; j++)
				{
					hs[0][j] = kernel_complex[j] * hs[0][j];
				}
				fftw_execute(h_plan);  // g now contains the results of the chunk's contribution to the convolution product
			
				// Select only the part of the ifft that contributes to the full output length
				for(j = 0 ; j < L_reste + L_kernel - 1 ; j++ ){
					output[N_chunks*L_chunk+j] += (OutputType)( ( ( (double*)hs[0] )[j] )/L_FFT );
				}
				// Out now contains the full length convolution
			}
		/////
	}
	
	// FREE MEMORY
	fftw_free(kernel_complex); kernel_complex = NULL ;
    for(int i = 0 ; i < n_threads ; i++)
    {
        fftw_free( gs[i] ) ; gs[i] = NULL ;
        fftw_free( hs[i] ) ; hs[i] = NULL ;
    }
    free(gs) ; gs = NULL ;
    free(hs) ; hs = NULL ;
	
	fftw_cleanup(); 
}
*/