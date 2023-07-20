// Numpy compatible functions
template < typename DataType , typename ComplexType >
py::array_t<ComplexType,py::array::c_style> bispectre_fft_parallel_py(const py::array_t<DataType,py::array::c_style>& data_input , uint l_fft , int n_threads )
{
	omp_set_num_threads(n_threads);
	
    Multi_array<DataType,1> data = Multi_array<DataType,1>::numpy(data_input); // This manages the numpy array 
	DataType* data_ptr = data.get() ; 
	
	uint n_j = l_fft/2+1 ; // Less local 2nd fast iterating index (f_2)
	uint n_i = l_fft/4+1 ; // Local fast iterating index (f_1)
	// This manages memeory for the 2D outputs array
		// It makes the array local in memory (faster)
		// It deduces strides from type (here ComplexType)
		// Two last argument are memory allocation function and dellocation function
			// fftw_malloc is used for compatibility with fftw_malloc
			// NULL mean that it wont be able to clean the memory by itself (we will manage memory through the handshake with python)
	Multi_array<ComplexType,2> out(n_j,n_i, fftw_malloc , NULL );
	// Initialize to 0
	#pragma omp parallel
	{	
		manage_thread_affinity(); // Fonction écrite par J.O. qui fait que tout fonction même sur des PC avec n_threads > 64	
		#pragma omp for
		for( uint j=0; j < n_j ; j++ )
		{
			for( uint i=0; i < n_i ; i++ )
			{
				out(j,i) = 0.0 ;
			}
		}
	}
	
	// Allocation de mémoire pour les variables intermédiares
		// Variables recevant la FFT (avec de l'espace d'extra pour les zeros sur les bords)
			// Une pour chaque threads
	Multi_array<ComplexType,2> fs( n_threads , l_fft/2+1, fftw_malloc , fftw_free ); // (l_fft/2+1) est le nombre de frequence représenté par r2c et on le double pour avoir assez de place pour ajouter des zeros
		// Variables intermédiaires pour éviter les atomic update
	Multi_array<ComplexType,3> gs( n_threads , n_j , n_i , fftw_malloc , fftw_free );
	
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	
	fftw_plan plan = 
		fftw_plan_dft_r2c_1d
		( 
			l_fft , 
			data_ptr , // Pointer to the data
			reinterpret_cast<fftw_complex*>(fs[0]) , // reinterpret is for compatibility with FFTW and fs[0] returns a pointer to the first row of fs
			FFTW_EXHAUSTIVE
		);
	
	fftw_export_wisdom_to_filename("FFTW_Wisdom.dat");
	
	// START CALCULATIONS
	uint n_chunks = data.get_n_i()/l_fft ;
	
	#pragma omp parallel
	{	
		manage_thread_affinity(); // Fonction écrite par J.O. qui fait que tout fonction même sur des PC avec n_threads > 64
		
		int this_thread = omp_get_thread_num();
		
	//// Loop on chunks ---->
		#pragma omp for
		for( uint i=0; i < n_chunks ; i++ )
		{ 

			fftw_execute_dft_r2c
			( 
				plan , // Le plan préparé ci dessus
				data_ptr + (i*l_fft) , // Pointer au chunk voulue
				reinterpret_cast<fftw_complex*>( fs[this_thread] ) // le output pour la présente thread
			);
			
			///// Faire le produit et copier dans le output
			
			uint index_fmax = (l_fft/2+1) ;
			// On separe en une somme sur f_1 <= f_2 où f_2 < f_max/2
			// puis une somme sur f_1 <= f_max - f_2 où f_2 >= f_max/2
			for ( uint j = 0 ; j< index_fmax/2 ; j++ ) 
			{
				for( uint k = 0 ; k < j ; k++)
				{
					// COULD BE FASTER WITH SIMD
					gs(this_thread,j,k) = fs(this_thread,k)*fs(this_thread,j)*std::conj(fs(this_thread,j+k)) ;
				}
			}
			for ( uint j = index_fmax/2 ; j< index_fmax ; j++ ) 
			{
				for( uint k = 0 ; k< index_fmax-j ; k++)
				{
					// COULD BE FASTER WITH SIMD
					gs(this_thread,j,k) = fs(this_thread,k)*fs(this_thread,j)*std::conj(fs(this_thread,j+k)) ;
				}
			}
		}
		
	}
	
	// Reduce all gs into out
	for( uint i=0; i < uint(n_threads) ; i++ )
	{
		uint index_fmax = (l_fft/2+1) ;
		// On separe en une somme sur f_1 <= f_2 où f_2 < f_max/2
		// puis une somme sur f_1 <= f_max - f_2 où f_2 >= f_max/2
		for ( uint j = 0 ; j< index_fmax/2 ; j++ ) 
		{
			for( uint k = 0 ; k< j ; k++)
			{
				out(j,k) += gs(i,j,k) ;
			}
		}
		for ( uint j = index_fmax/2 ; j< index_fmax ; j++ ) 
		{
			for( uint k = 0 ; k< index_fmax-j ; k++)
			{
				out(j,k) += gs(i,j,k) ; ;
			}
		}
	}
	
	py::capsule free_when_done( out.get(),fftw_free );
	
	return  py::array_t<ComplexType,py::array::c_style>
			(
				{out.get_n_j(), out.get_n_i()}, // shape
				{out.get_stride_j() , out.get_stride_i()}, // Strides
				out.get(), // Pointer
				free_when_done
			);
}

#define BISPECTRE_FFT_PARALLEL(DataType,ComplexType)\
m.def("bispectrum", &bispectre_fft_parallel_py<DataType,ComplexType> , "data"_a.noconvert() , "l_fft"_a.noconvert() = 1024 , "n_threads"_a.noconvert() = 8 );
void init_Bispectrum(py::module &m)
{
	BISPECTRE_FFT_PARALLEL(double,complex_d);
}

#undef BISPECTRE_FFT_PARALLEL
