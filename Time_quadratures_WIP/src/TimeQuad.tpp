/////////////////////////////////////////
// FFT convolution constructors

#include <cstdio>

template<class Quads_Index_Type>
TimeQuad<Quads_Index_Type>::TimeQuad( double Z , double dt , uint64_t l_data , uint kernel_conf , np_complex_d betas , np_complex_d g , double alpha , uint l_fft 	, int n_threads )
: 
	Z(Z) , dt(dt) , l_kernel(compute_l_kernel(betas)) , l_hc(compute_l_hc(betas)), l_data(l_data) , kernel_conf(kernel_conf) ,\
	n_quads(compute_n_quads(kernel_conf)) , prefactor(compute_prefactor(Z)),\
	l_valid( compute_l_valid( compute_l_kernel(betas) ,l_data ) ) , l_full( compute_l_full(  compute_l_kernel(betas) ,l_data ) ) , 
	alpha   ( alpha ) , 
	n_kernels(compute_n_kernels(betas)),n_threads(n_threads),\
    l_fft		(l_fft)									, 
    l_chunk		(compute_l_chunk(l_kernel,l_fft)) 		,
    n_chunks    (compute_n_chunks(l_data,l_chunk)) 		,
    l_reste     (compute_l_reste(l_data,l_chunk)) 		,
    betas   ( Multi_array<complex_d,2>::numpy_copy(betas) ) , \
	g       ( Multi_array<complex_d,1>::numpy_copy(g) ) , \
	filters ( Multi_array<complex_d,2>(n_kernels,compute_l_hc(betas)) ),\
	ks_complex	( Multi_array<complex_d,3>				( n_quads , n_kernels , l_hc , fftw_malloc , fftw_free ) ) ,\
	ks			( Multi_array<double,3>					( (double*)ks_complex.get_ptr() , n_quads , n_kernels , l_kernel, n_kernels*l_hc*sizeof(complex_d),l_hc*sizeof(complex_d) , sizeof(double) ) ),\
	half_norms	( Multi_array<double,2>					( n_quads , n_kernels ) ),\
	quads		( Multi_array<double,3,Quads_Index_Type>( n_quads , n_kernels , l_full ) ),
	gs			( Multi_array<double,2>		(n_threads,2*(l_fft/2+1),fftw_malloc,fftw_free) ) 	,
	fs			( Multi_array<complex_d,2>	(n_threads,(l_fft/2+1)	,fftw_malloc,fftw_free) ) 	,
	hs( Multi_array<complex_d,4>			(n_quads,n_kernels,n_threads,(l_fft/2+1),fftw_malloc,fftw_free) )
{ 
	checks();
	// checks_n_threads();
	omp_set_num_threads(n_threads);
	checks_g();
	checks_betas();
		
	prepare_plans();
	make_kernels();
    
    /////////////////////////////////////////
	prepare_kernels();
    /////////////////////////////////////////
};
/////////////////////////////////////////

// DESTRUCTOR
template<class Quads_Index_Type>
TimeQuad<Quads_Index_Type>::~TimeQuad()
{	
	destroy_plans();
}

// INITIALISOR/STATIC METHODS
template<class Quads_Index_Type>
uint TimeQuad<Quads_Index_Type>::compute_n_quads(uint kernel_conf)
{
	if(kernel_conf==0) // q seulement
	{
		return 1 ;
	}
	else if (kernel_conf==1) // p et q 
	{
		return 2 ;
	}
	// else if (kernel_conf==2)  // pi/4 et 3pi/4
	// {
		// return 2 ;
	// }
	else if (kernel_conf==3) // Pas de kernel (i.e. ones )
	{
		return 1 ;
	}
	else
	{
		throw std::runtime_error(" Invalid kernel_conf ");
	}
}

template<class Quads_Index_Type>
np_complex_d TimeQuad<Quads_Index_Type>::compute_flat_band(uint l_hc,double dt,double f_min_analog_start,double f_min_analog_stop,double f_max_analog_start,double f_max_analog_stop)
{
	double f_Nyquist = compute_f_Nyquist( dt );
	double f[l_hc];
	uint l_kernel = compute_l_kernel_from_l_hc 	( l_hc );
	Multi_array<complex_d,1> filter (l_hc);
    
    for ( uint i = 0 ; i<l_hc  ; i++ )
    { 
        f[i] = fft_freq( i , l_kernel , dt );
        filter[i] = 1.0;
    };
    
	Tukey_Window( filter.get_ptr() , f , l_hc , f_min_analog_start, f_min_analog_stop, f_max_analog_start, f_max_analog_stop , f_Nyquist ) ;
	
	return filter.copy_py() ;
}

// CHECKS 
template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::checks()
{
	if (l_kernel %2 != 1)
	{
		throw std::runtime_error(" l_kernel is not odd dont expect this to work... ");
	} 
	if (l_kernel > l_data)
	{
		throw std::runtime_error(" l_kernel > l_data dont expect this to work... ");
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::checks_betas()
{
	if( (betas.get_n_i() != l_hc) )
	{
		throw std::runtime_error(" length of betas not mathching with length of kernels ");
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::checks_g()
{
	if( (betas.get_ptr() != NULL) and (g.get_n_i() != l_hc) )
	{
		throw std::runtime_error(" length of g not mathching with length of kernels ");
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::execution_checks( uint64_t l_data  )
{
	if ( this->l_data != l_data )
	{
		throw std::runtime_error(" data length given during execution dont match data length declared at instentiation ");
	}
}

// PREPARE_PLANS METHOD
template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::prepare_plans()
{   
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	k_foward 	= fftw_plan_dft_r2c_1d( l_kernel 	, ks(0,0) 									, reinterpret_cast<fftw_complex*>(ks(0,0)) 	, FFTW_ESTIMATE); // FFTW_ESTIMATE
	k_backward 	= fftw_plan_dft_c2r_1d( l_kernel	, reinterpret_cast<fftw_complex*>(ks(0,0)) 	, ks(0,0) 									, FFTW_ESTIMATE); 
    //////////////////////////////////////////////////////////////////////////////////
	kernel_plan = fftw_plan_dft_r2c_1d	( l_fft , (double*)ks_complex(0,0) 	, reinterpret_cast<fftw_complex*>(ks_complex(0,0)) 	, FFTW_EXHAUSTIVE);
	g_plan = fftw_plan_dft_r2c_1d		( l_fft , gs[0] 					, reinterpret_cast<fftw_complex*>( fs[0] ) 			, FFTW_EXHAUSTIVE);
	h_plan = fftw_plan_dft_c2r_1d		( l_fft , reinterpret_cast<fftw_complex*>(hs(0,0,0)) , (double*)hs(0,0,0) 			 , FFTW_EXHAUSTIVE); /* The c2r transform destroys its input array */
    //////////////////////////////////////////////////////////////////////////////////
	fftw_export_wisdom_to_filename("FFTW_Wisdom.dat");	
}

// FRREE METHODS
template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::destroy_plans()
{
	fftw_destroy_plan(k_foward); 
    fftw_destroy_plan(k_backward); 
    fftw_destroy_plan(kernel_plan); 
    fftw_destroy_plan(g_plan); 
    fftw_destroy_plan(h_plan); 	
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::make_kernels()
{
	normalize_betas(); /* Can be done first because only depend on betas */
    
	vanilla_kernels();
	normalize_for_dfts(); 
	apply_windows();
	compute_filters();
	apply_filters();
	
    half_normalization();
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::prepare_kernels()
{
	double norm_factor = dt/l_fft; /*MOVED IN PREPARE_KERNELS*/

	for ( uint k = 0 ; k<n_quads ; k++ ) 
	{
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			/* Value assignment and zero padding */
			for( uint i = 0 ; i < l_kernel ; i++)
			{
				( (double*)ks_complex(k,j) )[i] = ks(k,j,i)*norm_factor ; /*Normalisation done here*/
			}
			for(uint i = l_kernel ; i < l_fft ; i++)
			{
				( (double*)ks_complex(k,j) )[i] = 0 ; 
			}
			fftw_execute_dft_r2c(kernel_plan, (double*)ks_complex(k,j) , reinterpret_cast<fftw_complex*>(ks_complex(k,j)) ); 
		}
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::set_g( Multi_array<complex_d,1> g)
{
    if(  g.get_n_i() != this->g.get_n_i() )
	{
		throw std::runtime_error(" length of the new g doensn't match the old one");
	}
    
    for (uint i = 0 ; i < l_hc ; i++ )
	{
        this->g[i] = g[i] ;
    }
    
	vanilla_kernels();
	normalize_for_dfts(); 
	apply_windows();
	compute_filters();
	apply_filters();
	
    half_normalization();
    
    prepare_kernels(); // Update algorithm kernels (not all necessarly shared)
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::set_g_py(np_complex_d g)
{
	set_g( Multi_array<complex_d,1>::numpy_copy(g) ) ;
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::vanilla_kp(uint quadrature_index, uint mode_index)
{
	double t ; 	/* Abscisse positif */
	double prefact ;
	double argument ;
	double tmp1;
	
	uint k = quadrature_index;
	uint j = mode_index;
	
	for (uint i = 0 ; i < l_kernel/2; i++ ) // l_kernel doit être impaire
	{
		t = ( i + 1 )*dt;
		prefact = 2.0/sqrt(t);
		argument = sqrt( 2.0*t/dt );
		/* Right part */
		tmp1 = prefact * Fresnel_Cosine_Integral( argument ) ;
		
		ks( k , j , l_hc + i ) = tmp1;
		/* Left part */
		ks( k , j , l_hc - 2 - i ) = tmp1;
	}
	/* In zero */
	ks( k , j , l_kernel/2 ) = 2.0*sqrt(2.0)/sqrt(dt);
    
    for (uint i = 0 ; i < l_kernel; i++ )
    {
        ks( k , j , i ) *= prefactor ; /* units_correction * sqrt( 2/ Zh ) */
    }
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::delta(uint quadrature_index, uint mode_index)
{	
	uint k = quadrature_index;
	uint j = mode_index;
    
    for (uint i = 0 ; i < l_kernel; i++ )
    {
        ks( k , j , i ) = 0.0  ;
    }
	
	/* In zero */
	ks( k , j , l_kernel/2 ) = units_correction/dt ;
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::vanilla_kq(uint quadrature_index, uint mode_index)
{
	double t ; 	/* Abscisse positif */
	double prefact ;
	double argument ;
	double tmp1;
	
	uint k = quadrature_index;
	uint j = mode_index;
	
	for (uint i = 0 ; i < l_kernel/2; i++ ) // l_kernel doit être impaire
	{
		t = ( i + 1 )*dt;
		prefact = 2.0/sqrt(t);
		argument = sqrt( 2.0*t/dt );
		/* Right part */
		tmp1 = prefact * Fresnel_Sine_Integral( argument ) ;
		
		ks( k , j , l_hc + i ) = tmp1;
		/* Left part */
		ks( k , j , l_hc - 2 - i ) = (-1)*tmp1;
	}
	/* In zero */
	ks( k , j , l_kernel/2 ) = 0 ;
    
    for (uint i = 0 ; i < l_kernel; i++ )
    {
        ks( k , j , i ) *= prefactor ; /* units_correction * sqrt( 2/ Zh ) */
    }
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::vanilla_k_pi_over_4(uint quadrature_index, uint mode_index)
{
	double t ; 	/* Abscisse positif */
	double prefact ;
	double argument ;
	double tmp1;
	
	uint k = quadrature_index;
	uint j = mode_index;
	
	for (uint i = 0 ; i < l_kernel/2; i++ ) // l_kernel doit être impaire
	{
		t = ( i + 1 )*dt;
		prefact = 2.0/sqrt(t);
		argument = sqrt( 2.0*t/dt );
		/* Right part */
		tmp1 = prefact * Fresnel_Cosine_Integral( argument ) ;
		
		ks( k , j , l_hc + i ) = tmp1;
		/* Left part */
		ks( k , j , l_hc - 2 - i ) = 0.0 ; /* LEFT PART IS ZERO */
	}
	/* In zero */
	ks( k , j , l_kernel/2 ) = sqrt(2.0)/sqrt(dt); /* THE VALUES IN 0 IS HALF OF KP  VOIR NOTE SV REGULARISATION */
    
    for (uint i = 0 ; i < l_kernel; i++ )
    {
        ks( k , j , i ) *= prefactor ; /* units_correction * sqrt( 2/ Zh ) */
    }
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::vanilla_k_3_pi_over_4(uint quadrature_index, uint mode_index)
{
	double t ; 	/* Abscisse positif */
	double prefact ;
	double argument ;
	double tmp1;
	
	uint k = quadrature_index;
	uint j = mode_index;
	
	for (uint i = 0 ; i < l_kernel/2; i++ ) // l_kernel doit être impaire
	{
		t = ( i + 1 )*dt;
		prefact = 2.0/sqrt(t);
		argument = sqrt( 2.0*t/dt );
		/* Right part */
		tmp1 = prefact * Fresnel_Cosine_Integral( argument ) ;
		
		ks( k , j , l_hc + i ) = 0.0; /* RIGHT PART IS ZERO */
		/* Left part */
		ks( k , j , l_hc - 2 - i ) = tmp1 ; /* RIGHT PART IS ZERO */
	}
	/* In zero */
	ks( k , j , l_kernel/2 ) = sqrt(2.0)/sqrt(dt); /* THE VALUES IN 0 IS HALF OF KP VOIR NOTE SV REGULARISATION */
    
    for (uint i = 0 ; i < l_kernel; i++ )
    {
        ks( k , j , i ) *= prefactor ; /* units_correction * sqrt( 2/ Zh ) */
    }
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::vanilla_kernels()
{
	if(kernel_conf==0)
	{
		for ( uint i = 0 ; i<n_kernels ; i++ ) 
		{	
			vanilla_kq(0,i); //quadrature_index , mode_index
		} 
	}
	else if (kernel_conf==1)
	{
		for ( uint i = 0 ; i<n_kernels ; i++ ) 
		{	
			vanilla_kp(0,i); //quadrature_index , mode_index
			vanilla_kq(1,i); //quadrature_index , mode_index
		} 
	}
	// else if ((kernel_conf==2)
	// {
		// for ( uint i = 0 ; i<n_kernels ; i++ ) 
		// {	
			// vanilla_k_pi_over_4(0,i); //quadrature_index , mode_index
			// vanilla_k_3_pi_over_4(0,i); //quadrature_index , mode_index
		// }
	// }
	else if(kernel_conf==3)
	{
		for ( uint i = 0 ; i<n_kernels ; i++ ) 
		{	
			delta(0,i); //quadrature_index , mode_index
		} 
	}
	else
	{
		throw std::runtime_error(" Invalid kernel_conf ");
	}
}	

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::normalize_for_dfts()
{	
	/*
	This is the normalisation relative to the filtering occuring in the construction of ther Kernel 
	(as nothing to do with the convolution product happening afterward in TimeQuad_FFT)
	*/
	double fft_norm = 1.0/l_kernel ; // Normalization for FFT's
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ )
		{
			for ( uint i = 0 ; i<l_kernel ; i++ )
			{
				ks(k,j,i) *= fft_norm ;
			}
		}
	}
}	

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::compute_filters()
{
	for ( uint j = 0 ; j<n_kernels ; j++ ) 
	{
		for ( uint i = 0 ; i<l_hc ; i++ )
		{
			filters(j,i) = ( betas(j,i)/g(i) ) ;
		}
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::apply_filters()
{	
	
	/* Foward transforms */
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ )
		{
			fftw_execute_dft_r2c( k_foward, ks(k,j) , reinterpret_cast<fftw_complex*>( ks(k,j) ) ); 
		}
	}
	/* Apply betas */
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			for ( uint i = 0 ; i<l_hc ; i++ )
			{
				ks_complex(k,j,i) *= filters(j,i) ;
			}
		}
	}
	/* Returning to real space */
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			/* c2r destroys its inputs array */
			fftw_execute_dft_c2r(k_backward, reinterpret_cast<fftw_complex*>(ks(k,j)) , ks(k,j) ); 
		}
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::normalize_betas()
{
	for ( uint j = 0 ; j<n_kernels ; j++ )
	{
		normalize_a_beta(j);
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::normalize_a_beta(uint mode_index)
{
	/*	Normalized to 1/2 :
						 beta(f)
	Beta(f)	=	---------------------------
			   ( 2 int |beta(f)|^2 df )^1/2
	*/
	uint j = mode_index;
	double sum = 0 ;
    double df = fft_freq(1,l_kernel,dt);
	for ( uint i = 0 ; i<l_hc ; i++ )
	{
		sum += std::norm(betas(j,i));
	}
    
	sum *= df ; 
	sum = sqrt(2.0)*sqrt(sum);
	for ( uint i = 0 ; i<l_hc ; i++ )
	{
		betas(j,i) /= sum ;
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::apply_windows()
{
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			Tukey_Window( ks(k,j) , alpha , l_kernel ) ;
		}
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::half_normalization()
{	
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			double tmp = 0;
			for ( uint i = 0 ; i<l_kernel ; i++ )
			{
				tmp += ks(k,j,i)*ks(k,j,i);
			}
			
			half_norms(k,j) = sqrt(tmp);
			
			for ( uint i = 0 ; i<l_kernel ; i++ )
			{
				ks(k,j,i) /= half_norms(k,j);
			}
		}
	}
}

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::half_denormalization()
{	
	for ( uint k = 0 ; k<n_quads ; k++ )
	{
		for ( uint j = 0 ; j<n_kernels ; j++ ) 
		{
			for ( uint i = 0 ; i<l_kernel ; i++ )
			{
				ks(k,j,i) *= half_norms(k,j);
			}
		}
	}	
}

template<class Quads_Index_Type>
template<class DataType>
void TimeQuad<Quads_Index_Type>::execute( Multi_array<DataType,1,uint64_t>& data )
{
	execution_checks( data.get_n_i() );
	convolution( data );	
}

/*
template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::reset_quads()
{
	#pragma omp parallel
    {
        manage_thread_affinity();		
        #pragma omp for simd collapse(4)
		for ( uint l = 0 ; l<n_quads ; l++ ) 
		{
			for ( uint j = 0 ; j<n_kernels ; j++ ) 
			{  
				for( uint i=0; i < n_chunks-1 ; i++ )
				{	
					// Last l_kernel-1.0 points
					// Subject to race conditions
					for( uint k=l_chunk ; k < l_fft; k++ )
					{	
						quads(l,j,i*l_chunk+k) = 0.0 ;
					}
				}	
			}
		}
    }
	if (l_reste != 0)
	{			
		// Product 
		for ( uint l = 0 ; l<n_quads ; l++ ) 
		{
			for ( uint j = 0 ; j<n_kernels ; j++ ) 
			{
				// Select only the part of the ifft that contributes to the full output length
				for( uint k = 0 ; k < l_reste + l_kernel - 1 ; k++ )
				{
					quads(l,j,n_chunks*l_chunk+k) = 0.0 ;
				}
			}
		}
	}
}
*/

template<class Quads_Index_Type>
void TimeQuad<Quads_Index_Type>::convolution( Multi_array<int16_t,1,uint64_t>& data )
{
    //omp_set_num_threads(n_threads); // This works
	#pragma omp parallel
	{	
		manage_thread_affinity();
        int this_thread = omp_get_thread_num();
        printf("Hi : %d ! \n", this_thread);
        
		#pragma omp for simd collapse(2)
        for( uint i=0; i < (uint) n_threads ; i++  )
        {
            for( uint k=0; k < l_fft ; k++ )
            {
                gs(i,k) = 0;
            }
        }
	//// Loop on chunks ---->
		
        
		#pragma omp for
		for( uint i=0; i < n_chunks ; i++ )
		{
            
			///// THIS ONLY ONCE
			// fft_data ///
			
			for( uint j=0 ; j < l_chunk ; j++ )
			{
				gs(this_thread,j) = (double)data[i*l_chunk + j] ; // Cast data to double
			}
			
			fftw_execute_dft_r2c( g_plan, gs[this_thread] , reinterpret_cast<fftw_complex*>( fs[this_thread] ) );
			
			/////
			
			///// THIS FOR EACH KERNELS PAIRS
			for ( uint l = 0 ; l<n_quads ; l++ ) 
			{
				for ( uint j = 0 ; j<n_kernels ; j++ ) 
				{
					// Product	
					for( uint k=0 ; k < (l_fft/2+1) ; k++ )
					{	
						hs(l,j,this_thread,k) = ks_complex(l,j,k) * fs(this_thread,k);
					}
					
					
					// ifft
					fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(hs(l,j,this_thread)) , (double*)hs(l,j,this_thread) );   
					
					// First l_kernel-1.0 points
					// Subject to race conditions
					for( uint k=0; k < l_kernel-1 ; k++ )
					{
						#pragma omp atomic update
						quads(l,j,i*l_chunk+k) += ( (double*)hs(l,j,this_thread))[k] ;
					}
					// Copy result to p and q 
					// Not subject to race conditions
					for( uint k=l_kernel-1; k < l_chunk ; k++ )
					{	
						quads(l,j,i*l_chunk+k) = ( (double*)hs(l,j,this_thread))[k] ;
					}
					// Last l_kernel-1.0 points
					// Subject to race conditions
					for( uint k=l_chunk ; k < l_fft ; k++ )
					{
						#pragma omp atomic update
						quads(l,j,i*l_chunk+k) += ( (double*)hs(l,j,this_thread))[k] ;
					}
				}
			}
		}
	}
	///// The rest ---->
	/*
    if (l_reste != 0)
	{	
        // add the rest
        uint k=0 ;
		for(; k < l_reste ; k++ )
		{
			gs(0,k) = (double)data[n_chunks*l_chunk + k] ;
		}
		// make sure g only contains zeros
		for(; k < l_fft ; k++ )
		{
			gs(0,k) = 0 ;
		}
		
		fftw_execute_dft_r2c(g_plan, gs[0] , reinterpret_cast<fftw_complex*>( fs[0]) );
		
		// Product 
		for ( uint l = 0 ; l<n_quads ; l++ ) 
		{
			for ( uint j = 0 ; j<n_kernels ; j++ ) 
			{
				complex_d tmp;
				for( uint k=0; k < (l_fft/2+1) ; k++)
				{
					tmp = fs(0,k) ;
					hs(l,j,0,k) = ks_complex(l,j,k) * tmp;
				}
				
				fftw_execute_dft_c2r(h_plan , reinterpret_cast<fftw_complex*>(hs(l,j,0)) , (double*)hs(l,j,0) );  
			
				// Select only the part of the ifft that contributes to the full output length
				for( uint k = 0 ; k < l_reste + l_kernel - 1 ; k++ )
				{
					quads(l,j,n_chunks*l_chunk+k) += ( (double*)hs(l,j,0))[k] ;
				}
			}
		}
	}
    */
}

template<class Quads_Index_Type>
template<class DataType>
void TimeQuad<Quads_Index_Type>::execute_py( py::array_t<DataType, py::array::c_style> np_data )
{ 
   // Multi_array<DataType,1,uint64_t> data = Multi_array<DataType,1,uint64_t>::numpy_share(np_data) ;
    py::buffer_info buffer = np_data.request() ;    
    Multi_array<DataType,1,uint64_t> data =  Multi_array<DataType,1,uint64_t>( (DataType*)buffer.ptr , buffer.shape[0] , buffer.strides[0] ) ;
    py::gil_scoped_release release; 
	convolution( data );
}

template<class Quads_Index_Type>
np_double TimeQuad<Quads_Index_Type>::get_quads()
{
	/*
	Pybind11 doesn't work with uint64_t 
		This coul potentially cause problems with l_data, l_valid and l_full
	*/
	// Numpy will not copy the array when using the assignement operator=
	double* ptr = quads[0] + l_kernel -1 ;
	py::capsule free_dummy(	ptr, [](void *f){;} );
	
	return py::array_t<double, py::array::c_style>
	(
		{uint(n_quads) ,uint(n_kernels), uint(l_valid) },      // shape
		{quads.get_stride_k(),quads.get_stride_j(), quads.get_stride_i() },   // C-style contiguous strides for double
		ptr  ,       // the data pointer
		free_dummy // numpy array references this parent
	);
}

// MACROS UNDEF
#undef INPUTS_0 
#undef INPUTS_1_cpp 
#undef INPUTS_1_python 
#undef INPUTS_2_fft
#undef BETAS_G_INITS
#undef BETAS_G_NUMPY_INITS
#undef INIT_LIST_PART1
#undef INIT_LIST_PART2
