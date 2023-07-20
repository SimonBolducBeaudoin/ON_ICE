#pragma once
#include <omp_extra.h>
#include <special_functions.h>
#include <Windowing.h>
#include <fftw3.h>

#include <Multi_array.h>
#include <cmath>

typedef unsigned int uint ;
typedef std::complex<double> complex_d;
typedef py::array_t<double,py::array::c_style> np_double;
typedef py::array_t<complex_d,py::array::c_style> np_complex_d;
/*
	TODOS
    - Add an options to toggle on or off the half_normalization
        Since the betas are now normalized Idk if half_normalization is necessary any more
    - The logic to build the betas,filter,g and kernels should be in a class of it own
        This class should manage the logic for the virtual clases and
        the construction ect..?
*/


/*
	The member TimeQuad_algorithm is implementing virtual functions to acheive run time polymorphism.
	See :
		- https://www.geeksforgeeks.org/polymorphism-in-c/
		- https://www.geeksforgeeks.org/virtual-function-cpp/
*/

template<class Quads_Index_Type>
class TimeQuad
{
	public :
		// Contructors
		/* FFT convolution */
		TimeQuad ( double Z , double dt , uint64_t l_data , uint kernel_conf , np_complex_d betas , np_complex_d g ,double alpha , uint l_fft , int n_threads ); /* numpy array*/
		// Destructor
		~TimeQuad();
		
		// Python getters
		np_double 		get_ks()			{ return ks.copy_py() ;};
		np_complex_d 	get_betas()			{ return betas.copy_py() 		;};
		np_complex_d 	get_g()				{ return g.copy_py() 			;};
		np_complex_d 	get_filters()		{ return filters.copy_py() 		;};
		
		np_double 		get_half_norms()	{ return half_norms.copy_py() 	;};
		
			// Returns only the valid part of the convolution
		np_double 		get_quads(); // Data are shared
		
		// Utilities
		static uint 	compute_l_hc_from_l_kernel	( uint l_kernel )					{ return l_kernel/2+1 			;};
		static uint 	compute_l_kernel_from_l_hc 	( uint l_hc )						{ return l_hc*2-1 				;};
		static double 	fft_freq					( uint i , uint l_fft , double dt )	{ return ((double)i)/(dt*l_fft) ;};
		static double 	compute_f_Nyquist			( double dt )						{ return 1.0/(2.0*dt) 			;};
		static uint64_t compute_l_valid				( uint l_kernel, uint64_t l_data )	{ return l_data - l_kernel + 1 	;};
		static uint64_t compute_l_full				( uint l_kernel, uint64_t l_data )	{ return l_kernel + l_data - 1 	;};
		
		static uint 	compute_n_kernels			(Multi_array<complex_d,2> betas)	{ return betas.get_n_j();};
		static uint 	compute_n_kernels			(np_complex_d betas)				{ py::buffer_info buffer=betas.request();return buffer.shape[0];};
		static uint 	compute_l_kernel			(Multi_array<complex_d,2> betas)	{ return compute_l_kernel_from_l_hc ( betas.get_n_i() );};
		static uint 	compute_l_kernel			(np_complex_d betas)				{ py::buffer_info buffer=betas.request();return compute_l_kernel_from_l_hc (buffer.shape[1]) ;};
		static uint 	compute_l_hc				(Multi_array<complex_d,2> betas)	{ return betas.get_n_i() ;};
		static uint 	compute_l_hc				(np_complex_d betas)				{ py::buffer_info buffer=betas.request();return buffer.shape[1] ;};
		static uint 	compute_n_quads				(uint kernel_conf) ;
        ////////////////////////
        uint compute_l_chunk			( uint l_kernel ,  uint l_fft  )					{ return l_fft - l_kernel + 1 	    ;};
        uint compute_n_chunks			( uint64_t l_data , uint l_chunk )					{ return l_data/l_chunk 		    ;};
        uint compute_l_reste			( uint64_t l_data , uint l_chunk )					{ return l_data%l_chunk 		    ;};
        ////////////////////////
		
		// Python utilities
		static np_complex_d compute_flat_band   (uint l_hc,double dt,double f_min_analog_start,double f_min_analog_stop,double f_max_analog_start,double f_max_analog_stop );
		
        
		//// C++ INTERFACE
		void half_denormalization(); // Undoes half-normalize kernels
        void set_g (Multi_array<complex_d,1> g);
		
		template<class DataType>
		void execute( Multi_array<DataType,1,uint64_t>& data );
        
        /*
        void reset_quads();
        */
        void convolution( Multi_array<int16_t,1,uint64_t>& data );
        
        
		//// Python interface
        void set_g_py (np_complex_d g);
		template<class DataType>
		void execute_py( py::array_t<DataType, py::array::c_style> np_data );
		
	private :
		// Kernels info
		double 									Z ;
		double 									dt ;  
		uint 									l_kernel ;
		uint 									l_hc ;
		uint64_t 								l_data ;
		uint 									kernel_conf ;
		uint 									n_quads ;
		
		const double 							h 					= 6.62607004*pow(10.0,-25.0) ; 
		const double 							units_correction 	= pow(10.0,-(9.0/2.0)) ; 
        double 									prefactor ; // sqrt( 2/ Zh )
   
   
		double 									compute_prefactor( double Z ){ return units_correction*sqrt( 1.0/ (Z*h) );} ; 
		
		// Quadratures info
		uint64_t 								l_valid ;
		uint64_t 								l_full  ;
		
		double 									alpha ;
		uint 									n_kernels ;
		int 									n_threads ;
        
        /////////////////////////////////////////////////////
        uint l_fft; // Lenght(/number of elements) of the FFT used for FFT convolution
        uint l_chunk ; // The length of a chunk
        uint n_chunks ; // The number of chunks
        uint l_reste ; // The length of what didn't fit into chunks
        /////////////////////////////////////////////////////
		
        Multi_array<complex_d,2> 				betas ; // Determine the linear combination of \hat{a}(f)
		Multi_array<complex_d,1> 				g ; // half-complex 
		Multi_array<complex_d,2> 				filters ; // Computed from betas and g
        
		// Kernels
		Multi_array<complex_d,3> 				ks_complex ; // Manages memory for ks_p
		Multi_array<double,3> 					ks ; // quadrature_index , mode/betas_index , freq_index // Uses memory managed by ks_p_complex
                
		Multi_array<double,2> 					half_norms ; //  quadrature_index, mode/betas_index
		// Quadratures
		Multi_array<double,3,Quads_Index_Type> 	quads ; // quadrature_index , mode/betas_index , time_index
		
		fftw_plan 								k_foward ;
		fftw_plan 								k_backward ;
				
		// init sequence
			void checks();
			void checks_n_threads();
			void checks_betas();
			void checks_g();
			void execution_checks( uint64_t l_data );
			
			void prepare_plans();
			void make_kernels();
            
            /////////////////////////////////////////////////////
            /* Constructor sequence */
            void prepare_kernels() ;
            /////////////////////////////////////////////////////
            
		// make_kernels sequence
			void normalize_betas();
				void normalize_a_beta(uint mode_index);
		
			void vanilla_kernels(); // Generates vanilla (analitical filter at Nyquist's frequency) k_p and k_q and outputs then in ks_p[0] and ks_q[0]
				void delta(uint quadrature_index, uint mode_index);
				void vanilla_kp(uint quadrature_index, uint mode_index);
				void vanilla_kq(uint quadrature_index, uint mode_index);
				void vanilla_k_pi_over_4(uint quadrature_index, uint mode_index);
				void vanilla_k_3_pi_over_4(uint quadrature_index, uint mode_index);
                    
			void normalize_for_dfts();
			void apply_windows(); 
			void compute_filters();
			void apply_filters();
			
			void half_normalization();
            
            
        /////////////////////////////////////////////////////
        fftw_plan kernel_plan;
        fftw_plan g_plan;
        fftw_plan h_plan;
        
        // Memory allocated for fft_conv computing
        /* Triple pointers */
        Multi_array<double,2> 		gs ; // [thread_num][frequency] Catches data from data*
        Multi_array<complex_d,2> 	fs ; // [thread_num][frequency] Catches DFT of data
        Multi_array<complex_d,4> 	hs ; // [quads][n_kernel][thread_num][frequency]
        
        /////////////////////////////////////////////////////
			
		// Destructor sequence
		void destroy_plans();	
    
};

#include "../src/TimeQuad.tpp"
