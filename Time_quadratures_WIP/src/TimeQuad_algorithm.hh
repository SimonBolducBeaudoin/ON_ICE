#pragma once
#include<stdint.h>
#include <Multi_array.h>

class TimeQuad_algorithm
{
	public :
	// Contructor
	TimeQuad_algorithm(){};
	// Destructor
	~TimeQuad_algorithm(){};
	
	void execute( Multi_array<int16_t,1,uint64_t>& data ){};	
	    
    void  update_kernels();
							
	#undef VIRTUAL_EXECUTE
};
