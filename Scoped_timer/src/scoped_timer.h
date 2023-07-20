#pragma once

#include<chrono>
#include <inttypes.h>

// see : https://www.youtube.com/watch?v=YG4jexlSAjc
// Credit to youtube channel : TheChernoProject
// Gives time in nano seconds in a uint64_t container
class Timer
{
	// measures the time in between the creation of this object and it's destruction (usually at the end of a scope)
	public: 
		Timer(uint64_t* external_tracker);
		~Timer();
		
		void Stop();
	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
		uint64_t* external_tracker;
};

#include "../src/scoped_timer.cpp"