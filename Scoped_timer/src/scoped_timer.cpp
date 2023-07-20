// Contructor
Timer::Timer(uint64_t* external_tracker): external_tracker(external_tracker)
{
	m_StartTimePoint = std::chrono::high_resolution_clock::now();
}

// Destructor
Timer::~Timer()
{
	Stop();
}

void Timer::Stop()
{
	auto endTimepoint = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::time_point_cast<std::chrono::nanoseconds>(m_StartTimePoint).time_since_epoch().count();
	auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(endTimepoint).time_since_epoch().count();
	auto duration = end - start ;
	
	// increment external tracker
	*external_tracker += duration ;
}
