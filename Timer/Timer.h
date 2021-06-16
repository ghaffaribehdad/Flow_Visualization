#pragma once
#include <chrono>

//comment the below line to remove time lapse
#define CHECK_TIME

#ifdef CHECK_TIME

#define TIMELAPSE(function,string) \
	timer.Start();\
	function;\
	timer.Stop();\
	std::printf(  "%s takes %f ms \n", string,timer.GetMilisecondsElapsed());
	//timer.Restart();

#endif
#ifndef CHECK_TIME

#define TIMELAPSE(function,string) function;
	
#endif



class Timer
{
public:
	Timer();
	double GetMilisecondsElapsed();
	void Restart();
	void Reset();
	bool Stop();
	bool Start();
private:
	bool isrunning = false;
#ifdef _WIN32
	std::chrono::time_point<std::chrono::steady_clock> start;
	std::chrono::time_point<std::chrono::steady_clock> stop;
#else //for linux
	std::chrono::time_point<std::chrono::system_clock> start;
	std::chrono::time_point<std::chrono::system_clock> stop;
#endif

};