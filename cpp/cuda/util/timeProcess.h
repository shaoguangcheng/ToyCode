// g++ -o test test.cpp -I /opt/cuda-5.0/include/ -L /opt/cuda-5.0/lib64/ -lcudart

#ifndef TIMEPROCESS_H
#define TIMEPROCESS_H

#include <time.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

class LogTime
{   
public :
    LogTime() {}
    virtual ~LogTime(){}
	virtual void start(){
		_start_ = clock();
	}

	virtual void end(){
		_end_ = clock();
	}

    virtual double getTime(){
		return (double)(_end_ - _start_); // ms
	}

private :
	clock_t _start_;
	clock_t _end_;    
};

class LogTimeCPU : public LogTime
{
public:
    LogTimeCPU() {}

    void start(){
	timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        _start_ = (long int)t.tv_sec*1000000000 + t.tv_nsec;		
    }
    
    ~LogTimeCPU(){}

	void end(){
       timespec t;
       clock_gettime(CLOCK_MONOTONIC, &t);
       _end_ = (long int)t.tv_sec*1000000000 + t.tv_nsec;	
	}

    double getTime(){
	   return (double)(_end_ - _start_)/(double)1000000; 	   		
    }

private:
    long int _start_;
    long int _end_;
};

class LogTimeGPU : public LogTime // used to test the time consuming for executing kernel function
{
public:
	LogTimeGPU(){
		cudaEventCreate(&_start_);
		cudaEventCreate(&_stop_);
	}
	
	~LogTimeGPU(){
		cudaEventDestroy(_start_);
		cudaEventDestroy(_stop_);
	}

	void start(){
		cudaEventRecord(_start_, 0);		
	}

	void end(){
		cudaEventRecord(_stop_, 0);		
	}

	double getTime(){
        while(cudaEventQuery(_stop_) == cudaErrorNotReady); // because kernel function execution is asyn

		float t;
		cudaEventElapsedTime(&t, _start_, _stop_);
		return (double)t;
	}

private:
	cudaEvent_t _start_;
	cudaEvent_t _stop_;
};

#endif // TIMEPROCESS_H
