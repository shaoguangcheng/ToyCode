#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <iostream>

#include "../util/timeProcess.h"

typedef unsigned long long int uint64;

int rand_()
{
	static thrust::default_random_engine rng;
    static thrust::uniform_int_distribution<int> dist(0,999);
	return dist(rng);
}

template <typename T>
class plus_
{
public:
	__host__ __device__
	T operator()(const T& x, const T& y)const{
		return x+y;
	}
};

int main()
{
    const uint64 N = 10000000;

	thrust::host_vector<int> v_host(N);
	thrust::device_vector<int> v_device(N);
	
	thrust::generate(v_host.begin(), v_host.end(), rand_);
	thrust::copy(v_host.begin(), v_host.end(), v_device.begin());

	int init = 0;
    
  	LogTimeCPU CPUTimer;
  	LogTimeGPU GPUTimer; 
	
	CPUTimer.start();
	uint64 sum = thrust::reduce(v_host.begin(), v_host.end(), init, plus_<int>());
	CPUTimer.end();

	std::cout << "sum : " << sum << ", CPU time : " << CPUTimer.getTime() << std::endl;

	GPUTimer.start();
	sum = thrust::reduce(v_device.begin(), v_device.end(), init, plus_<int>());
	GPUTimer.end();

	std::cout << "sum : " << sum << ", GPU time : " << GPUTimer.getTime() << std::endl;

	return 0;
}
