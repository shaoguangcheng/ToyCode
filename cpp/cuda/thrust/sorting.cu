#include <iostream>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>

#include "../util/timeProcess.h"

// test the performace of thrust when sorting

typedef unsigned long long int uint64;

/* c++ 11 can not be recognized by nvcc
void randomFill(std::vector<int>& v)
{
  std::random_device randSeed;
  std::default_random_engine e(randSeed());

  typedef std::vector<int>::iterator vecIntIt;
  vecIntIt it;
  for(it = v.begin(); it != v.end(); ++it){
	*it = e();
  }
}
*/

void randomFill(std::vector<int>& v)
{
  std::srand(123456);
  
  const uint64 size = v.size();
  for(int i = 0; i < size; ++i)
     v[i] = rand()%10000;
}

void randomFill(thrust::device_vector<int>& v)
{
  thrust::default_random_engine rng(123456);
   
  const uint64 size = v.size();
  for(int i = 0; i < size; ++i)
     v[i] = rng()%10000;
}

int main(int argc, char* argv[])
{
  const uint64 N = 1000000;

  std::vector<int> v_host(N);
  thrust::device_vector<int> v_device(N);

  randomFill(v_host);
//  randomFill(v_device);
  thrust::copy(v_host.begin(), v_host.end(), v_device.begin());

  LogTimeCPU CPUTimer;
  LogTimeGPU GPUTimer;

  CPUTimer.start();
  std::sort(v_host.begin(), v_host.end());
  CPUTimer.end();

  GPUTimer.start();
  thrust::sort(v_device.begin(), v_device.end());
  GPUTimer.end();

  std::cout << "CPU time : " << CPUTimer.getTime() <<std::endl;
  std::cout << "GPU time : " << GPUTimer.getTime() <<std::endl;

  return 0;
}
