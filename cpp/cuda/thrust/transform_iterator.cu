#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/iterator/transform_iterator.h>

#include <iostream>

template <typename T>
class rand_
{
public:
  __host__ __device__
  T operator() (){
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<T> dist(1, 100);
	return dist(rng);
  }
};

template <typename T>
struct square : public thrust::unary_function<T, T>
{
  __host__ __device__
  T operator() (const T& x) const {
	return x*x;
  }
};

int main()
{
  int N = 1000;

  thrust::device_vector<int> v_device(N);

  thrust::generate(v_device.begin(), v_device.end(), rand_<int>());

  thrust::make_transform_iterator(v_device.begin(), square<int>());

  int sum = thrust::reduce(v_device.begin(), v_device.end(), 0, thrust::plus<int>());

  std::cout << "sum : " << sum << std::endl;

  return 0;
}

