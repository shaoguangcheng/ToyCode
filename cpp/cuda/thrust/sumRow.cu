#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/generate.h>

#include <iostream>
#include <cstdlib>
#include <ctime>

typedef unsigned long long int uint64;

template <typename T>
class rand_
{
public:
  __host__ __device__
  T operator() (){
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<T> dist(1,100);
	return dist(rng);
  }
};

template <typename T>
class getRowIndex : public thrust::unary_function<T, T>
{
private:
  T C_;
  
public:
  __host__ __device__
  getRowIndex(const T& C) : C_(C){}

  __host__ __device__
  T operator()(const T& index) const {
	return index/C_;
  }
};

int main()
{
  int R = 5;
  int C = 5;

  thrust::device_vector<int> v_device(R*C);
  thrust::device_vector<int> rowSum_device(R);
  thrust::device_vector<int> rowIndex_device(R);

  thrust::generate(v_device.begin(), v_device.end(), rand_<int>());

  thrust::transform_iterator<getRowIndex<int>, thrust::counting_iterator<int> > first = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), getRowIndex<int>(C));
  thrust::transform_iterator<getRowIndex<int>, thrust::counting_iterator<int> > last  = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), getRowIndex<int>(C)) + R*C;

  thrust::reduce_by_key(first, 
						last,
						v_device.begin(),
						rowIndex_device.begin(),
						rowSum_device.begin());

  std::cout << "matrix : " << std::endl;
  for(int i = 0; i < R; ++i){
	for(int j = 0; j < C; ++j){
		std::cout << v_device[i*C+j] << " ";
	}
	std::cout << std::endl;
  }

  std::cout << "sum of Row : " << std::endl;
  for(int i = 0; i < R; ++i)
	std::cout << rowSum_device[i] << " ";
  std::cout << std::endl;
  
  return 0;
}
