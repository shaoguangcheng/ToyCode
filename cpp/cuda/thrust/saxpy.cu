#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include <iostream>

template <typename T>
class saxpy : public thrust::binary_function<T, T, T>
{
private:
  T factor;

public :
  __host__ __device__
  saxpy(const T& factor) : factor(factor){}

  __host__ __device__
  T operator()(const T& x, const T& y) const{
	return factor*x+y;
  }
};

int main()
{
  const int N = 100000;

  thrust::device_vector<int> x_dev(N);
  thrust::device_vector<int> y_dev(N);

  for(int i = 0; i < N; ++i)
	x_dev[i] = i;

  saxpy<int> func(1);
  
  thrust::fill(y_dev.begin(), y_dev.end(), 2);
  thrust::transform(x_dev.begin(), x_dev.end(), y_dev.begin(),  y_dev.begin(), func);

  thrust::copy(y_dev.begin(), y_dev.end(), std::ostream_iterator<int>(std::cout, " "));

  return 0;
}
