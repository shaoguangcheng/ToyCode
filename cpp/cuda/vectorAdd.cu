#include <stdio.h>
#include <stdlib.h>

#define DEBUG

__global__ void add(const int* x, const int* y, int* z, const int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < n)
    z[index] = x[index] + y[index];
}

void checkCudaError(const char* filename, const int linenum)
{
  cudaThreadSynchronize();
  cudaError error = cudaGetLastError();
  if(error != cudaSuccess){
    printf("File: %s, line: %d, CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));
    exit(-1);
  }
}

void addVec(const int* x, const int*y, int* z, const int N)
{
  const int THREAD_PER_BLOCK = 512;
  const int nByte = N * sizeof(int);

  int *dev_x, *dev_y, *dev_z;

  cudaMalloc((void**)(&dev_x), nByte);
  cudaMalloc((void**)(&dev_y), nByte);
  cudaMalloc((void**)(&dev_z), nByte);

  cudaMemcpy(dev_x, x, nByte, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, nByte, cudaMemcpyHostToDevice);
  
  add<<<(N + THREAD_PER_BLOCK - 1)/THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(dev_x, dev_y, dev_z, N);

  cudaMemcpy(z, dev_z, nByte, cudaMemcpyDeviceToHost);

  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(dev_z);
}

void random_int(int* array, int N)
{
  if(array){
    for(int i = 0; i < N; ++i)
      array[i] = rand()/1000;
  }
}

int main(void)
{
  const int N = 512*512;
  const int nByte = sizeof(int) * N;
  
  int *x = (int*)malloc(nByte);
  int *y = (int*)malloc(nByte);
  int *z = (int*)malloc(nByte);
  
  random_int(x, N);
  random_int(y, N);

  addVec(x, y, z, N);

  free(x);
  free(y);
  free(z);

  return 0;
}
