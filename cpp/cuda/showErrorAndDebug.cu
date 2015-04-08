#include <stdio.h>
#include <stdlib.h>

#define DEBUG

inline void checkCudaError(const char* filename, const int linenum)
{
#ifdef DEBUG
  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess){
    printf("File: %s, line: %d, CUDA error : %s\n", filename, linenum, cudaGetErrorString(error));
    exit(-1);
  }
#endif			
}

__global__ void foo(int *ptr)
{
  *ptr = 7;
}

int main(void)
{
  foo<<<1,1>>>(0);
  
  checkCudaError(__FILE__, __LINE__);
  
  return 0;
}
