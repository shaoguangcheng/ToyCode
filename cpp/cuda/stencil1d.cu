#include <stdio.h>
#include <stdlib.h>

#define DEBUG
#define RADIUS 2
#define BLOCK_SIZE 10

void checkCudaError(const char* filename, const int linenum)
{
#ifdef DEBUG
  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess){
    printf("File: %s, line: %d, CUDA Error: %s\n", filename, linenum, cudaGetErrorString(error));
    exit(-1);
  }
#endif
}

__global__ void cudaStencil1d(int* in, int* out)
{
  __shared__ int tmp[BLOCK_SIZE+2*RADIUS];

  int threadIndex = threadIdx.x + blockIdx.x*blockDim.x;
  int arrayIndex  = threadIdx.x + RADIUS;

  tmp[arrayIndex] = in[threadIndex];
  if(threadIdx.x < RADIUS){
    if(threadIndex-RADIUS > 0)
       tmp[arrayIndex-RADIUS] = in[threadIndex-RADIUS];
    else
       tmp[arrayIndex-RADIUS] = 0;
    tmp[arrayIndex+BLOCK_SIZE] = in[threadIndex+BLOCK_SIZE];
  }

  __syncthreads();

  int result = 0;
  int offset;
  for(offset = -RADIUS; offset <= RADIUS; ++offset)
    result += tmp[arrayIndex+offset];

  out[threadIndex] = result;
}

void stencil1d(int* in, int* out)
{
  int* dev_in;
  int* dev_out;

  int nByte1 = sizeof(int)*(BLOCK_SIZE+2*RADIUS);
  int nByte2 = sizeof(int)*(BLOCK_SIZE);
	
  cudaError_t error = cudaMalloc((void**)&dev_in, nByte1);
  if(error != cudaSuccess){
    exit(-1);
  }

  error = cudaMalloc((void**)&dev_out, nByte2);
  if(error != cudaSuccess){
    exit(-1);
  }

  error = cudaMemcpy(dev_in, in, nByte1, cudaMemcpyHostToDevice);
  if(error != cudaSuccess){
    exit(-1);
  }
  
  cudaStencil1d<<<1, BLOCK_SIZE>>>(dev_in, dev_out);
  checkCudaError(__FILE__, __LINE__);

  error = cudaMemcpy(out, dev_out, nByte2, cudaMemcpyDeviceToHost);
  if(error != cudaSuccess){
    exit(-1);
  }

  cudaFree(dev_in);
  cudaFree(dev_out);
}

void random_int(int* array, int N)
{
  if(array){
    for(int i = 0; i < N; ++i)
      array[i] = rand()%10;
  }
}

int main(void)
{
  int nByte1 = sizeof(int)*(BLOCK_SIZE+2*RADIUS);
  int nByte2 = sizeof(int)*(BLOCK_SIZE);

  int* in = (int*)malloc(nByte1);
  int* out = (int*)malloc(nByte2);

  random_int(in, BLOCK_SIZE+2*RADIUS);

  stencil1d(in, out);

  for(int i = 0; i < 12; ++i)
      printf("%d ", in[i]);
  printf("\n");

  for(int i = 0; i < 10; ++i)
      printf("%d ", out[i]);
  printf("\n");

  free(in);
  free(out);

  return 0;
}
