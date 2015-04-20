#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda.h>

#include <thrust/for_each.h>

typedef unsigned int uint;
typedef unsigned char uchar;

void checkCudaError(cudaError_t error, const char* const filename, const int linenum)
{
  if(error != cudaSuccess){
    fprintf(stderr, "File %s, line %d, CUDA error: %s\n", filename, linenum, cudaGetErrorString(error));
    exit(-1);
  }
}

#define CHECK_CUDA_ERROR(error) checkCudaError(error, __FILE__, __LINE__)

///////////////////////////////////////////////////////
// CPU methods

void histOnCPU(const uchar* const src, uint* hist, const int N)
{
  int i = 0;
  for(; i < N; ++i){
		hist[src[i]]++;
  }
}

bool checkCorrectness(const uint*  hist1, const uint*  hist2, const int N)
{
  return (memcmp(hist1, hist2, sizeof(uint) * N) == 0) ? true : false;
}

/////////////////////////////////////////////////////
__global__ void histKernel_1(const uchar* src, uint* hist, int N)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;

	if(index >= N)
		return;

  const uchar val = src[index];

  atomicAdd(&hist[val], 1);
}

// once read 32x4 = 128 byte
__global__ void histKernel_2(const uchar* src, uint* hist, int N)
{
  int index = (blockIdx.x*blockDim.x+threadIdx.x)*4;

	if(index >= N)
		return;

  uchar val[4];

  val[0] = src[index];
  val[1] = src[index+1];
  val[2] = src[index+2];
  val[3] = src[index+3];

  atomicAdd(&hist[val[0]], 1);
  atomicAdd(&hist[val[1]], 1);
  atomicAdd(&hist[val[2]], 1);
  atomicAdd(&hist[val[3]], 1);
}

//using shared memory
__shared__ uint histTmp[256];
__global__ void histKernel_3(const uchar* src, uint* hist, int N)
{
  int index = (blockIdx.x*blockDim.x+threadIdx.x)*4;

	if(index >= N)
		return;

	histTmp[threadIdx.x] = 0;

  uchar val[4];

  val[0] = src[index];
  val[1] = src[index+1];
  val[2] = src[index+2];
  val[3] = src[index+3];

	__syncthreads();
  atomicAdd(&histTmp[val[0]], 1);
  atomicAdd(&histTmp[val[1]], 1);
  atomicAdd(&histTmp[val[2]], 1);
  atomicAdd(&histTmp[val[3]], 1);
  __syncthreads();

  atomicAdd(&hist[threadIdx.x], histTmp[threadIdx.x]);
}

void computeHist(const uchar* src, uint* hist, int N)
{
  const int threadPerBlock = 256;
  const int nByteSrc = sizeof(uchar)*N;
  const int nByteHist = sizeof(uint)*256; 

  uchar* dev_src;
  uint* dev_hist;

  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_src, nByteSrc));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_hist, nByteHist));

  CHECK_CUDA_ERROR(cudaMemcpy(dev_src, src, nByteSrc, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemset(dev_hist, 0, nByteHist));

//  histKernel_1<<<(N+threadPerBlock-1)/threadPerBlock, threadPerBlock>>>(dev_src, dev_hist, N);
  histKernel_2<<<(N+4*threadPerBlock-1)/(4*threadPerBlock), threadPerBlock>>>(dev_src, dev_hist, N);
//  histKernel_3<<<(N+threadPerBlock-1)/threadPerBlock, threadPerBlock>>>(dev_src, dev_hist, N);
	
  CHECK_CUDA_ERROR(cudaMemcpy(hist, dev_hist, nByteHist, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(dev_src));
  CHECK_CUDA_ERROR(cudaFree(dev_hist));

  uint* histCPU = (uint*)malloc(nByteHist);

  memset(histCPU, 0, 256*sizeof(int));

  histOnCPU(src, histCPU, N);
  
  if(checkCorrectness(hist, histCPU, 256))
		printf("Correct\n");
  else
		printf("Error\n");
}

void randomFillArray(uchar* src, int N)
{
	srand(time(NULL));
	
	for(int i = 0; i < N; ++i)
		src[i] = (rand()%256);
}

int main()
{
  const int N = 256;
  const int nByte = sizeof(uchar)*N;

  uchar* src = (uchar*)malloc(nByte);
  uint* hist = (uint*)malloc(256*sizeof(uint));

  randomFillArray(src, N);

  computeHist(src, hist, N);

  free(src);
  free(hist);

  return 0;
}
