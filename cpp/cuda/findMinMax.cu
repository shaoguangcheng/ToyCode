#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
//nvcc -arch=compute_11 -code=sm_11 ...

__device__ float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ float atomicMin(float* address, float val)
{
   unsigned int* address_as_ull = ( unsigned int*)address;
   unsigned int  old = *address_as_ull, assumed;
   do{
      assumed = old;
      old = ::atomicCAS(address_as_ull, assumed, 
		__float_as_int(::fminf(val, __int_as_float(assumed))));
   }while(assumed != old);
   return __int_as_float(old);
}


void checkCudaError(cudaError_t error, const char* const filename, const int linenum)
{
  if(error != cudaSuccess){
    fprintf(stderr, "File %s, line %d, CUDA error: %s\n", filename, linenum, cudaGetErrorString(error));
    exit(-1);
  }
}

#define CHECK_CUDA_ERROR(error) checkCudaError(error, __FILE__, __LINE__)

//////////////////////////////////////////////////////

template <typename DataType>
class lessThan{
public:
  bool operator () (const DataType& x, const DataType& y){
    return (x < y) ? true : false;
  }
}; 

template <typename DataType>
class greatThan{
public:
  bool operator () (const DataType& x, const DataType& y){
    return (x > y) ? true : false;
  }
}; 

template <typename DataType, typename Func>
void findOPCPU(const DataType* const array, DataType* val, int N, Func op)
{
  int i = 0;
  for(; i < N; ++i){
    if(op(*val, array[i]))
      *val = array[i];
  }
}

template <typename DataType>
void findMaxMinCPU(const DataType* const array, DataType& min, DataType& max, int N)
{
  min = (array[0] > array[1] ? array[1] : array[0]);
  max = (array[0] > array[1] ? array[0] : array[1]);
  
  DataType tmpMax, tmpMin;
  
  int i = 3;
  for(; i < N; i += 2){
    tmpMax = (array[i-1] > array[i] ? array[i-1] : array[i]);
    tmpMin = (array[i-1] > array[i] ? array[i] : array[i-1]);
    if(tmpMax > max)
      max = tmpMax;
    if(tmpMin < min)
      min = tmpMin;
  }

  if(N&0x01 == 1){
    if(max < array[N-1])
      max = array[N-1];
    if(min > array[N-1])
      min = array[N-1];
  }
}

template <typename DataType>
bool checkCorrectness(const DataType* const array, DataType min, DataType max, int N)
{
  DataType min_, max_;
  
  findMaxMinCPU(array, min_, max_, N);

  return (min - min_ <= 1e-5) && 
    (min - min_ >= -1e-5) &&
    (max - max_ <= 1e-5) &&
    (max - max_ >= -1e-5); 
}

void fillArray(float* array, const int N)
{
  int i = 0; 
  srand(time(NULL));
  for(; i < N; ++i){
    array[i] = (float)(rand()%1000);
  }
}

// here must be careful. blockDim.x must be 2^x
//------------------ very foolish method----------------------
// 1, compute the max and min inside each block, then get the final result on host  

template <typename DataType>
__global__ void findMinMaxKernel_1(const DataType* const array, DataType* min, DataType* max, int N)
{
  __shared__ DataType _min[256];
  __shared__ DataType _max[256];

  int arrayIndex = gridDim.x*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;

  if(arrayIndex >= N)
     return;

  _min[threadIdx.x] = _max[threadIdx.x] = array[arrayIndex];
  __syncthreads();

  int nThreads = blockDim.x;
  DataType tmp;
  while(nThreads > 1){
    nThreads = (nThreads >> 1);
    if(threadIdx.x < nThreads){
      tmp = _min[threadIdx.x + nThreads];
      if(tmp < _min[threadIdx.x])
      	_min[threadIdx.x] = tmp;
      tmp = _max[threadIdx.x + nThreads];
      if(tmp > _max[threadIdx.x])
	_max[threadIdx.x] = tmp;
    }
    __syncthreads();
  }

  if(threadIdx.x == 0){
    min[gridDim.x*blockIdx.y + blockIdx.x] = _min[0];
    max[gridDim.x*blockIdx.y + blockIdx.x] = _max[0];
  }
}

template <typename DataType>
void findMaxMin_1(const DataType* array, DataType* min, DataType* max, const int N)
{
  const int BLOCK_SIZE = 512;
  const int GRID_SIZE  = 128; 
  const int GRID_AREA = GRID_SIZE*GRID_SIZE * sizeof(DataType);
  const int nByte = sizeof(DataType) * N;
  
  DataType* tmpMin = (DataType*)malloc(GRID_AREA);
  DataType* tmpMax = (DataType*)malloc(GRID_AREA);

  DataType* dev_array;
  DataType* dev_tmpMin;
  DataType* dev_tmpMax;

  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_array, nByte));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_tmpMin, GRID_AREA));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_tmpMax, GRID_AREA)); 

  CHECK_CUDA_ERROR(cudaMemcpy(dev_array, array, nByte, cudaMemcpyHostToDevice));
  
  dim3 gridSize(GRID_SIZE, GRID_SIZE, 1);
  dim3 blockSize(BLOCK_SIZE, 1, 1);
 
  findMinMaxKernel_1<<<gridSize, blockSize>>>(dev_array, dev_tmpMin, dev_tmpMax, N);

  CHECK_CUDA_ERROR(cudaMemcpy(tmpMin, dev_tmpMin, GRID_AREA, cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(tmpMax, dev_tmpMax, GRID_AREA, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(dev_array));
  CHECK_CUDA_ERROR(cudaFree(dev_tmpMax));
  CHECK_CUDA_ERROR(cudaFree(dev_tmpMin));

  findOPCPU(tmpMax, max, GRID_SIZE*GRID_SIZE, lessThan<DataType>());
  findOPCPU(tmpMin, min, GRID_SIZE*GRID_SIZE, greatThan<DataType>());

  free(tmpMax);
  free(tmpMin);

  if(checkCorrectness(array, *min, *max, N))
    printf("Correct\n");
  else
    printf("Error\n");
}

//----------------------- naive method ------------------------
// each thread compute one result and use the atomic operation in each thread
// low efficiency
template <typename DataType>
__global__ void findMaxMinKernel_2(const DataType* const array, DataType* max, DataType* min, const int N)
{
  int arrayIndex = gridDim.x*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
  
  if(arrayIndex >= N)
    return;

  DataType tmp = array[arrayIndex];
  
  atomicMax(max, tmp);
  atomicMin(min, tmp);
}

//------------------------- better method ---------------------
template <typename DataType>
__global__ void findMaxMinKernel_3(const DataType* const array, DataType* max, DataType* min, const int N)
{
  __shared__ DataType _max[256];
  __shared__ DataType _min[256];

  int arrayIndex = gridDim.x*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
  
  if(arrayIndex >= N)
    return;

  _max[threadIdx.x] = _min[threadIdx.x] = array[arrayIndex];
  __syncthreads();
  
  DataType tmp;
  int nThreads = blockDim.x;

  while(nThreads > 1){
    nThreads = (nThreads >> 1);
    if(threadIdx.x < nThreads){
       tmp = _max[threadIdx.x+nThreads];
       if(_max[threadIdx.x] < tmp)
          _max[threadIdx.x] = tmp;
       tmp = _min[threadIdx.x+nThreads];
       if(_min[threadIdx.x] > tmp)
          _min[threadIdx.x] = tmp;
    }
    __syncthreads();
  }

  if(threadIdx.x == 0){
    atomicMax(max, _max[0]);
    atomicMin(min, _min[0]);
  }
}

template <typename DataType>
void findMaxMin(const DataType* const array, DataType* max, DataType* min, const int N)
{
  const int BLOCK_SIZE = 512;
  const int GRID_SIZE  = 128; 
  const int nByte = sizeof(DataType) * N;

  DataType* dev_array;
  DataType* dev_max;
  DataType* dev_min;

  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_array, nByte));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_max, sizeof(DataType)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_min, sizeof(DataType)));

  CHECK_CUDA_ERROR(cudaMemcpy(dev_array, array, nByte, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(dev_max, max, sizeof(DataType), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(dev_min, min, sizeof(DataType), cudaMemcpyHostToDevice));
  
  dim3 gridSize(GRID_SIZE, GRID_SIZE, 1);
  dim3 blockSize(BLOCK_SIZE, 1, 1);
 
//  findMaxMinKernel_2<<<gridSize, blockSize>>>(dev_array, dev_max, dev_min, N);
  findMaxMinKernel_3<<<gridSize, blockSize>>>(dev_array, dev_max, dev_min, N);

  CHECK_CUDA_ERROR(cudaMemcpy(min, dev_min, sizeof(DataType), cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(max, dev_max, sizeof(DataType), cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(dev_array));
  CHECK_CUDA_ERROR(cudaFree(dev_max));
  CHECK_CUDA_ERROR(cudaFree(dev_min));

  if(checkCorrectness(array, *min, *max, N))
    printf("Correct\n");
  else
    printf("Error\n");
}

/*
__global__ void test(float* x, float* y)
{
	atomicMax(x, *y);
}

int main()
{
	float x =  3.0, y = 6.0;
	float* dev_x, *dev_y;

	cudaMalloc((void**)&dev_x, sizeof(float));
	cudaMalloc((void**)&dev_y, sizeof(float));

	cudaMemcpy(dev_x, &x, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, &y, sizeof(float), cudaMemcpyHostToDevice);

	test<<<2,2>>>(dev_x, dev_y);

	cudaMemcpy(&x, dev_x, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_x);
	cudaFree(dev_y);

	printf("x = %f", x);

	return 0;	
}
*/
//--------------------------------------------------------------------

int main(int argc, char* argv[])
{
  const int N = 100000000;
  const int nByte = N * sizeof(float);
  float* array = (float*)malloc(nByte);
  float max = 0,  min = 9999;
  
  fillArray(array, N);

//  findMaxMin_1(array, &min, &max, N);
  findMaxMin(array, &max, &min, N);

  free(array);
  
  return 0;
}
