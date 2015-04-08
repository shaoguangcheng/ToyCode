#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* demo to show the usage of share memory*/

#define DEBUG

typedef float dataType;

void checkCudaError(cudaError_t error, const char* filename, const int linenum)
{
  if(error != cudaSuccess){
    printf("File: %s, line: %d, CUDA error: %s\n", filename, linenum, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA_ERROR(error) checkCudaError(error, __FILE__, __LINE__)
#define BLOCK_SIZE 16

typedef struct Matrix
{
  int width;
  int height;
  int stride;
  dataType* element;
}Matrix;

__device__ dataType getElement(Matrix x, int row, int col)
{
  return x.element[row * x.stride + col];
}

__device__ void setElement(Matrix x, int row, int col, dataType val)
{
  x.element[col + row * x.stride] = val;
}

__device__ Matrix getSubMatrix(Matrix x, int row, int col)
{
  Matrix subX;
  
  int row_ = (row + 1) * BLOCK_SIZE, col_ = (col + 1) * BLOCK_SIZE;

  subX.height = ((row_ <= x.height) ? BLOCK_SIZE : x.height%BLOCK_SIZE);
  subX.width  = ((col_ <= x.width)  ? BLOCK_SIZE : x.width%BLOCK_SIZE);
  //subX.height = subX.width = BLOCK_SIZE;
  subX.stride = x.stride;

  subX.element = &x.element[x.stride * row * BLOCK_SIZE + col * BLOCK_SIZE];

  return subX;
}

__device__ int divCeil(int x, int y)
{
  return (x%y == 0) ? x/y : (x/y+1);
}

// no shared memory
__global__ void MatMulKernel(Matrix x, Matrix y, Matrix z)
{
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  dataType ret = 0;
  int i = 0;
  
  if(col < z.width && row < z.height){
     for(i = 0; i < x.width; ++i)
        ret += getElement(x, row, i) * getElement(y, i, col);

     setElement(z, row, col, ret);
  }
}

// shared memory
__global__ void MatMulSharedMemory1(Matrix x, Matrix y, Matrix z)
{
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  Matrix subZ = getSubMatrix(z, blockRow, blockCol);

  int row = threadIdx.y;
  int col = threadIdx.x;
  if(row >= subZ.height || col >= subZ.width)
      return;
  dataType val = 0.0;

  int i = 0;
  int size = divCeil(x.width, BLOCK_SIZE);
  for(; i < size; ++i){
    Matrix subX = getSubMatrix(x, blockRow, i);
    Matrix subY = getSubMatrix(y, i, blockCol);
    
    __shared__ dataType tmpX[BLOCK_SIZE][BLOCK_SIZE]; 
    __shared__ dataType tmpY[BLOCK_SIZE][BLOCK_SIZE];

    tmpX[row][col] = getElement(subX, row, col);
    tmpY[row][col] = getElement(subY, row, col);

    __syncthreads();

    int j = 0;
    for(;j < subX.width; ++j)
      val += tmpX[row][j] * tmpY[j][col];

    __syncthreads();
  }

  setElement(subZ, row, col, val);
}

bool checkCorrectness(Matrix x, Matrix y, Matrix z)
{
  Matrix z_;

  z_.width = z.width;
  z_.height = z.height;
  z_.stride = z.stride;
  z_.element = (dataType*)malloc(sizeof(dataType) * z_.width * z_.height);
  
  for(int i = 0; i < z_.height; ++i){
    for(int j = 0; j < z_.width; ++j){
      dataType val = 0;
      for(int k = 0; k < x.width; ++k)
		val += x.element[i*x.stride+k] * y.element[k*y.stride+j];
      z_.element[j + i * z_.stride] = val;
    }
  }

  bool flag;

  if(memcmp(z_.element, z.element, z_.width * z_.height * sizeof(dataType)) == 0)
    flag = true;
  else
    flag = false;

  free(z_.element);

  return flag;
} 

void MatMul(Matrix x, Matrix y, Matrix z)
{
  Matrix dev_x, dev_y, dev_z;
  
  dev_x.width  = x.width;
  dev_x.height = x.height;
  dev_x.stride = x.stride;

  dev_y.width  = y.width;
  dev_y.height = y.height;
  dev_y.stride = y.stride;

  dev_z.width  = z.width;
  dev_z.height = z.height; 
  dev_z.stride = z.stride;

  int nByte = sizeof(dataType) * dev_x.width * dev_x.height; 

  CHECK_CUDA_ERROR(cudaMalloc((void**)(&dev_x.element), nByte));
  CHECK_CUDA_ERROR(cudaMemcpy(dev_x.element, x.element, nByte, cudaMemcpyHostToDevice));

  nByte = sizeof(dataType) * dev_y.width * dev_y.height; 
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_y.element, nByte));
  CHECK_CUDA_ERROR(cudaMemcpy(dev_y.element, y.element, nByte, cudaMemcpyHostToDevice));

  nByte = sizeof(dataType) * dev_z.width * dev_z.height; 
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_z.element, nByte));

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(dev_z.width/dimBlock.x + (dev_z.width%dimBlock.x == 0 ? 0 : 1), dev_z.height/dimBlock.y + (dev_z.height%dimBlock.y == 0 ? 0 : 1));

  //MatMulKernel<<<dimGrid, dimBlock>>>(dev_x, dev_y, dev_z);
  MatMulSharedMemory1<<<dimGrid, dimBlock>>>(dev_x, dev_y, dev_z);

  CHECK_CUDA_ERROR(cudaMemcpy(z.element, dev_z.element, nByte, cudaMemcpyDeviceToHost));

  if(checkCorrectness(x, y, z) == false)
    printf("Error occur\n");
  else
    printf("Correct\n");

  CHECK_CUDA_ERROR(cudaFree(dev_x.element));
  CHECK_CUDA_ERROR(cudaFree(dev_y.element));
  CHECK_CUDA_ERROR(cudaFree(dev_z.element));
}

void randomFillMatrix(Matrix x)
{
  int size = x.width*x.height;
  
  for(int i = 0; i < size; ++i){
    srand(time(NULL)+i);
    x.element[i] = rand()%100;
  }
}

int main(void)
{
  Matrix x, y, z;

  x.width = x.height = x.stride = 32;
  y.width = y.height = y.stride = 32;
  z.width = z.height = z.stride = 32;

  int nByte = sizeof(dataType) * x.width * x. height;

  x.element = (dataType*)malloc(nByte);
  y.element = (dataType*)malloc(nByte);
  z.element = (dataType*)malloc(nByte);

  randomFillMatrix(x);
  randomFillMatrix(y);

  MatMul(x, y, z);

  free(x.element);
  free(y.element);
  free(z.element);

  return 0;
}
