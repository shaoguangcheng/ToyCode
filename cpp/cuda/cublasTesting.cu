#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

//#include <boost/shared_ptr.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// using boost::shared_ptr;

typedef float DataType;

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

#define CUBLAS_CHECK(condition) \
	do{ \
		cublasStatus_t status = condition;\
		if(status != CUBLAS_STATUS_SUCCESS){ \
			printf("File: %s, Line: %d, Error: %s\n", \
					__FILE__, __LINE__, cublasGetErrorString(status)); \
		}\
	}while(0)

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
	if(error != cudaSuccess){ \
		printf("File: %s, Line: %d, Error: %s\n", \
				__FILE__, __LINE__, cudaGetErrorString(error)); \
	} \
  } while (0)

void printArray(DataType* array, int M, int N, const char* prompt = "")
{
	if(NULL == array || M <= 0 || N <= 0){
		printf("Array is NULL\n");
		exit(EXIT_FAILURE);
	}

	printf("%s :", prompt);
	for(int i = 0; i < M; ++i){
		for(int j = 0; j < N; ++j)
			printf("%f\t", array[i*N+j]);
		printf("\n");
	}
}

template <class ForwardIterator, class DataType>
void sequence_fill(ForwardIterator first, ForwardIterator end, DataType init, DataType delta = DataType(1))
{
	while(first != end){
		*first = init;
		++first;
		init += delta;
	}
}
/////////////////////////////////////////////////////////
/*
class warp
{
private:
	warp() : cublas_handle_(NULL){
		CUBLAS_CHECK(cublasCreate(&cublas_handle_));
	}

	warp(const warp&);
	warp& operator=(const warp&);

public:
  ~warp() {
	CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  }

  inline static warp& get(){
	if(!singleton.get())
	  singleton.reset(new warp());
	return *singleton;
  }
  
  inline static cublasHandle_t cublas_handle(){
	return get().cublas_handle_;
  } 

private:
  static shared_ptr<warp> singleton;
	
	cublasHandle_t cublas_handle_;
};
*/
//////////////////////////////////////////////////////////

// Y = alpha*X + Y
void simpleAxpy(DataType* X, DataType alpha, DataType* Y, int N)
{
	for(int i = 0; i < N; ++i){
		Y[i] = alpha*X[i] + Y[i];
	}
}

// C = alpha*A*B + beta*C
void simpleGemm(DataType* A, DataType* B, DataType* C, DataType alpha, DataType beta, 
	int m, int k, int n)
{
	DataType tmp;

	for(int i = 0; i < m; ++i){
		for(int j = 0; j < n; ++j){
			tmp = (DataType)0;
			for(int s = 0; s < k; ++s)
				tmp += A[i*k+s]*B[s*n+j];
			C[i*n+j] = alpha*tmp + beta*C[i*n+j];
		}
	}
}

void randomFillArray(DataType* array, int N)
{
  srand(time(NULL));
  for(int i = 0; i < N; ++i){
	array[i] = DataType(rand()%100);
  }
}

/////////////////////////////////////////////////////////////////////////

bool checkCorrectnessAxpy(DataType* X, DataType alpha, DataType* Y, int N, DataType* Y_)
{
  simpleAxpy(X, alpha, Y, N);
  for(int i = 0; i < N; ++i){
	if((Y[i]-Y_[i] > 1e-5) || (Y[i]-Y_[i]<-1e-5))
	  return false;
  }
  return true;
}

bool checkCorrectnessGemm(DataType* A, DataType* B, DataType* C, DataType alpha, DataType beta, 
	int m, int k, int n, DataType* C_)
{
	simpleGemm(A, B, C, alpha, beta, m, k, n);
	const int num = m*n;
	for(int i = 0; i < num; ++i){
		if((C[i] - C_[i] > 1e-5) || (C[i] - C_[i] < -1e-5))
			return false;
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
 void cublasAxpy(DataType* X, DataType alpha, DataType* Y, int N)
 {
   cublasHandle_t cublas_handle_;
   CUBLAS_CHECK(cublasCreate(&cublas_handle_));

   DataType* dev_X;
   DataType* dev_Y;

   const int nByte = N*sizeof(DataType);

   CUDA_CHECK(cudaMalloc((void**)&dev_X, nByte));
   CUDA_CHECK(cudaMalloc((void**)&dev_Y, nByte));

   //  CUBLAS_CHECK(cublasSetVector(N, sizeof(DataType), X, 1, dev_X, 1));
   // CUBLAS_CHECK(cublasSetVector(N, sizeof(DataType), Y, 1, dev_Y, 1));

   CUDA_CHECK(cudaMemcpy(dev_X, X, nByte, cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(dev_Y, Y, nByte, cudaMemcpyHostToDevice));

   CUBLAS_CHECK(cublasSaxpy(cublas_handle_, N, &alpha, dev_X, 1, dev_Y, 1));

   DataType* Y_ = (DataType*)malloc(nByte);
   memcpy(Y_, Y, nByte);

   //   CUBLAS_CHECK(cublasGetVector(N, sizeof(DataType), dev_Y, 1, Y_, 1));
   CUDA_CHECK(cudaMemcpy(Y_, dev_Y, nByte, cudaMemcpyDeviceToHost));

   CUDA_CHECK(cudaFree(dev_X));
   CUDA_CHECK(cudaFree(dev_Y));
   CUBLAS_CHECK(cublasDestroy(cublas_handle_));

   if(checkCorrectnessAxpy(X, alpha, Y, N, Y_))
	 printf("Correct\n");
   else
	 printf("Error\n");
 }

void cublasGemm(DataType* A, DataType* B, DataType* C, DataType alpha, DataType beta, 
	int m, int k, int n)
{
	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	DataType* dev_A;
	DataType* dev_B;
	DataType* dev_C;

	CUDA_CHECK(cudaMalloc((void**)&dev_A, sizeof(DataType)*m*k));
	CUDA_CHECK(cudaMalloc((void**)&dev_B, sizeof(DataType)*k*n));
	CUDA_CHECK(cudaMalloc((void**)&dev_C, sizeof(DataType)*m*n));

	CUBLAS_CHECK(cublasSetVector(m*k, sizeof(DataType), A, 1, dev_A, 1));
	CUBLAS_CHECK(cublasSetVector(k*n, sizeof(DataType), B, 1, dev_B, 1));
	CUBLAS_CHECK(cublasSetVector(m*n, sizeof(DataType), C, 1, dev_C, 1));

	CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
							 &alpha, dev_B, n, dev_A, k, &beta, dev_C, n)); // cublas using fortran order

	CUBLAS_CHECK(cublasGetVector(m*n, sizeof(DataType), dev_C, 1, C, 1));

	CUDA_CHECK(cudaFree(dev_A));
	CUDA_CHECK(cudaFree(dev_B));
	CUDA_CHECK(cudaFree(dev_C));

	CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

 void testAxpy()
 {
   const int N = 100000;
   const int nByte = sizeof(DataType) * N;
   DataType alpha = DataType(1.0);

   DataType *X = (DataType*)malloc(nByte);
   DataType *Y = (DataType*)malloc(nByte);

   randomFillArray(X, N);
   randomFillArray(Y, N);

   cublasAxpy(X, alpha, Y, N);

   free(X);
   free(Y);
}

void testGemm(void)
{
	DataType A[3*4];
	DataType B[4*5];
	DataType C[3*5];
	DataType C_[3*5];

	sequence_fill(A, A+12, 1);
	sequence_fill(B, B+20, 1, 1);
	sequence_fill(C, C+15, 0, 0);
	sequence_fill(C_, C_+15, 0, 0);

	cublasGemm(A, B, C, 1, 1, 3, 4, 5);
	simpleGemm(A, B, C_, 1, 1, 3, 4, 5);	

	printArray(A, 3, 4, "A");
	printArray(B, 4, 5, "B");
	printArray(C, 3, 5, "C");
	printArray(C_, 3, 5, "C_");
}

int main()
{
	testGemm();

	return 0;
}