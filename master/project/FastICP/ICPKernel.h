#ifndef ICPKERNEL_H__
#define ICPKERNEL_H__

#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "defs.h"


// Parallel reduction
//----------------------------------------------------------------------------
template<int NumPts, int Dim, int Offset>
__device__ void ParallelReduction(int tid, float* SM)
{
	int pos = tid*Dim;

	if ( NumPts >= 512 )
		{
			const int Threshold = 256;
			if ( tid < Threshold )
			{
				SM[pos + Offset] += SM[pos + Threshold*Dim + Offset];
			}
			__syncthreads();
		}

	if ( NumPts >= 256 )
		{
			const int Threshold = 128;
			if ( tid < Threshold )
			{
				SM[pos + Offset] += SM[pos + Threshold*Dim + Offset];
			}
			__syncthreads();
		}

	if ( NumPts >= 128 )
		{
			const int Threshold = 64;
			if ( tid < Threshold )
			{
				SM[pos + Offset] += SM[pos + Threshold*Dim + Offset];
			}
			__syncthreads();
		}

	if ( NumPts >= 64 )
		{
			const int Threshold = 32;
			if ( tid < Threshold )
			{
				SM[pos + Offset] += SM[pos + Threshold*Dim + Offset];
			}
			__syncthreads();
		}

	if ( NumPts >= 32 )
		{
			const int Threshold = 16;
			if ( tid < Threshold )
			{
				SM[pos + Offset] += SM[pos + Threshold*Dim + Offset];
			}
			__syncthreads();
		}

	if ( NumPts >= 16 )
		{
			const int Threshold = 8;
			if ( tid < Threshold )
			{
				SM[pos + Offset] += SM[pos + Threshold*Dim + Offset];
			}
			__syncthreads();
		}

	if ( NumPts >= 8 )
		{
			const int Threshold = 4;
			if ( tid < Threshold )
			{
				SM[pos + Offset] += SM[pos + Threshold*Dim + Offset];
			}
			__syncthreads();
		}

	if ( NumPts >= 4 )
		{
			const int Threshold = 2;
			if ( tid < Threshold )
			{
				SM[pos + Offset] += SM[pos + Threshold*Dim + Offset];
			}
			__syncthreads();
		}

	if ( NumPts >= 2 )
		{
			const int Threshold = 1;
			if ( tid < Threshold )
			{
				SM[pos + Offset] += SM[pos + Threshold*Dim + Offset];
			}
			__syncthreads();
		}
}


// Compute centroid for a given dataset
//----------------------------------------------------------------------------
template<unsigned int NumPts, unsigned int Dim>
__global__ void kernelComputeCentroid(float* points, float* out, unsigned long* correspondences)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if ( tid >= NumPts ) return;

	const unsigned int NumPtsPerBlock = CUDA_THREADS_PER_BLOCK < NumPts ? CUDA_THREADS_PER_BLOCK : NumPts;

	__shared__ float TmpShared[NumPtsPerBlock * 3];

	int posPoints;
	if (correspondences == NULL)
		posPoints = tid*Dim;
	else
		posPoints = correspondences[tid]*Dim;

	int posTmp = threadIdx.x*3;

	// Copy data (x, y and z coordinates of all points)
	TmpShared[posTmp+0] = points[posPoints+0];
	TmpShared[posTmp+1] = points[posPoints+1];
	TmpShared[posTmp+2] = points[posPoints+2];

	__syncthreads();
	
	
	// Perform parallel reduction for x, y and z coordinate	
	ParallelReduction<NumPtsPerBlock, 3, 0>(threadIdx.x, TmpShared); // X
	ParallelReduction<NumPtsPerBlock, 3, 1>(threadIdx.x, TmpShared); // Y
	ParallelReduction<NumPtsPerBlock, 3, 2>(threadIdx.x, TmpShared); // Z

	// Store values to global memory
	if (NumPts < 3)
	{
		if (threadIdx.x == 0)
		{
			out[0] = TmpShared[0] / NumPtsPerBlock;
			out[1] = TmpShared[1] / NumPtsPerBlock;
			out[2] = TmpShared[2] / NumPtsPerBlock;
		}
	}
	else if (threadIdx.x < 3)
	{
		out[threadIdx.x + blockIdx.x*3] = TmpShared[threadIdx.x] / NumPtsPerBlock;
	}
}


// Build M matrices (3x3) for each corresponding pair of points
//----------------------------------------------------------------------------
template<unsigned int NumPts, unsigned int Dim>
__global__ void kernelBuildMMatrices(float* moving, float* fixed, float* centroidMoving, float* centroidFixed, float* out, unsigned long* correspondences)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if ( tid >= NumPts ) return;

	const unsigned int NumPtsPerBlock = CUDA_THREADS_PER_BLOCK < NumPts ? CUDA_THREADS_PER_BLOCK : NumPts;

	float a[3];
	a[0] = moving[tid*Dim+0] - centroidMoving[0];
	a[1] = moving[tid*Dim+1] - centroidMoving[1];
	a[2] = moving[tid*Dim+2] - centroidMoving[2];

	float b[3];
	unsigned long NNIdx = correspondences[tid];
	b[0] = fixed[NNIdx*Dim+0] - centroidFixed[0];
	b[1] = fixed[NNIdx*Dim+1] - centroidFixed[1];
	b[2] = fixed[NNIdx*Dim+2] - centroidFixed[2];

	float M[3][3] = {0,0,0, 0,0,0, 0,0,0};

	// Accumulate the products a*T(b) into the matrix M
	for (int j = 0; j < 3; ++j) 
	{
		M[j][0] += a[j]*b[0];
		M[j][1] += a[j]*b[1];
		M[j][2] += a[j]*b[2];
	}


	__shared__ float TmpShared[NumPtsPerBlock * 9]; // Shared memory for all 3x3 M matrices of this particular block

	// Linearize matrix (shared memory)
	float* startIdx = &TmpShared[threadIdx.x * 9];
	startIdx[0] = M[0][0];
	startIdx[1] = M[0][1];
	startIdx[2] = M[0][2];
	startIdx[3] = M[1][0];
	startIdx[4] = M[1][1];
	startIdx[5] = M[1][2];
	startIdx[6] = M[2][0];
	startIdx[7] = M[2][1];
	startIdx[8] = M[2][2];

	__syncthreads();


	// Reduce results
	ParallelReduction<NumPtsPerBlock, 9, 0>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 1>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 2>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 3>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 4>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 5>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 6>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 7>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 8>(threadIdx.x, TmpShared);

	// Store sum of matrices of this block to global memory
	if (NumPts < 9)
	{
		if (threadIdx.x == 0)
		{
			out[0] = TmpShared[0];
			out[1] = TmpShared[1];
			out[2] = TmpShared[2];
			out[3] = TmpShared[3];
			out[4] = TmpShared[4];
			out[5] = TmpShared[5];
			out[6] = TmpShared[6];
			out[7] = TmpShared[7];
			out[8] = TmpShared[8];
		}
	}
	else if (threadIdx.x < 9)
	{
		out[threadIdx.x + blockIdx.x*9] = TmpShared[threadIdx.x];
	}
}


// Reduce previously computed 3x3 M matrices
//----------------------------------------------------------------------------
template<unsigned int NumPts>
__global__ void kernelReduceMMatrices(float* matrices, float* out)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if ( tid >= NumPts ) return;

	const unsigned int NumPtsPerBlock = CUDA_THREADS_PER_BLOCK < NumPts ? CUDA_THREADS_PER_BLOCK : NumPts;

	__shared__ float TmpShared[NumPtsPerBlock * 9]; // For all 3x3 M matrices of this particular block

	int posMatrices = tid*9;
	int posTmp = threadIdx.x*9;

	// Copy data (x, y and z coordinates of all points)
	TmpShared[posTmp+0] = matrices[posMatrices+0];
	TmpShared[posTmp+1] = matrices[posMatrices+1];
	TmpShared[posTmp+2] = matrices[posMatrices+2];
	TmpShared[posTmp+3] = matrices[posMatrices+3];
	TmpShared[posTmp+4] = matrices[posMatrices+4];
	TmpShared[posTmp+5] = matrices[posMatrices+5];
	TmpShared[posTmp+6] = matrices[posMatrices+6];
	TmpShared[posTmp+7] = matrices[posMatrices+7];
	TmpShared[posTmp+8] = matrices[posMatrices+8];

	__syncthreads();
	
	
	// Reduce
	ParallelReduction<NumPtsPerBlock, 9, 0>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 1>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 2>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 3>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 4>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 5>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 6>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 7>(threadIdx.x, TmpShared);
	ParallelReduction<NumPtsPerBlock, 9, 8>(threadIdx.x, TmpShared);

	// Store values to global memory
	if (NumPts < 9)
	{
		if (threadIdx.x == 0)
		{
			out[0] = TmpShared[0];
			out[1] = TmpShared[1];
			out[2] = TmpShared[2];
			out[3] = TmpShared[3];
			out[4] = TmpShared[4];
			out[5] = TmpShared[5];
			out[6] = TmpShared[6];
			out[7] = TmpShared[7];
			out[8] = TmpShared[8];
		}
	}
	else if (threadIdx.x < 9)
	{
		out[threadIdx.x + blockIdx.x*9] = TmpShared[threadIdx.x];
	}
}


// Transform points w.r.t. a given 4x4 homogeneous transformation matrix m
//----------------------------------------------------------------------------
template<unsigned int NumPts, unsigned int Dim>
__global__ void kernelTransformPoints3D(float* points, float* m)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if ( tid >= NumPts ) return;

	// Pull transformation matrix in parallel
	__shared__ float mShared[16];

	if (threadIdx.x < 16)
		mShared[threadIdx.x] = m[threadIdx.x];

	__syncthreads();

	if (tid >= NumPts)
		return;

	// Modified index for operating on a point of dimension Dim
	int pos = tid * Dim;

	// Get point data
	float p0 = points[pos+0];
	float p1 = points[pos+1];
	float p2 = points[pos+2];

	// Compute transformation using the homogeneous transformation matrix m
	float x = mShared[0]  * p0 + mShared[1]  * p1 + mShared[2]  * p2 + mShared[3];
	float y = mShared[4]  * p0 + mShared[5]  * p1 + mShared[6]  * p2 + mShared[7];
	float z = mShared[8]  * p0 + mShared[9]  * p1 + mShared[10] * p2 + mShared[11];
	
	// We do not have to compute the homogeneous coordinate w and divide x, y and z by w,
	// since w will always be 1 in this ICP implementation

	__syncthreads();


	// Update coordinates
	points[pos+0] = x;
	points[pos+1] = y;
	points[pos+2] = z;
}


// Basically a 4x4 matrix-matrix multiplication
//----------------------------------------------------------------------------
__global__ void kernelAccumulateMatrix(float* accu, float* m)
{
	__shared__ float accuShared[16];
	__shared__ float mShared[16];

	// Fetch data
	accuShared[threadIdx.x] = accu[threadIdx.x];
	mShared[threadIdx.x] = m[threadIdx.x];

	if (threadIdx.x >= 4) return;

	__syncthreads();

	// Compute matrix-matrix (4x4) multiplication
	float sum;
	for (int j = 0; j < 4; ++j)
	{
		sum = 0;
		for (int k = 0; k < 4; ++k)
		{
			sum += mShared[threadIdx.x*4+k] * accuShared[k*4+j];
		}
		accu[threadIdx.x*4+j] = sum;
	}
}


#endif // ICPKERNEL_H__