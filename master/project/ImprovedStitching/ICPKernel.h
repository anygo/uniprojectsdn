#ifndef ICPKERNEL_H__
#define ICPKERNEL_H__

#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "defs.h"


// Parallel reduction
//----------------------------------------------------------------------------
template<unsigned int NumPts, unsigned int Dim, unsigned int Offset>
__device__ void ParallelReduction(int tid, float* SM)
{
	// Compute index position for current thread
	const int pos = tid*Dim;

	// Run reduction
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


// Parallel reduction
//----------------------------------------------------------------------------
template<unsigned int NumPts, unsigned int Dim, unsigned int Offset>
__device__ void ParallelReductionFloat3(int tid, float* SM)
{
	const int DimDiv3 = Dim/3;

	// Compute index position for current thread
	const int pos = tid*DimDiv3;

	float3* SMFloat3 = (float3*)SM;

	// Run reduction
	if ( NumPts >= 512 )
	{
		const int Threshold = 256;
		if ( tid < Threshold )
		{
			float3 First = SMFloat3[pos + Offset];
			float3 Second = SMFloat3[pos + Threshold*DimDiv3 + Offset];
			SMFloat3[pos + Offset] = make_float3(First.x+Second.x, First.y+Second.y, First.z+Second.z);
		}
		__syncthreads();
	}

	if ( NumPts >= 256 )
	{
		const int Threshold = 128;
		if ( tid < Threshold )
		{
			float3 First = SMFloat3[pos + Offset];
			float3 Second = SMFloat3[pos + Threshold*DimDiv3 + Offset];
			SMFloat3[pos + Offset] = make_float3(First.x+Second.x, First.y+Second.y, First.z+Second.z);
		}
		__syncthreads();
	}

	if ( NumPts >= 128 )
	{
		const int Threshold = 64;
		if ( tid < Threshold )
		{
			float3 First = SMFloat3[pos + Offset];
			float3 Second = SMFloat3[pos + Threshold*DimDiv3 + Offset];
			SMFloat3[pos + Offset] = make_float3(First.x+Second.x, First.y+Second.y, First.z+Second.z);
		}
		__syncthreads();
	}

	if ( NumPts >= 64 )
	{
		const int Threshold = 32;
		if ( tid < Threshold )
		{
			float3 First = SMFloat3[pos + Offset];
			float3 Second = SMFloat3[pos + Threshold*DimDiv3 + Offset];
			SMFloat3[pos + Offset] = make_float3(First.x+Second.x, First.y+Second.y, First.z+Second.z);
		}
		__syncthreads();
	}

	if ( NumPts >= 32 )
	{
		const int Threshold = 16;
		if ( tid < Threshold )
		{
			float3 First = SMFloat3[pos + Offset];
			float3 Second = SMFloat3[pos + Threshold*DimDiv3 + Offset];
			SMFloat3[pos + Offset] = make_float3(First.x+Second.x, First.y+Second.y, First.z+Second.z);
		}
		__syncthreads();
	}

	if ( NumPts >= 16 )
	{
		const int Threshold = 8;
		if ( tid < Threshold )
		{
			float3 First = SMFloat3[pos + Offset];
			float3 Second = SMFloat3[pos + Threshold*DimDiv3 + Offset];
			SMFloat3[pos + Offset] = make_float3(First.x+Second.x, First.y+Second.y, First.z+Second.z);
		}
		__syncthreads();
	}

	if ( NumPts >= 8 )
	{
		const int Threshold = 4;
		if ( tid < Threshold )
		{
			float3 First = SMFloat3[pos + Offset];
			float3 Second = SMFloat3[pos + Threshold*DimDiv3 + Offset];
			SMFloat3[pos + Offset] = make_float3(First.x+Second.x, First.y+Second.y, First.z+Second.z);
		}
		__syncthreads();
	}

	if ( NumPts >= 4 )
	{
		const int Threshold = 2;
		if ( tid < Threshold )
		{
			float3 First = SMFloat3[pos + Offset];
			float3 Second = SMFloat3[pos + Threshold*DimDiv3 + Offset];
			SMFloat3[pos + Offset] = make_float3(First.x+Second.x, First.y+Second.y, First.z+Second.z);
		}
		__syncthreads();
	}

	if ( NumPts >= 2 )
	{
		const int Threshold = 1;
		if ( tid < Threshold )
		{
			float3 First = SMFloat3[pos + Offset];
			float3 Second = SMFloat3[pos + Threshold*DimDiv3 + Offset];
			SMFloat3[pos + Offset] = make_float3(First.x+Second.x, First.y+Second.y, First.z+Second.z);
		}
		__syncthreads();
	}
}


// Compute centroid for a given dataset
//----------------------------------------------------------------------------
template<unsigned int NumPts, unsigned int Dim, unsigned int BlockSizeX>
__global__ void kernelComputeCentroid(float* points, float* out, unsigned long* correspondences)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if ( tid >= NumPts ) return;

	tid += blockIdx.x*blockDim.x;

	// Compute number of points per block, check if NumPts is less than blocksize
	const unsigned int NumPtsPerBlock = BlockSizeX < NumPts ? BlockSizeX : NumPts;

	__shared__ float TmpShared[NumPtsPerBlock * 3];

	// Compute indices for current thread (for copying data)
	int posPoints, posPoints2;
	if (correspondences == NULL)
	{
		posPoints = tid;//*Dim;
		posPoints2 = (tid+blockDim.x);//*Dim;
	}
	else
	{
		posPoints = correspondences[tid];//*Dim;
		posPoints2 = correspondences[tid+blockDim.x];//*Dim;
	}

	// Copy data (x, y and z coordinates of all points)
	float3* PointsFloat3 = (float3*)points;
	float3* TmpSharedFloat3 = (float3*)TmpShared;
	float3 First = PointsFloat3[Dim == 6 ? posPoints*2 : posPoints]; // If we still are in 6D, we need to multiply the position by 2, since we assume float3
	float3 Second = PointsFloat3[Dim == 6 ? posPoints2*2 : posPoints2];
	TmpSharedFloat3[threadIdx.x] = make_float3(First.x + Second.x, First.y + Second.y, First.z + Second.z);

	__syncthreads();
	
	
	// Perform parallel reduction for x, y and z coordinate	
	ParallelReductionFloat3<NumPtsPerBlock, 3, 0>(threadIdx.x, TmpShared); // X

	// Store values to global memory
	const float Factor = 1.f / ( (float)BlockSizeX*2.f);
	if (NumPts < 3)
	{
		// If we have less than 3 threads left running, let the first thread do all the work
		if (threadIdx.x == 0)
		{
			out[0 + blockIdx.x*3] = TmpShared[0] * Factor;
			out[1 + blockIdx.x*3] = TmpShared[1] * Factor;
			out[2 + blockIdx.x*3] = TmpShared[2] * Factor;
		}
	}
	else if (threadIdx.x < 3)
	{
		// If there are enough threads available, let thread 0, 1 and 2 copy the data in parallel
		out[threadIdx.x + blockIdx.x*3] = TmpShared[threadIdx.x] * Factor;
	}
}


// Build M matrices (3x3) for each corresponding pair of points
//----------------------------------------------------------------------------
template<unsigned int NumPts, unsigned int Dim>
__global__ void kernelBuildMMatrices(float* moving, float* fixed, float* centroidMoving, float* centroidFixed, float* out, unsigned long* correspondences)
{
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if ( tid >= NumPts ) return;

	const unsigned int NumPtsPerBlock = CUDA_THREADS_PER_BLOCK < NumPts ? CUDA_THREADS_PER_BLOCK : NumPts;

	float a[3];
	float3 CentroidMoving = *((float3*)centroidMoving);
	a[0] = moving[tid*Dim+0] - CentroidMoving.x;
	a[1] = moving[tid*Dim+1] - CentroidMoving.y;
	a[2] = moving[tid*Dim+2] - CentroidMoving.z;

	float b[3];
	float3 CentroidFixed = *((float3*)centroidFixed);
	const unsigned long NNIdx = correspondences[tid];
	b[0] = fixed[NNIdx*Dim+0] - CentroidFixed.x;
	b[1] = fixed[NNIdx*Dim+1] - CentroidFixed.y;
	b[2] = fixed[NNIdx*Dim+2] - CentroidFixed.z;

	float M[9] = {0,0,0, 0,0,0, 0,0,0};

	// Accumulate the products a*T(b) into the matrix M
	for (int j = 0; j < 3; ++j) 
	{
		M[j*3+0] += a[j]*b[0];
		M[j*3+1] += a[j]*b[1];
		M[j*3+2] += a[j]*b[2];
	}

	__shared__ float TmpShared[NumPtsPerBlock * 9]; // Shared memory for all 3x3 M matrices of this particular block

	// Linearize matrix (shared memory)
	float* StartIdx = &TmpShared[threadIdx.x * 9];
	float4* StartIdxFloat4 = (float4*)StartIdx;
	float4* MFloat4 = (float4*)M;

	StartIdxFloat4[0] = MFloat4[0];
	StartIdxFloat4[1] = MFloat4[1];
	StartIdx[8] = M[8];

	__syncthreads();


	// Reduce results
	ParallelReductionFloat3<NumPtsPerBlock, 9, 0>(threadIdx.x, TmpShared);
	ParallelReductionFloat3<NumPtsPerBlock, 9, 1>(threadIdx.x, TmpShared);
	ParallelReductionFloat3<NumPtsPerBlock, 9, 2>(threadIdx.x, TmpShared);

	// Store sum of matrices of this block to global memory
	if (NumPts < 3)
	{
		// If we have less than 3 threads left running, let the first thread do all the work
		if (threadIdx.x == 0)
		{
			// Here we use float4, since we might gain additional speedup
			float4* OutFloat4 = (float4*)out;
			float4* TmpSharedFloat4 = (float4*)TmpShared;

			OutFloat4[0] = TmpSharedFloat4[0];
			OutFloat4[1] = TmpSharedFloat4[1];
			out[8] = TmpShared[8];
		}
	}
	else if (threadIdx.x < 3)
	{
		// If there are enough threads available, let threads 0..2 copy data in parallel, 3 values at a time
		float3* OutFloat3 = (float3*)out;
		float3* TmpSharedFloat3 = (float3*)TmpShared;
		OutFloat3[threadIdx.x + blockIdx.x*3] = TmpSharedFloat3[threadIdx.x];
	}
}


// Reduce previously computed 3x3 M matrices
//----------------------------------------------------------------------------
template<unsigned int NumPts>
__global__ void kernelReduceMMatrices(float* matrices, float* out)
{
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if ( tid >= NumPts ) return;

	// Compute number of points per block, check if NumPts is less than blocksize
	const unsigned int NumPtsPerBlock = CUDA_THREADS_PER_BLOCK < NumPts ? CUDA_THREADS_PER_BLOCK : NumPts;

	__shared__ float TmpShared[NumPtsPerBlock * 9]; // For all 3x3 M matrices of this particular block

	// Copy data (float3 version)
	const int posMatrices = tid*3; // 9 divided by 3, since we use float3
	const int posTmp = threadIdx.x*3; // 9 divided by 3, since we use float3

	float3* MatricesFloat3 = (float3*)matrices;
	float3* TmpSharedFloat3 = (float3*)TmpShared;

	TmpSharedFloat3[posTmp+0] = MatricesFloat3[posMatrices+0];
	TmpSharedFloat3[posTmp+1] = MatricesFloat3[posMatrices+1];
	TmpSharedFloat3[posTmp+2] = MatricesFloat3[posMatrices+2];

	__syncthreads();
	
	
	// Reduce
	ParallelReductionFloat3<NumPtsPerBlock, 9, 0>(threadIdx.x, TmpShared);
	ParallelReductionFloat3<NumPtsPerBlock, 9, 1>(threadIdx.x, TmpShared);
	ParallelReductionFloat3<NumPtsPerBlock, 9, 2>(threadIdx.x, TmpShared);

	// Store values to global memory
	if (NumPts < 3)
	{
		// If we have less than 3 threads left running, let the first thread do all the work
		if (threadIdx.x == 0)
		{
			// Here we use float4, since we might gain additional speedup
			float4* OutFloat4 = (float4*)out;
			float4* TmpSharedFloat4 = (float4*)TmpShared;

			OutFloat4[0] = TmpSharedFloat4[0];
			OutFloat4[1] = TmpSharedFloat4[1];
			out[8] = TmpShared[8];
		}
	}
	else if (threadIdx.x < 3)
	{
		// If there are enough threads available, let threads 0..2 copy data in parallel, 3 values at a time
		float3* OutFloat3 = (float3*)out;
		OutFloat3[threadIdx.x + blockIdx.x*3] = TmpSharedFloat3[threadIdx.x];
	}
}


// Transform points w.r.t. a given 4x4 homogeneous transformation matrix m
//----------------------------------------------------------------------------
template<unsigned int NumPts, unsigned int Dim>
__global__ void kernelTransformPoints3D(float* points, float* m)
{
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if ( tid >= NumPts ) return;

	// Pull transformation matrix in parallel
	__shared__ float MatShared[16];

	// Modified index for operating on a point of dimension Dim
	const int pos = tid * Dim;

	// Use float3 to make memory operations faster
	float3* PointsFloat3 = (float3*)(points+pos);
	float4* MatSharedFloat4 = (float4*)MatShared;
	float4* MatFloat4 = (float4*)m;

	if (threadIdx.x < 4)
		MatSharedFloat4[threadIdx.x] = MatFloat4[threadIdx.x];

	__syncthreads();


	if (tid >= NumPts)
		return;

	// Get point data
	const float3 pt = PointsFloat3[0];

	// Compute transformation using the homogeneous transformation matrix m
	const float x = MatShared[0]  * pt.x + MatShared[1]  * pt.y + MatShared[2]  * pt.z + MatShared[3];
	const float y = MatShared[4]  * pt.x + MatShared[5]  * pt.y + MatShared[6]  * pt.z + MatShared[7];
	const float z = MatShared[8]  * pt.x + MatShared[9]  * pt.y + MatShared[10] * pt.z + MatShared[11];
	
	// We do not have to compute the homogeneous coordinate w and divide x, y and z by w,
	// since w will always be 1 in this ICP implementation

	// Update coordinates (one float3 instead of 3 floats sequentially)
	*PointsFloat3 = make_float3(x, y, z);
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

	__syncthreads();


	if (threadIdx.x >= 4) return;

	// Compute matrix-matrix (4x4) multiplication
	float sum;
	#pragma unroll
	for (int j = 0; j < 4; ++j)
	{
		sum = 0;
		#pragma unroll
		for (int k = 0; k < 4; ++k)
		{
			sum += mShared[threadIdx.x*4+k] * accuShared[k*4+j];
		}
		accu[threadIdx.x*4+j] = sum;
	}
}


// Compute Eigenvectors from a given 4x4 matrix
//----------------------------------------------------------------------------
__device__ void kernelJacobi4x4(float *Matrix, float *Eigenvalues, float *Eigenvectors)
{
	/////////////////////////////////////////////////////////////////
	// The following code is a modified version of parts of		   //
	// 'vtkJacobiN' from 'vtkMath' the Visualization Toolkit (VTK, //
	//								          http://www.vtk.org/) //
	/////////////////////////////////////////////////////////////////

#define ROTATE(Matrix, i, j, k, l) g = Matrix[i*4+j]; h = Matrix[k*4+l]; Matrix[i*4+j] = g-s*(h+g*tau); Matrix[k*4+l] = h + s*(g-h*tau);

	float bspace[4];
	float zspace[4] = {0,0,0,0};
	float *b = bspace;
	float *z = zspace;

	const int MaxRotations = 20;

	// Initialize
	for (int ip = 0; ip < 4; ++ip)
	{
		b[ip] = Eigenvalues[ip] = Matrix[ip*4+ip];
	}

	// Begin rotation sequence
	for (int i = 0; i < MaxRotations; ++i)
	{
		float sm = 0.f;
		for (int ip = 0; ip < 4-1; ++ip)
		{
			#pragma unroll
			for (int iq = ip+1; iq < 4; ++iq)
			{
				sm += fabs(Matrix[ip*4+iq]);
			}
		}
		if (sm == 0.f) break;

		float tresh = i < 3 ? 0.2f*sm/(4*4) : 0.f;
		for (int ip = 0; ip < 4-1; ++ip)
		{
			for (int iq = ip+1; iq < 4; ++iq)
			{
				float g = 100.f*fabs(Matrix[ip*4+iq]);

				// After 4 sweeps
				if (i > 3 && (fabs(Eigenvalues[ip]) + g) == fabs(Eigenvalues[ip]) && (fabs(Eigenvalues[iq]) + g) == fabs(Eigenvalues[iq]))
					Matrix[ip*4+iq] = 0.f;
				else if (fabs(Matrix[ip*4+iq]) > tresh)
				{
					float h = Eigenvalues[iq] - Eigenvalues[ip];
					float t;
					if ( (fabs(h)+g) == fabs(h) ) t = Matrix[ip*4+iq] / h;
					else
					{
						float theta = 0.5f*h / Matrix[ip*4+iq];
						t = 1.f / (fabs(theta) + sqrt(1.f+theta*theta));
						if (theta < 0.f) t = -t;
					}
					float c = 1.f / sqrt(1.f + t*t);
					float s = t*c;
					float tau = s/(1.f + c);
					h = t*Matrix[ip*4+iq];
					z[ip] -= h;
					z[iq] += h;
					Eigenvalues[ip] -= h;
					Eigenvalues[iq] += h;
					Matrix[ip*4+iq] = 0.f;

					// ip already shifted left by 1 unit
					#pragma unroll
					for (int j = 0; j <= ip-1; ++j)
					{
						ROTATE(Matrix, j, ip, j, iq);
					}

					// ip and iq already shifted left by 1 unit
					#pragma unroll
					for (int j = ip+1; j <= iq-1; ++j)
					{
						ROTATE(Matrix, ip, j, j, iq);
					}

					// iq already shifted left by 1 unit
					#pragma unroll
					for (int j = iq+1; j < 4; ++j)
					{
						ROTATE(Matrix, ip, j, iq, j);
					}

					#pragma unroll
					for (int j = 0; j < 4; ++j)
					{
						ROTATE(Eigenvectors, j, ip, j, iq);
					}
				}
			}
		}

		#pragma unroll
		for (int ip = 0; ip < 4; ++ip)
		{
			b[ip] += z[ip];
			Eigenvalues[ip] = b[ip];
			z[ip] = 0.f;
		}
	}

	// Sort eigenfunctions
	for (int j = 0; j < 4-1; ++j)
	{
		int k = j;
		float tmp = Eigenvalues[k];
		#pragma unroll
		for (int i = j + 1; i < 4; ++i)
		{
			if (Eigenvalues[i] >= tmp)
			{
				k = i;
				tmp = Eigenvalues[k];
			}
		}

		if (k != j)
		{
			Eigenvalues[k] = Eigenvalues[j];
			Eigenvalues[j] = tmp;
			#pragma unroll
			for (int i = 0; i < 4; ++i)
			{
				tmp = Eigenvectors[i*4+j];
				Eigenvectors[i*4+j] = Eigenvectors[i*4+k];
				Eigenvectors[i*4+k] = tmp;
			}
		}
	}

	// Ensure eigenvector consistency (i.e., Jacobi can compute vectors that
	// are negative of one another (.707,.707,0) and (-.707,-.707,0). This can
	// reek havoc in hyperstreamline/other stuff. We will select the most
	// positive eigenvector.
	int ceil_half_n = (4 >> 1) + (4 & 1);
	#pragma unroll
	for (int j = 0; j < 4; ++j)
	{
		int numPos = 0;
		for (int i = 0; i < 4; ++i)
			if (Eigenvectors[i*4+j] >= 0.f)
				numPos++;

		if (numPos < ceil_half_n)
			for (int i = 0; i < 4; ++i)
				Eigenvectors[i*4+j] *= -1.f;
	}
}


// Compute Eigenvectors from a given 4x4 matrix
//----------------------------------------------------------------------------
__global__ void kernelEstimateTransformationFromMMatrix(float* centroidMoving, float* centroidFixed, float* matrix, float* outMatrix)
{
	// Copy centroids
	const float3 CentroidMoving = *((float3*)centroidMoving);
	const float3 CentroidFixed = *((float3*)centroidFixed);

	// Copy matrix
	float M[3*3];
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			M[r*3+c] = matrix[r*3+c];

	// Build the 4x4 matrix N
	float Ndata[4*4] = {M[0*3+0] + M[1*3+1] + M[2*3+2], M[1*3+2] - M[2*3+1], M[2*3+0] - M[0*3+2], M[0*3+1] - M[1*3+0],
						M[1*3+2] - M[2*3+1], M[0*3+0] - M[1*3+1] - M[2*3+2], M[0*3+1] + M[1*3+0], M[2*3+0] + M[0*3+2],
						M[2*3+0] - M[0*3+2], M[0*3+1] + M[1*3+0], - M[0*3+0] + M[1*3+1] - M[2*3+2], M[1*3+2] + M[2*3+1],
						M[0*3+1] - M[1*3+0], M[2*3+0] + M[0*3+2], M[1*3+2] + M[2*3+1], - M[0*3+0] - M[1*3+1] + M[2*3+2] };

	// Eigen-decompose N (is symmetric)
	float Eigenvectors[4*4] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	float Eigenvalues[4];

	kernelJacobi4x4(Ndata, Eigenvalues, Eigenvectors);

	// The eigenvector with the largest eigenvalue is the quaternion we want
	// (they are sorted in decreasing order for us by JacobiN)
	float w, x, y, z;

	// Points are not collinear
	w = Eigenvectors[0*4+0];
	x = Eigenvectors[1*4+0];
	y = Eigenvectors[2*4+0];
	z = Eigenvectors[3*4+0];

	// Convert quaternion to a rotation matrix
	const float ww = w*w;
	const float wx = w*x;
	const float wy = w*y;
	const float wz = w*z;

	const float xx = x*x;
	const float yy = y*y;
	const float zz = z*z;

	const float xy = x*y;
	const float xz = x*z;
	const float yz = y*z;

	// Create temporary matrix (top 3x4 part of the 4x4 matrix)
	float TmpMat[12];

	// Fill output matrix
	TmpMat[0*4+0] = ww + xx - yy - zz; 
	TmpMat[1*4+0] = 2.f*(wz + xy);
	TmpMat[2*4+0] = 2.f*(-wy + xz);

	TmpMat[0*4+1] = 2.f*(-wz + xy);  
	TmpMat[1*4+1] = ww - xx + yy - zz;
	TmpMat[2*4+1] = 2.f*(wx + yz);

	TmpMat[0*4+2] = 2.f*(wy + xz);
	TmpMat[1*4+2] = 2.f*(-wx + yz);
	TmpMat[2*4+2] = ww - xx - yy + zz;

	// The translation is given by the difference in the transformed moving centroid and the fixed centroid
	const float TransX = TmpMat[0*4+0] * CentroidMoving.x + TmpMat[0*4+1] * CentroidMoving.y + TmpMat[0*4+2] * CentroidMoving.z;
	const float TransY = TmpMat[1*4+0] * CentroidMoving.x + TmpMat[1*4+1] * CentroidMoving.y + TmpMat[1*4+2] * CentroidMoving.z;
	const float TransZ = TmpMat[2*4+0] * CentroidMoving.x + TmpMat[2*4+1] * CentroidMoving.y + TmpMat[2*4+2] * CentroidMoving.z;

	TmpMat[0*4+3] = CentroidFixed.x - TransX;
	TmpMat[1*4+3] = CentroidFixed.y - TransY;
	TmpMat[2*4+3] = CentroidFixed.z - TransZ;

	// Write matrix to global memory using float4 for faster access
	float4* TmpMatFloat4 = (float4*)TmpMat;
	float4* OutMatrixFloat4 = (float4*)outMatrix;

	OutMatrixFloat4[0] = TmpMatFloat4[0]; // First row
	OutMatrixFloat4[1] = TmpMatFloat4[1]; // Second row
	OutMatrixFloat4[2] = TmpMatFloat4[2]; // Third row

	// Fill the bottom row of the 4x4 matrix
	OutMatrixFloat4[3] = make_float4(0.f, 0.f, 0.f, 1.f);  // Fourth row
}


#endif // ICPKERNEL_H__