#ifndef ICPKERNEL_H__
#define ICPKERNEL_H__

#include <cuda_runtime.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "defs.h"


#ifdef USE_TEXTURE_MEMORY
// Texture that holds the set of fixed points
texture<float4, cudaTextureType1D, cudaReadModeElementType> PtsTexture;
#endif


// Parallel reduction
//----------------------------------------------------------------------------
template<unsigned int NumPts, unsigned int Dim, unsigned int Offset>
__device__ void ParallelReduction(int tid, float* SM)
{
	// Compute index position for current thread
	int pos = tid*Dim;

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


#ifdef USE_TEXTURE_MEMORY
// Compute x, y and z coords from (misused) float4 texture
//----------------------------------------------------------------------------
template<unsigned int Dim>
__device__ void getCoordsFromFloat4Tex(float& x, float& y, float& z, int posPoints);

template<>
__device__ void getCoordsFromFloat4Tex<3>(float& x, float& y, float& z, int posPoints)
{
	const int posPointsMod4 = posPoints % 4;
	const int posPointsDiv4Times3 = (posPoints/4)*3;
	float4 tmp;

	switch (posPointsMod4)
	{
	case 0: tmp = tex1Dfetch(PtsTexture, posPointsDiv4Times3);
		x = tmp.x;
		y = tmp.y;
		z = tmp.z;
		break;
	case 1: tmp = tex1Dfetch(PtsTexture, posPointsDiv4Times3);
		x = tmp.w;		
		tmp = tex1Dfetch(PtsTexture, posPointsDiv4Times3+1);
		y = tmp.x;
		z = tmp.y;
		break;
	case 2: tmp = tex1Dfetch(PtsTexture, posPointsDiv4Times3+1);
		x = tmp.z;
		y = tmp.w;
		tmp = tex1Dfetch(PtsTexture, posPointsDiv4Times3+2);
		z = tmp.x;
		break;
	case 3: tmp = tex1Dfetch(PtsTexture, posPointsDiv4Times3+2);
		x = tmp.y;
		y = tmp.z;
		z = tmp.w;
		break;
	}
}

template<>
__device__ void getCoordsFromFloat4Tex<6>(float& x, float& y, float& z, int posPoints)
{
	const int posPointsMod2 = posPoints % 2;
	const int posPointsDiv4Times6 = (posPoints/4)*6;
	float4 tmp;

	switch (posPointsMod2)
	{
	case 0: tmp = tex1Dfetch(PtsTexture, posPointsDiv4Times6);
		x = tmp.x;
		y = tmp.y;
		z = tmp.z;
		break;
	case 1: tmp = tex1Dfetch(PtsTexture, posPointsDiv4Times6+1);
		x = tmp.z;
		y = tmp.w;
		tmp = tex1Dfetch(PtsTexture, posPointsDiv4Times6+2);
		z = tmp.x;
		break;
	}
}
#endif


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
#ifdef USE_TEXTURE_MEMORY
	float x1,y1,z1, x2,y2,z2;
	getCoordsFromFloat4Tex<Dim>(x1, y1, z1, posPoints);
	getCoordsFromFloat4Tex<Dim>(x2, y2, z2, posPoints2);

	TmpShared[posTmp+0] = x1 + x2;
	TmpShared[posTmp+1] = y1 + y2;
	TmpShared[posTmp+2] = z1 + z2;
	/*TmpShared[posTmp+0] = tex1Dfetch(PtsTexture, posPoints+0) + tex1Dfetch(PtsTexture, posPoints2+0);
	TmpShared[posTmp+1] = tex1Dfetch(PtsTexture, posPoints+1) + tex1Dfetch(PtsTexture, posPoints2+1);
	TmpShared[posTmp+2] = tex1Dfetch(PtsTexture, posPoints+2) + tex1Dfetch(PtsTexture, posPoints2+2);*/
#else
	float3* PointsFloat3 = (float3*)points;
	float3* TmpSharedFloat3 = (float3*)TmpShared;
	float3 First = PointsFloat3[Dim == 6 ? posPoints*2 : posPoints]; // If we still are in 6D, we need to multiply the position by 2, since we assume float3
	float3 Second = PointsFloat3[Dim == 6 ? posPoints2*2 : posPoints2];
	TmpSharedFloat3[threadIdx.x] = make_float3(First.x + Second.x, First.y + Second.y, First.z + Second.z);

	/*posPoints *= Dim;
	posPoints2 *= Dim;
	TmpShared[posTmp+0] = points[posPoints+0] + points[posPoints2+0];
	TmpShared[posTmp+1] = points[posPoints+1] + points[posPoints2+1];
	TmpShared[posTmp+2] = points[posPoints+2] + points[posPoints2+2];*/
#endif

	__syncthreads();
	
	
	// Perform parallel reduction for x, y and z coordinate	
	ParallelReduction<NumPtsPerBlock, 3, 0>(threadIdx.x, TmpShared); // X
	ParallelReduction<NumPtsPerBlock, 3, 1>(threadIdx.x, TmpShared); // Y
	ParallelReduction<NumPtsPerBlock, 3, 2>(threadIdx.x, TmpShared); // Z

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
	a[0] = moving[tid*Dim+0] - centroidMoving[0];
	a[1] = moving[tid*Dim+1] - centroidMoving[1];
	a[2] = moving[tid*Dim+2] - centroidMoving[2];

	float b[3];
	const unsigned long NNIdx = correspondences[tid];
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
			out[9] = TmpShared[9];
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

	// Compute indices in matrices array and shared memory
	//const int posMatrices = tid*9;
	//const int posTmp = threadIdx.x*9;

	// Copy data
	//TmpShared[posTmp+0] = matrices[posMatrices+0];
	//TmpShared[posTmp+1] = matrices[posMatrices+1];
	//TmpShared[posTmp+2] = matrices[posMatrices+2];
	//TmpShared[posTmp+3] = matrices[posMatrices+3];
	//TmpShared[posTmp+4] = matrices[posMatrices+4];
	//TmpShared[posTmp+5] = matrices[posMatrices+5];
	//TmpShared[posTmp+6] = matrices[posMatrices+6];
	//TmpShared[posTmp+7] = matrices[posMatrices+7];
	//TmpShared[posTmp+8] = matrices[posMatrices+8];

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
			out[9] = TmpShared[9];
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
	PointsFloat3[0] = make_float3(x, y, z);
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
	const float CentroidMoving[3] = {centroidMoving[0], centroidMoving[1], centroidMoving[2]};
	const float CentroidFixed[3] = {centroidFixed[0], centroidFixed[1], centroidFixed[2]};

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

	// Fill output matrix
	outMatrix[0*4+0] = ww + xx - yy - zz; 
	outMatrix[1*4+0] = 2.f*(wz + xy);
	outMatrix[2*4+0] = 2.f*(-wy + xz);

	outMatrix[0*4+1] = 2.f*(-wz + xy);  
	outMatrix[1*4+1] = ww - xx + yy - zz;
	outMatrix[2*4+1] = 2.f*(wx + yz);

	outMatrix[0*4+2] = 2.f*(wy + xz);
	outMatrix[1*4+2] = 2.f*(-wx + yz);
	outMatrix[2*4+2] = ww - xx - yy + zz;

	// The translation is given by the difference in the transformed moving centroid and the fixed centroid
	float TransX, TransY, TransZ;
	TransX = outMatrix[0*4+0] * CentroidMoving[0] + outMatrix[0*4+1] * CentroidMoving[1] + outMatrix[0*4+2] * CentroidMoving[2];
	TransY = outMatrix[1*4+0] * CentroidMoving[0] + outMatrix[1*4+1] * CentroidMoving[1] + outMatrix[1*4+2] * CentroidMoving[2];
	TransZ = outMatrix[2*4+0] * CentroidMoving[0] + outMatrix[2*4+1] * CentroidMoving[1] + outMatrix[2*4+2] * CentroidMoving[2];

	outMatrix[0*4+3] = CentroidFixed[0] - TransX;
	outMatrix[1*4+3] = CentroidFixed[1] - TransY;
	outMatrix[2*4+3] = CentroidFixed[2] - TransZ;

	// Fill the bottom row of the 4x4 matrix
	outMatrix[3*4+0] = 0.f;
	outMatrix[3*4+1] = 0.f;
	outMatrix[3*4+2] = 0.f;
	outMatrix[3*4+3] = 1.f;
}


#endif // ICPKERNEL_H__