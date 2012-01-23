#include <algorithm>
#include <iterator>

#include "ICP.h"
#include "defs.h"
#include "ritkDebugManager.h"


//----------------------------------------------------------------------------
extern "C"
void CUDATransformPoints3D(float* points, float* m, unsigned int numPts, unsigned int dim);

extern "C"
void CUDAAccumulateMatrix(float* accu, float* m);

extern "C"
void CUDAComputeCentroid3D(float* points, float* out, unsigned long* correspondences, unsigned int numPts, unsigned int dim);

extern "C"
void CUDABuildMMatrices(float* moving, float* fixed, float* centroidMoving, float* centroidFixed, float* out, unsigned long* correspondences, unsigned int numPts, unsigned int dim);

extern "C"
void CUDAReduceMMatrices(float* matrices, float* out, unsigned int numPts);

extern "C"
void CUDAEstimateTransformationFromMMatrix(float* centroidMoving, float* centroidFixed, float* matrix, float* outMatrix);
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
ICP<NumPts, Dim>::ICP(unsigned long MaxIter) : m_MaxIter(MaxIter), m_NormThreshold(25*FLT_EPSILON)
{
	// Create RBC object
	const unsigned long NumReps = 0; // numReps=0 will result in sqrt(NumPts) automatically during construction of RBC
	m_RBC = std::shared_ptr<RBC<NumPts, Dim> >(new RBC<NumPts, Dim>(NumReps));

	// Init container for set of fixed points
	m_FixedPtsInitialized = false;

	// Init container for set of moving points
	m_MovingPtsInitialized = false;

	// Init container for intermediate transformation matrix
	m_TmpMat = MatrixContainer::New();
	MatrixContainer::SizeType MatSize;
	MatSize.SetElement(0, 4*4);
	m_TmpMat->SetContainerSize(MatSize);
	m_TmpMat->Reserve(MatSize[0]);

	// Init matrix that accumulates intermediate transformation matrices
	m_AccumulateMat = MatrixContainer::New();
	m_AccumulateMat->SetContainerSize(MatSize);
	m_AccumulateMat->Reserve(MatSize[0]);
	const float Id4x4[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	std::copy(Id4x4, Id4x4+16, stdext::checked_array_iterator<float*>(m_AccumulateMat->GetBufferPointer(), 16));
	m_AccumulateMat->SynchronizeDevice();

	// Init container for temporary Cuda memory
	m_TmpCudaMemory = DatasetContainer::New();
	DatasetContainer::SizeType DataSize;
	DataSize.SetElement(0, 1*4096);
	m_TmpCudaMemory->SetContainerSize(DataSize);
	m_TmpCudaMemory->Reserve(DataSize[0]);

	// Init container for transferring centroids and M matrix (3+3+9 = 15 elements)
	m_CentroidsAndMMatrix = DatasetContainer::New();
	DatasetContainer::SizeType AnotherSize;
	AnotherSize.SetElement(0, 1*15);
	m_CentroidsAndMMatrix->SetContainerSize(AnotherSize);
	m_CentroidsAndMMatrix->Reserve(AnotherSize[0]);
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
ICP<NumPts, Dim>::~ICP()
{
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
void ICP<NumPts, Dim>::SetFixedPts(DatasetContainer::Pointer Fixed)
{
	m_FixedPts = Fixed;
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
void ICP<NumPts, Dim>::SetFixedPts(float* Fixed)
{
	// Init container for set of fixed points
	if (!m_FixedPtsInitialized)
	{
		m_FixedPts = DatasetContainer::New();
		DatasetContainer::SizeType DataSize;
		DataSize.SetElement(0, Dim*NumPts);
		m_FixedPts->SetContainerSize(DataSize);
		m_FixedPts->Reserve(DataSize[0]);

		m_FixedPtsInitialized = true;
	}	

	// Copy data
	std::copy(Fixed, Fixed+(NumPts*Dim), stdext::checked_array_iterator<float*>(m_FixedPts->GetBufferPointer(), NumPts*Dim));

	// Synchronize to GPU
	m_FixedPts->SynchronizeDevice();
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
void ICP<NumPts, Dim>::SetMovingPts(DatasetContainer::Pointer Moving)
{
	m_MovingPts = Moving;
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
void ICP<NumPts, Dim>::SetMovingPts(float* Moving)
{
	// Init container for set of moving points
	if (!m_MovingPtsInitialized)
	{
		m_MovingPts = DatasetContainer::New();
		DatasetContainer::SizeType DataSize;
		DataSize.SetElement(0, Dim*NumPts);
		m_MovingPts->SetContainerSize(DataSize);
		m_MovingPts->Reserve(DataSize[0]);

		m_MovingPtsInitialized = true;
	}

	// Copy data
	std::copy(Moving, Moving+(NumPts*Dim), stdext::checked_array_iterator<float*>(m_MovingPts->GetBufferPointer(), NumPts*Dim));

	// Synchronize to GPU
	m_MovingPts->SynchronizeDevice();
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
void ICP<NumPts, Dim>::SetPayloadWeight(float Weight)
{
	// You should only provide values between [0;1]
	if (Weight < 0 || Weight > 1)
	{
		LOG_DEB("Warning: You should only provide values in [0;1].");
	}

	float Weights[Dim];
	float SpatialWeight = (1.f-Weight)/3.f;
	float PayloadWeight = Weight/static_cast<float>(Dim-3U);
	std::fill(Weights, Weights+3, SpatialWeight);
	std::fill(Weights+3, Weights+Dim, PayloadWeight);

	m_RBC->SetWeights(Weights);	
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
void ICP<NumPts, Dim>::Run(unsigned long* NumIter, float* FinalNorm)
{
	// Initialize the ICP (build RBC tree, etc.)
	Initialize();

	// Iterate
	while (NextIteration()); // Run next iteration

	// Fill output parameters if appropriate
	if (NumIter) *NumIter = m_Iter;
	if (FinalNorm) *FinalNorm = m_FrobNorm;
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
void ICP<NumPts, Dim>::Initialize()
{
	// Build RBC tree
	m_RBC->BuildRBC(m_FixedPts);

	// Fill elements of accumulation matrix so that we get the identity matrix (4x4) and transfer to GPU
	//const float Id4x4[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	//std::copy(Id4x4, Id4x4+16, stdext::checked_array_iterator<float*>(m_AccumulateMat->GetBufferPointer(), 16));
	//m_AccumulateMat->SynchronizeDevice();

	// Apply previous transform
	// TODO UNDO!!!
	CUDATransformPoints3D(
		m_MovingPts->GetCudaMemoryPointer(),
		m_AccumulateMat->GetCudaMemoryPointer(),
		NumPts,
		Dim
		);

	// We use the Frobenius norm of the intermediate transformation matrix as an indicator for convergence
	m_FrobNorm = FLT_MAX;

	// Iterate until convergence or reaching maximum number of iterations
	m_Iter = 0;
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
bool ICP<NumPts, Dim>::NextIteration()
{
	// Step 1: Compute NN correspondences (using RBC), do *not* copy correspondences to host (faster!)
	m_RBC->Query(m_MovingPts, false);

	// Step 2: Estimate transformation matrix using NN correspondences (this is done internally)
	EstimateTransformationMatrix(m_MovingPts->GetBufferPointer(), m_FixedPts->GetBufferPointer());

	// Step 3: Transform points w.r.t. estimated intermediate transformation
	CUDATransformPoints3D(
		m_MovingPts->GetCudaMemoryPointer(),
		m_TmpMat->GetCudaMemoryPointer(),
		NumPts,
		Dim
		);

	// Accumulate intermediate matrices
	CUDAAccumulateMatrix(
		m_AccumulateMat->GetCudaMemoryPointer(),
		m_TmpMat->GetCudaMemoryPointer()
		);

	// Increment iteration count
	++m_Iter;

	// Check for convergence each 10'th iteration only (due to runtime concerns)
	if (m_Iter % 10 == 0)
	{
		// Get matrix from GPU (memory transfer is most expensive part)
		m_TmpMat->SynchronizeHost();

		// Compute Frobenius norm of (estimated matrix minus Id_4x4)
		m_FrobNorm = 0;
		float* TmpMatPtr = m_TmpMat->GetBufferPointer();
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				float El = TmpMatPtr[i*4+j];
				if (i == j) El = El-1; // Subtract 4x4 identity matrix
				m_FrobNorm += El*El;
			}
		}
	}	

	// Check for convergence
	bool AnotherIterationRequired = m_Iter < m_MaxIter && m_FrobNorm > m_NormThreshold;

	if (!AnotherIterationRequired)
	{
		// Copy accumulated matrix (ICP output) from GPU to CPU
		m_AccumulateMat->SynchronizeHost();

		// Copy points to host
		m_MovingPts->SynchronizeHost();
	}

	return AnotherIterationRequired;
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
void ICP<NumPts, Dim>::Jacobi4x4(float *Matrix, float *Eigenvalues, float *Eigenvectors)
{
	/////////////////////////////////////////////////////////////////
	// The following code is a modified version of parts of		   //
	// 'vtkJacobiN' from 'vtkMath' the Visualization Toolkit (VTK, //
	//								          http://www.vtk.org/) //
	/////////////////////////////////////////////////////////////////

#define ROTATE(Matrix, i, j, k, l) g = Matrix[i*n+j]; h = Matrix[k*n+l]; Matrix[i*n+j] = g-s*(h+g*tau); Matrix[k*n+l] = h + s*(g-h*tau);

	float bspace[4], zspace[4];
	float *b = bspace;
	float *z = zspace;

	// We have a 4x4 matrix
	const int n = 4;
	const int MaxRotations = 20;

	// Initialize
	for (int ip = 0; ip < n; ++ip)
	{
		b[ip] = Eigenvalues[ip] = Matrix[ip*n+ip];
		z[ip] = 0.f;
	}

	// Begin rotation sequence
	for (int i = 0; i < MaxRotations; ++i)
	{
		float sm = 0.f;
		for (int ip = 0; ip < n-1; ++ip)
		{
			for (int iq = ip+1; iq < n; ++iq)
			{
				sm += fabs(Matrix[ip*n+iq]);
			}
		}
		if (sm == 0.f) break;

		float tresh = i < 3 ? 0.2f*sm/(n*n) : 0.f;
		for (int ip = 0; ip < n-1; ++ip)
		{
			for (int iq = ip+1; iq < n; ++iq)
			{
				float g = 100.f*fabs(Matrix[ip*n+iq]);

				// After 4 sweeps
				if (i > 3 && (fabs(Eigenvalues[ip]) + g) == fabs(Eigenvalues[ip]) && (fabs(Eigenvalues[iq]) + g) == fabs(Eigenvalues[iq]))
					Matrix[ip*n+iq] = 0.f;
				else if (fabs(Matrix[ip*n+iq]) > tresh)
				{
					float h = Eigenvalues[iq] - Eigenvalues[ip];
					float t;
					if ( (fabs(h)+g) == fabs(h) ) t = Matrix[ip*n+iq] / h;
					else
					{
						float theta = 0.5f*h / Matrix[ip*n+iq];
						t = 1.f / (fabs(theta) + sqrt(1.f+theta*theta));
						if (theta < 0.f) t = -t;
					}
					float c = 1.f / sqrt(1.f + t*t);
					float s = t*c;
					float tau = s/(1.f + c);
					h = t*Matrix[ip*n+iq];
					z[ip] -= h;
					z[iq] += h;
					Eigenvalues[ip] -= h;
					Eigenvalues[iq] += h;
					Matrix[ip*n+iq] = 0.f;

					// ip already shifted left by 1 unit
					for (int j = 0; j <= ip-1; ++j)
					{
						ROTATE(Matrix, j, ip, j, iq);
					}

					// ip and iq already shifted left by 1 unit
					for (int j = ip+1; j <= iq-1; ++j)
					{
						ROTATE(Matrix, ip, j, j, iq);
					}

					// iq already shifted left by 1 unit
					for (int j = iq+1; j < n; ++j)
					{
						ROTATE(Matrix, ip, j, iq, j);
					}

					for (int j = 0; j < n; ++j)
					{
						ROTATE(Eigenvectors, j, ip, j, iq);
					}
				}
			}
		}

		for (int ip = 0; ip < n; ++ip)
		{
			b[ip] += z[ip];
			Eigenvalues[ip] = b[ip];
			z[ip] = 0.f;
		}
	}

	// Sort eigenfunctions
	for (int j = 0; j < n-1; ++j)
	{
		int k = j;
		float tmp = Eigenvalues[k];
		for (int i = j + 1; i < n; ++i)
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
			for (int i = 0; i < n; ++i)
			{
				tmp = Eigenvectors[i*n+j];
				Eigenvectors[i*n+j] = Eigenvectors[i*n+k];
				Eigenvectors[i*n+k] = tmp;
			}
		}
	}

	// Ensure eigenvector consistency (i.e., Jacobi can compute vectors that
	// are negative of one another (.707,.707,0) and (-.707,-.707,0). This can
	// reek havoc in hyperstreamline/other stuff. We will select the most
	// positive eigenvector.
	int ceil_half_n = (n >> 1) + (n & 1);
	for (int j = 0; j < n; ++j)
	{
		int numPos = 0;
		for (int i = 0; i < n; ++i)
			if (Eigenvectors[i*n+j] >= 0.f)
				numPos++;

		if (numPos < ceil_half_n)
			for (int i = 0; i < n; ++i)
				Eigenvectors[i*n+j] *= -1.f;
	}
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
void ICP<NumPts, Dim>::EstimateTransformationMatrix(DatasetContainer::Element* Moving, DatasetContainer::Element* Fixed)
{
	/////////////////////////////////////////////////////////////////
	// The following code is a modified version of parts of		   //
	// 'vtkLandmarkTransform' from the Visualization Toolkit (VTK, //
	//										  http://www.vtk.org/) //
	/////////////////////////////////////////////////////////////////

	// Compute Centroid for set of moving points
	CUDAComputeCentroid3D(
		m_MovingPts->GetCudaMemoryPointer(),
		m_TmpCudaMemory->GetCudaMemoryPointer(), // Write to position 0,1,2
		NULL,
		NumPts/2,
		Dim
		);	

	// Finalize reduction
	CUDAComputeCentroid3D(
		m_TmpCudaMemory->GetCudaMemoryPointer(), // Read from position 0,1,2
		m_CentroidsAndMMatrix->GetCudaMemoryPointer(), // Write to position 0,1,2
		NULL,
		((NumPts/2)/CUDA_THREADS_PER_BLOCK)/2,
		3
		);

	// Compute Centroid for set of fixed points
	CUDAComputeCentroid3D(
		m_FixedPts->GetCudaMemoryPointer(),
		m_TmpCudaMemory->GetCudaMemoryPointer(),
		m_RBC->GetCorrespondencesContainer()->GetCudaMemoryPointer(),
		NumPts/2,
		Dim
		);	

	// Finalize reduction
	CUDAComputeCentroid3D(
		m_TmpCudaMemory->GetCudaMemoryPointer(),
		m_CentroidsAndMMatrix->GetCudaMemoryPointer()+3, // Write to positon 3,4,5
		NULL,
		((NumPts/2)/CUDA_THREADS_PER_BLOCK)/2,
		3
		);

	// Build 3x3 matrices M and do first reduction step
	CUDABuildMMatrices(
		m_MovingPts->GetCudaMemoryPointer(),
		m_FixedPts->GetCudaMemoryPointer(),
		m_CentroidsAndMMatrix->GetCudaMemoryPointer(), // Centroid of moving point set
		m_CentroidsAndMMatrix->GetCudaMemoryPointer()+3, // Centroid of fixed point set
		m_TmpCudaMemory->GetCudaMemoryPointer(), // Write M matrices here
		m_RBC->GetCorrespondencesContainer()->GetCudaMemoryPointer(),
		NumPts,
		Dim
		);

	// Finalize reduction of these matrices
	CUDAReduceMMatrices(
		m_TmpCudaMemory->GetCudaMemoryPointer(),
		m_CentroidsAndMMatrix->GetCudaMemoryPointer()+6, // Start writing after both centroids (3+3 = 6)
		NumPts/CUDA_THREADS_PER_BLOCK
		);

#ifdef ESTIMATE_MATRIX_ON_GPU
	// Estimate transformation matrix
	CUDAEstimateTransformationFromMMatrix(
		m_CentroidsAndMMatrix->GetCudaMemoryPointer(), // Centroid of moving point set
		m_CentroidsAndMMatrix->GetCudaMemoryPointer()+3, // Centroid of fixed point set
		m_CentroidsAndMMatrix->GetCudaMemoryPointer()+6, // M matrix
		m_TmpMat->GetCudaMemoryPointer() // Final transformation matrix (output)
		);

#else
	m_CentroidsAndMMatrix->SynchronizeHost();

	// Copy centroids
	const float xMoving = m_CentroidsAndMMatrix->GetBufferPointer()[0];
	const float yMoving = m_CentroidsAndMMatrix->GetBufferPointer()[1];
	const float zMoving = m_CentroidsAndMMatrix->GetBufferPointer()[2];
	const float xFixed = m_CentroidsAndMMatrix->GetBufferPointer()[3];
	const float yFixed = m_CentroidsAndMMatrix->GetBufferPointer()[4];
	const float zFixed = m_CentroidsAndMMatrix->GetBufferPointer()[5];

	const float CentroidMoving[3] = {xMoving, yMoving, zMoving};
	const float CentroidFixed[3] = {xFixed, yFixed, zFixed};

	// Copy matrix
	float M[3*3];
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			M[r*3+c] = m_CentroidsAndMMatrix->GetBufferPointer()[r*3+c + 6]; // First element of matrix is at position 6

	// Build the 4x4 matrix N
	float Ndata[4*4] = {M[0*3+0] + M[1*3+1] + M[2*3+2], M[1*3+2] - M[2*3+1], M[2*3+0] - M[0*3+2], M[0*3+1] - M[1*3+0],
						M[1*3+2] - M[2*3+1], M[0*3+0] - M[1*3+1] - M[2*3+2], M[0*3+1] + M[1*3+0], M[2*3+0] + M[0*3+2],
						M[2*3+0] - M[0*3+2], M[0*3+1] + M[1*3+0], - M[0*3+0] + M[1*3+1] - M[2*3+2], M[1*3+2] + M[2*3+1],
						M[0*3+1] - M[1*3+0], M[2*3+0] + M[0*3+2], M[1*3+2] + M[2*3+1], - M[0*3+0] - M[1*3+1] + M[2*3+2] };

	// Eigen-decompose N (is symmetric)
	float Eigenvectors[4*4] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	float Eigenvalues[4];

	Jacobi4x4(Ndata, Eigenvalues, Eigenvectors);

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

	// Pointer to our matrix
	float* TmpMatPtr = m_TmpMat->GetBufferPointer();

	TmpMatPtr[0*4+0] = ww + xx - yy - zz; 
	TmpMatPtr[1*4+0] = 2.0*(wz + xy);
	TmpMatPtr[2*4+0] = 2.0*(-wy + xz);

	TmpMatPtr[0*4+1] = 2.0*(-wz + xy);  
	TmpMatPtr[1*4+1] = ww - xx + yy - zz;
	TmpMatPtr[2*4+1] = 2.0*(wx + yz);

	TmpMatPtr[0*4+2] = 2.0*(wy + xz);
	TmpMatPtr[1*4+2] = 2.0*(-wx + yz);
	TmpMatPtr[2*4+2] = ww - xx - yy + zz;

	// The translation is given by the difference in the transformed moving centroid and the fixed centroid
	float TransX, TransY, TransZ;
	TransX = TmpMatPtr[0*4+0] * CentroidMoving[0] + TmpMatPtr[0*4+1] * CentroidMoving[1] + TmpMatPtr[0*4+2] * CentroidMoving[2];
	TransY = TmpMatPtr[1*4+0] * CentroidMoving[0] + TmpMatPtr[1*4+1] * CentroidMoving[1] + TmpMatPtr[1*4+2] * CentroidMoving[2];
	TransZ = TmpMatPtr[2*4+0] * CentroidMoving[0] + TmpMatPtr[2*4+1] * CentroidMoving[1] + TmpMatPtr[2*4+2] * CentroidMoving[2];

	TmpMatPtr[0*4+3] = CentroidFixed[0] - TransX;
	TmpMatPtr[1*4+3] = CentroidFixed[1] - TransY;
	TmpMatPtr[2*4+3] = CentroidFixed[2] - TransZ;

	// Fill the bottom row of the 4x4 matrix
	TmpMatPtr[3*4+0] = 0.0;
	TmpMatPtr[3*4+1] = 0.0;
	TmpMatPtr[3*4+2] = 0.0;
	TmpMatPtr[3*4+3] = 1.0;

	// Synchronize to device
	m_TmpMat->SynchronizeDevice();
#endif
}
