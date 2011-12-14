#include <algorithm>
#include <iterator>

#include "ICP.h"
#include "defs.h"

#include "vtkMath.h"
#include "vtkBoxMuellerRandomSequence.h"
#include "vtkMinimalStandardRandomSequence.h"
#include "vtkTransform.h"


//----------------------------------------------------------------------------
extern "C"
void CUDATransformPoints3D(float* points, float* m, unsigned int numPts, unsigned int dim);

extern "C"
void CUDAAccumulateMatrix(float* accu, float* m);

extern "C"
void CUDAComputeCentroid(float* points, float* out, unsigned long* correspondences, unsigned int numPts, unsigned int dim);

extern "C"
void CUDABuildMMatrices(float* moving, float* fixed, float* centroidMoving, float* centroidFixed, float* out, unsigned long* correspondences, unsigned int numPts, unsigned int dim);

extern "C"
void CUDAReduceMMatrices(float* matrices, float* out, unsigned int numPts);
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
ICP<NumPts, Dim>::ICP(unsigned long MaxIter) : m_MaxIter(MaxIter), m_NormThreshold(25*FLT_EPSILON)
{
	// Create RBC object
	const unsigned long NumReps = 0; // numReps=0 will result in sqrt(NumPts) automatically during construction of RBC
	m_RBC = new RBC<NumPts, Dim>(NumReps);

	// Init container for set of fixed points
	m_FixedPtsInitialized = false;

	// Init container for set of moving points
	m_MovingPtsInitialized = false;

	// Init container for intermediate transformation matrix
	m_TmpMat = MatrixContainer::New();
	m_TmpMat->SetContainerSize(4, 4);
	m_TmpMat->Reserve(16);

	// Init matrix that accumulates intermediate transformation matrices
	m_AccumulateMat = MatrixContainer::New();
	m_AccumulateMat->SetContainerSize(4, 4);
	m_AccumulateMat->Reserve(16);

	// Init container for temporary Cuda memory
	m_TmpCudaMemory = DatasetContainer::New();
	m_TmpCudaMemory->SetContainerSize(1, 4096);
	m_TmpCudaMemory->Reserve(1*4096);

	// Init container for transferring centroids and M matrix (3+3+9 = 15 elements)
	m_CentroidsAndMMatrix = DatasetContainer::New();
	m_CentroidsAndMMatrix->SetContainerSize(1, 15);
	m_CentroidsAndMMatrix->Reserve(1*15);
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
ICP<NumPts, Dim>::~ICP()
{
	// Delete RBC object
	delete m_RBC;
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
		m_FixedPts->SetContainerSize(Dim, NumPts);
		m_FixedPts->Reserve(Dim*NumPts);

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
		m_MovingPts->SetContainerSize(Dim, NumPts);
		m_MovingPts->Reserve(Dim*NumPts);

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
	//assert(Weight >= 0 && Weight <= 1);

	float Weights[Dim];
	float SpatialWeight = (1.f-Weight)/3.f;
	float PayloadWeight = Weight/static_cast<float>(Dim-3);
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
	float Id4x4[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	std::copy(Id4x4, Id4x4+16, stdext::checked_array_iterator<float*>(m_AccumulateMat->GetBufferPointer(), 16));
	m_AccumulateMat->SynchronizeDevice();

	// We use the Frobenius norm of the intermediate transformation matrix as an indicator for convergence
	m_FrobNorm = FLT_MAX;

	// Iterate until convergence or reaching maximum number of iterations
	m_Iter = 0;
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
bool ICP<NumPts, Dim>::NextIteration()
{
	// Step 1: Compute NN correspondences (using RBC)
	unsigned long* Correspondences;
	Correspondences = m_RBC->Query(m_MovingPts);

	// Step 2: Estimate transformation matrix using NN correspondences
	EstimateTransformationMatrix(m_MovingPts->GetBufferPointer(), m_FixedPts->GetBufferPointer(), Correspondences);

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

	// Compute Frobenius norm of (estimated matrix - 4x4 identity matrix)
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

	// Increment iteration count
	++m_Iter;

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
void ICP<NumPts, Dim>::EstimateTransformationMatrix(DatasetContainer::Element* Moving, DatasetContainer::Element* Fixed, unsigned long* Correspondences)
{
	/////////////////////////////////////////////////////////////////
	// The following code is a modified version of parts of		   //
	// 'vtkLandmarkTransform' from the Visualization Toolkit (VTK, //
	//								   http://http://www.vtk.org/) //
	/////////////////////////////////////////////////////////////////

	// Compute Centroid for set of moving points
	CUDAComputeCentroid(
		m_MovingPts->GetCudaMemoryPointer(),
		m_TmpCudaMemory->GetCudaMemoryPointer(), // Write to position 0,1,2
		NULL,
		NumPts,
		Dim
		);	

	CUDAComputeCentroid(
		m_TmpCudaMemory->GetCudaMemoryPointer(), // Read from position 0,1,2
		m_CentroidsAndMMatrix->GetCudaMemoryPointer(), // Write to position 0,1,2
		NULL,
		NumPts/CUDA_THREADS_PER_BLOCK,
		3
		);

	// Compute Centroid for set of fixed points
	CUDAComputeCentroid(
		m_FixedPts->GetCudaMemoryPointer(),
		m_TmpCudaMemory->GetCudaMemoryPointer(),
		m_RBC->GetCorrespondencesContainer()->GetCudaMemoryPointer(),
		NumPts,
		Dim
		);	

	CUDAComputeCentroid(
		m_TmpCudaMemory->GetCudaMemoryPointer(),
		m_CentroidsAndMMatrix->GetCudaMemoryPointer()+3, // Write to positon 3,4,5
		NULL,
		NumPts/CUDA_THREADS_PER_BLOCK,
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

	m_CentroidsAndMMatrix->SynchronizeHost();

	// Copy centroids
	float xMoving = m_CentroidsAndMMatrix->GetBufferPointer()[0];
	float yMoving = m_CentroidsAndMMatrix->GetBufferPointer()[1];
	float zMoving = m_CentroidsAndMMatrix->GetBufferPointer()[2];
	float xFixed = m_CentroidsAndMMatrix->GetBufferPointer()[3];
	float yFixed = m_CentroidsAndMMatrix->GetBufferPointer()[4];
	float zFixed = m_CentroidsAndMMatrix->GetBufferPointer()[5];

	double CentroidMoving[3] = {xMoving, yMoving, zMoving};
	double CentroidFixed[3] = {xFixed, yFixed, zFixed};

	// Copy matrix
	double M[3][3];
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			M[r][c] = m_CentroidsAndMMatrix->GetBufferPointer()[r*3+c + 6]; // First element of matrix is at position 6

	// Build the 4x4 matrix N
	double Ndata[4][4] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
	double *NPtr[4];
	for (int i = 0; i < 4; ++i)
		NPtr[i] = Ndata[i];
	
	// On-diagonal elements
	NPtr[0][0] =   M[0][0] + M[1][1] + M[2][2];
	NPtr[1][1] =   M[0][0] - M[1][1] - M[2][2];
	NPtr[2][2] = - M[0][0] + M[1][1] - M[2][2];
	NPtr[3][3] = - M[0][0] - M[1][1] + M[2][2];

	// Off-diagonal elements
	NPtr[0][1] = NPtr[1][0] = M[1][2] - M[2][1];
	NPtr[0][2] = NPtr[2][0] = M[2][0] - M[0][2];
	NPtr[0][3] = NPtr[3][0] = M[0][1] - M[1][0];

	NPtr[1][2] = NPtr[2][1] = M[0][1] + M[1][0];
	NPtr[1][3] = NPtr[3][1] = M[2][0] + M[0][2];
	NPtr[2][3] = NPtr[3][2] = M[1][2] + M[2][1];

	// Eigen-decompose N (is symmetric)
	double EigenvectorData[4][4];
	double *Eigenvectors[4], Eigenvalues[4];

	Eigenvectors[0] = EigenvectorData[0];
	Eigenvectors[1] = EigenvectorData[1];
	Eigenvectors[2] = EigenvectorData[2];
	Eigenvectors[3] = EigenvectorData[3];

	vtkMath::JacobiN(NPtr, 4, Eigenvalues, Eigenvectors);

	// The eigenvector with the largest eigenvalue is the quaternion we want
	// (they are sorted in decreasing order for us by JacobiN)
	double w, x, y, z;

	// Points are not collinear
	w = Eigenvectors[0][0];
	x = Eigenvectors[1][0];
	y = Eigenvectors[2][0];
	z = Eigenvectors[3][0];

	// Convert quaternion to a rotation matrix
	double ww = w*w;
	double wx = w*x;
	double wy = w*y;
	double wz = w*z;

	double xx = x*x;
	double yy = y*y;
	double zz = z*z;

	double xy = x*y;
	double xz = x*z;
	double yz = y*z;

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
	double TransX, TransY, TransZ;
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
}
