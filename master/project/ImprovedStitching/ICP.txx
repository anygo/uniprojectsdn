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
	m_RBC = new RBC<NumPts, Dim>(NumReps);

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
		LOG_DEB("Warning: You should only provide values in [0;1]. Not " << Weight << ".");
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

	// Apply previous transform
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

	// Estimate transformation matrix
	CUDAEstimateTransformationFromMMatrix(
		m_CentroidsAndMMatrix->GetCudaMemoryPointer(), // Centroid of moving point set
		m_CentroidsAndMMatrix->GetCudaMemoryPointer()+3, // Centroid of fixed point set
		m_CentroidsAndMMatrix->GetCudaMemoryPointer()+6, // M matrix
		m_TmpMat->GetCudaMemoryPointer() // Final transformation matrix (output)
		);
}
