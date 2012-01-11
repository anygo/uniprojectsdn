#ifndef ICP_H__
#define ICP_H__

#include "RBC.h"

// If ESTIMATE_MATRIX_ON_GPU is defined, everything will be executed on the GPU, no memory transfers during ICP necessary
#define ESTIMATE_MATRIX_ON_GPU


/**	@class		ICP
 *	@brief		Implementation of the ICP algorithm
 *	@author		Dominik Neumann
 *
 *	@details
 *	Implementation of the iterative closest point (ICP) algorithm, partially based on VTK ICP implementation.
 *	Registration of points is always performed in 3-D. If Dim > 3, additional dimensions will be used during NN search only.
 */
template<unsigned long NumPts, unsigned long Dim>
class ICP
{
	/**	@name CUDA memory containers */
	//@{
	typedef ritk::CudaRegularMemoryImportImageContainerF DatasetContainer;
	typedef ritk::CudaRegularMemoryImportImageContainerF MatrixContainer;
	//@}

public:
	/// Constructor 
	ICP(unsigned long MaxIter = 250);

	/// Destructor
	~ICP();

	/// Set fixed points
	void SetFixedPts(DatasetContainer::Pointer Fixed);

	/// Import fixed points from 1-D linearized float array of size NumPts*Dim
	void SetFixedPts(float* Fixed);

	/// Set moving points
	void SetMovingPts(DatasetContainer::Pointer Moving);

	/// Import moving points from 1-D linearized float array of size NumPts*Dim
	void SetMovingPts(float* Moving);

	/// Returns pointer to the dataset container holding the fixed points (e.g. if you want to use the point on the GPU later)
	inline DatasetContainer::Pointer GetFixedPtsContainer() const { return m_FixedPts; }

	/// Returns pointer to the set of fixed points as a 1-D linearized float array of size NumPts*Dim
	inline float* GetFixedPts() const { return m_FixedPts->GetBufferPointer(); }

	/// Returns pointer to the dataset container holding the moving points (e.g. if you want to use the point on the GPU later)
	inline DatasetContainer::Pointer GetMovingPtsContainer() const { return m_MovingPts; }

	/// Returns pointer to the set of moving points as a 1-D linearized float array of size NumPts*Dim
	inline float* GetMovingPts() const { return m_MovingPts->GetBufferPointer(); }

	/// Set weights for NN search such that and \sum{weights for dim geq 3 (typically RGB)} = Weight and \sum{weights for dim leq 2 (typically XYZ)} = 1-Weight 
	void SetPayloadWeight(float Weight);

	/// Run ICP and write resulting transformation to 'Mat', writes final number of iterations and final matrix norm to NumIter and FinalNorm respectively
	void Run(unsigned long* NumIter = NULL, float* FinalNorm = NULL);

	/// Initializes the ICP (setup of data structures, etc)
	void Initialize();

	/// Run next ICP iteration (if you want to call from outside, use this function AFTER initializing the ICP once)
	bool NextIteration();

	/// Return resulting transformation matrix (4x4 linearized)
	inline float* GetTransformationMatrix() const { return m_AccumulateMat->GetBufferPointer(); }

	/// Return container with resulting transformation matrix (4x4 linearized)
	inline MatrixContainer::Pointer GetTransformationMatrixContainer() const { return m_AccumulateMat; }

protected:
	/// Given two sets of points, estimate the optimal rigid transformation
	void EstimateTransformationMatrix(DatasetContainer::Element* Moving, DatasetContainer::Element* Fixed);

	/// Compute Eigenvectors from 4x4 symmetric matrix
	void Jacobi4x4(float *Matrix, float *Eigenvalues, float *Eigenvectors);

	/// Maximum number of ICP iterations
	const unsigned long m_MaxIter;

	/// Threshold for convergence criterion (Frobenius norm of intermediate transformation matrix - Id(4x4))
	const float m_NormThreshold;

	/// Stores current number of iterations
	int m_Iter;

	/// Stores current Frobenius norm (implicates convergence)
	float m_FrobNorm;

	/// RBC data structure for efficient NN search
	RBC<NumPts, Dim>* m_RBC;

	/// Pointer to set of fixed points
	typename DatasetContainer::Pointer m_FixedPts;

	/// Pointer to set of moving points
	typename DatasetContainer::Pointer m_MovingPts;

	/// Indicates, whether set of fixed points is already initialized
	bool m_FixedPtsInitialized;

	/// Indicates, whether set of moving points is already initialized
	bool m_MovingPtsInitialized;

	/// Container for intermediate transformation matrix
	typename MatrixContainer::Pointer m_TmpMat;

	/// Container for accumulated transformation matrix (ICP output)
	typename MatrixContainer::Pointer m_AccumulateMat;

	/// Holds memory for temporary data (e.g. computing centroid)
	typename DatasetContainer::Pointer m_TmpCudaMemory;

	/// Used for transferring centroids an M matrix (3+3+9 elements) from GPU to CPU
	typename DatasetContainer::Pointer m_CentroidsAndMMatrix;

private:
	/// Purposely not implemented
	ICP(ICP&);

	/// Purposely not implemented
	void operator=(const ICP&); 
};


#endif // ICP_H__