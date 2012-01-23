#ifndef RBC_H__
#define RBC_H__

#include "ritkCudaRegularMemoryImportImageContainer.h"
#include "RepGPU.h"


/**	@class		RBC
 *	@brief		Efficient NN search
 *	@author		Dominik Neumann
 *
 *	@details
 *	Class for efficient NN search on the GPU using the one-shot random ball rover (RBC) data structure
 */
template<unsigned long NumPts, unsigned long Dim>
class RBC
{
	/**	@name CUDA memory containers */
	//@{
	typedef ritk::CudaRegularMemoryImportImageContainerF WeightsContainer;
	typedef ritk::CudaRegularMemoryImportImageContainerF DatasetContainer;
	typedef ritk::CudaRegularMemoryImportImageContainerU IndicesContainer;
	//@}

public:
	/// Constructor
	RBC(unsigned long reps = 0); 

	/// Destructor
	virtual ~RBC();

	/// RBC construction routine
	virtual void BuildRBC(DatasetContainer::Pointer Dataset);

	/// NN query for a set of query points; returns indices of (approximative) NNs w.r.t. supplied dataset
	virtual unsigned long* Query(DatasetContainer::Pointer QueryPts, bool SynchronizeCorrespondences = true);

	/// Get container with correspondence indices (if you want to use them on the GPU)
	virtual inline IndicesContainer::Pointer GetCorrespondencesContainer() { return m_NNIndices; }

	/// Set weights for all dimensions
	virtual void SetWeights(const float Weights[Dim]);

	/// Set weight only for particular dimension DimNew
	virtual void SetWeight(unsigned long Idx, float Weight);

protected:
	/// Number of representatives
	const unsigned long m_NumReps;

	/// Container for dataset that is used to build the RBC data structure
	typename DatasetContainer::Pointer m_Dataset;

	/// Acceleration structure (representative array) on GPU
	RepGPU* m_devReps;

	/// Container that holds the NN lists for all representatives in a single 1-D array
	typename IndicesContainer::Pointer m_NNLists;

	/// Container for array of NN indices (query result) on CPU/GPU
	typename IndicesContainer::Pointer m_NNIndices;

	/// Container holding indices, used during RBC construction
	typename IndicesContainer::Pointer m_PointToRep;

	/// Array of weighting factors for each dimension of the data points on CPU/GPU
	typename WeightsContainer::Pointer m_Weights;

private:
	/// purposely not implemented
	RBC(RBC&);

	/// purposely not implemented
	void operator=(const RBC&); 
};


#endif // RBC_H__