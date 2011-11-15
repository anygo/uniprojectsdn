#ifndef RBC_H__
#define RBC_H__

/**	@class		RBC
 *	@brief		Efficient NN search
 *	@author		Dominik Neumann
 *
 *	@details
 *	Class for efficient NN search on the GPU using the one-shot random ball rover (RBC) data structure
 */
template<uint NumPts, uint Dim>
class RBC
{
	/** @name	Representative structure on CPU (required for initialization) */
	//@{
	/// The struct
	typedef struct
	{
		/// Index of this representative among entire dataset
		unsigned int index;

		/// Indices of this representative's data points
		std::list<unsigned int> points;
	} Rep;
	//@}
	
	/** @name	Representative structure on GPU */
	//@{
	/// The struct
	typedef struct
	{
		/// Index of this representative among entire dataset
		uint repIdx;

		/// Number of points in NN list of this representative
		uint numPts;

		/// Pointer to indices of this representative's data points (in NN list)
		uint* NNList;
	} RepGPU;
	//@}

public:
	/// Constructor (automatically builds RBC data structure)
	RBC(uint reps = 0) : m_NumReps(reps != 0 ? reps : static_cast<int>(sqrt(static_cast<float>(NumPts)))) { /* init weights to 1.f, init GPU memory */ }

	/// Destructor
	~RBC() { /* release GPU memory */ }

	/// RBC construction routine; indicate whether data points are already on GPU using devMem
	void BuildRBC(float** dataset, bool devMem) { /* construct RBC */ }

	/// NN query for a set of query points; indicate whether queryPoints are already on GPU using devMem; returns indices of (approximative) NNs w.r.t. supplied dataset
	uint* Query(float** queryPts, bool devMem) { /* query */ return m_NNIndices; }

	/// Set weight for particular dimension dim
	void SetWeight(uint dim, float w) { m_Weights[dim] = w; /* copy to GPU */ }

	/// Set weights for all dimensions
	void SetWeights(float weights[Dim]) { std::copy(weights, weights+Dim, m_Weights); /* copy to GPU */ }

protected:
	/// Number of representatives
	uint m_NumReps;

	/// Pointer to dataset on GPU
	float** m_devDataset;

	/// Pointer acceleration structure (representative array) on GPU
	RepGPU* m_devReps;

	/// Array of NN indices (query result) on CPU/GPU
	uint m_NNIndices[NumPts];
	uint* m_devNNIndices;

	/// Array of weighting factors for each dimension of the data points on CPU/GPU
	float m_Weights[Dim];
	float* m_devWeights;

private:
	/// purposely not implemented
	RBC(RBC&);

	/// purposely not implemented
	void operator=(const RBC&); 
};

#endif // RBC_H__