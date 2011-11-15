#ifndef RBC_H__
#define RBC_H__

#include <algorithm>

/**	@class		RBC
 *	@brief		efficient NN search
 *	@author		Dominik Neumann
 *
 *	@details
 *	Class for efficient NN search on the GPU using the one-shot random ball rover (RBC) data structure
 */
template<uint Dim, uint NumPts>
class RBC
{
	// representative structure on CPU (required for initialization)
	typedef struct Rep
	{
		// index of this representative among entire dataset
		unsigned int index;

		// indices of this representative's data points
		std::list<unsigned int> points;
	} Rep;
	
	// representative structure on GPU
	typedef struct RepGPU
	{
		// index of this representative among entire dataset
		uint repIdx;

		// number of points in NN list of this representative
		uint numPts;

		// pointer to indices of this representative's data points (in NN list)
		uint* NNList;
	} RepGPU;

public:
	// constructor (automatically builds RBC data structure)
	RBC(uint reps = 0) : m_NumReps(reps != 0 ? reps : static_cast<int>(sqrt(static_cast<float>(NumPts)))) { std::cout << "RBC()" << std::endl; /* init weights to 1.f, init GPU memory */ }

	// destructor
	~RBC() { std::cout << "~RBC()" << std::endl; /* release GPU memory */ }

	// RBC construction routine; indicate whether queryPoints are already on GPU using devMem
	void BuildRBC(float** dataset, bool devMem) { std::cout << "BuildRBC()" << std::endl; std:: cout << "Dim: " << Dim << " NumPts: " << NumPts << " m_NumReps: " << m_NumReps << std::endl; }

	// NN query for a set of query points; indicate whether queryPoints are already on GPU using devMem; returns indices of (approximative) NNs w.r.t. supplied dataset
	uint* Query(float** queryPts, bool devMem) { std::cout << "Query()" << std::endl; return m_NNIndices; }

	// set weight for particular dimension dim
	void SetWeight(uint dim, float weight) { m_Weights[dim] = weight; /* copy to GPU */ }

	// set weights for all dimensions
	void SetWeights(float weights[Dim]) { for (int i = 0; i < Dim; ++i) std::cout << m_Weights[i]; std::cout << std::endl; std::copy(weights, weights+Dim, m_Weights); for (int i = 0; i < Dim; ++i) std::cout << m_Weights[i]; std::cout << std::endl; /* copy to GPU */ }

private:
	// number of representatives
	uint m_NumReps;

	// pointer to dataset on GPU
	float** m_devDataset;

	// pointer acceleration structure (representative array) on GPU
	RepGPU* m_devReps;

	// array of NN indices (query result) on CPU/GPU
	uint m_NNIndices[NumPts];
	uint* m_devNnIndices;

	// array of weighting factors for each dimension of the data points on CPU/GPU
	float m_Weights[Dim];
	float* m_devWeights;
};

#endif // RBC_H__