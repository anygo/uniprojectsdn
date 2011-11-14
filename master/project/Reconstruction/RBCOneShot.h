#ifndef RBCONESHOT_H__
#define RBCONESHOT_H__

/**	@class		RBCOneShot
 *	@brief		efficient NN search
 *	@author		Dominik Neumann
 *
 *	@details
 *	Class for efficient NN search on the GPU using the one-shot Random Ball Cover (RBC) data structure,
 */
template<uint Dim, uint NumPts>
class RBCOneShot
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
	RBCOneShot(uint reps = 0) : m_NumReps(reps != 0 ? reps : static_cast<int>(sqrt(static_cast<float>(NumPts)))) { std::cout << "RBCOneShot()" << std::endl; InitGPUMem(); }

	// destructor
	~RBCOneShot() { std::cout << "~RBCOneShot()" << std::endl; ReleaseGPUMem(); }

	// RBC construction routine (requires correctly initialized GPU memory, done in constructor)
	void Init(float** dataset) { std::cout << "Init()" << std::endl; std:: cout << "Dim: " << Dim << " NumPts: " << NumPts << " m_NumReps: " << m_NumReps << std::endl; }

	// NN query for a set of query points; indicate whether queryPoints are already on GPU using devMem
	void SetQueryPts(float** queryPts, bool devMem) { std::cout << "Query()" << std::endl; return m_NNIndices; }

	// NN query for the previous set of query points (e.g. if points were modified directly on GPU); returns indices of resulting NNs w.r.t. supplied dataset
	uint* Query() { std::cout << "Query()" << std::endl; return m_NNIndices; }

private:
	// allocate memory for RBC data structure on GPU
	void InitGPUMem() { std::cout << "InitGPUMem()" << std::endl; }

	// release all GPU memory
	void ReleaseGPUMem() { std::cout << "ReleaseGPUMem()" << std::endl; }	
	
	// number of representatives
	uint m_NumReps;

	// pointer to dataset on GPU
	float** m_devDataset;

	// pointer acceleration structure (representative array) on GPU
	RepGPU* m_devReps;

	// pointer to query points on GPU
	float** m_devQueryPts;

	// pointers to NN indices (query result) on CPU/GPU
	uint m_NNIndices[NumPts];
	uint* m_devNnIndices;
};

#endif // RBCONESHOT_H__