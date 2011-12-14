#include <algorithm>
#include <iterator>


//----------------------------------------------------------------------------
extern "C"
void CUDABuildRBC(float* data, float* weights, RepGPU* reps, unsigned long* NNLists, unsigned long* pointToRep, unsigned long numReps, unsigned long numPts, unsigned long dim);

extern "C"
void CUDAQueryRBC(float* data, float* weights, float* query, RepGPU* reps, unsigned long* NNIndices, unsigned long numReps, unsigned long numPts, unsigned long dim);
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
RBC<NumPts, Dim>::RBC(unsigned long reps) : m_NumReps(reps != 0 ? reps : static_cast<int>(sqrt(static_cast<float>(NumPts))))
{
	// Set dataset container to NULL (will be provided later by RBC::BuildRBC() call)
	m_Dataset = NULL;

	// Init container for representatives (can be done now, because m_NumReps is constant and therefore it will not change)
	m_Reps = new RepGPU[m_NumReps];
	ritkCudaSafeCall(cudaMalloc((void**)&(m_devReps), NumPts*sizeof(RepGPU)));

	// Make RBC indices reproducable
	srand(0);

	// Generate representative indices randomly in the range from [0;NumPts[
	for (int i = 0; i < m_NumReps; ++i)
	{
		int repIdx = rand() % NumPts;

		// Exclude duplicates
		bool duplicate = false;
		for (int j = 0; j < i; ++j)
		{
			if (m_Reps[j].repIdx == repIdx)
			{
				duplicate = true;
			}
		}

		if (duplicate)
		{
			--i;
			continue;
		}

		m_Reps[i].repIdx = repIdx;
	}

	// Copy representatives (so far only initialized with repIdx) to GPU
	ritkCudaSafeCall(cudaMemcpy(m_devReps, m_Reps, m_NumReps*sizeof(RepGPU), cudaMemcpyHostToDevice));

	// Init container for query results (array of indices of NNs)
	m_NNIndices = IndicesContainer::New(); // Set size later, since it depends on number of query points

	// Init container for RBC indices, also uses during RBC construction
	m_PointToRep = IndicesContainer::New();
	m_PointToRep->SetContainerSize(NumPts, 1);
	m_PointToRep->Reserve(NumPts*1);
	m_PointToRep->SynchronizeDevice();

	// Init container for NN lists (a single 1-D array, each representative knows its offset and length)
	m_NNLists = IndicesContainer::New();
	m_NNLists->SetContainerSize(NumPts, 1);
	m_NNLists->Reserve(NumPts*1);
	m_NNLists->SynchronizeDevice();

	// Init container for array of weights
	m_Weights = WeightsContainer::New();
	m_Weights->SetContainerSize(Dim, 1); // 1-D array of weights
	m_Weights->Reserve(Dim*1);
	std::fill(m_Weights->GetBufferPointer(), m_Weights->GetBufferPointer()+Dim, 1.f); // Init weights to 1.f
	m_Weights->SynchronizeDevice(); // Transfer weights to GPU
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
RBC<NumPts, Dim>::~RBC()
{
	// GPU memory should be released automatically

	delete [] m_Reps;
	ritkCudaSafeCall(cudaFree(m_devReps));
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
void RBC<NumPts, Dim>::BuildRBC(DatasetContainer::Pointer Dataset)
{
	m_Dataset = Dataset;
	// TODO how do I know, if m_Dataset needs to be synchronized now?
	//m_Dataset->SynchronizeDevice();

	// Call the GPU routine that takes care of the rest
	CUDABuildRBC(
		m_Dataset->GetCudaMemoryPointer(),
		m_Weights->GetCudaMemoryPointer(),
		m_devReps,
		m_NNLists->GetCudaMemoryPointer(),
		m_PointToRep->GetCudaMemoryPointer(),
		m_NumReps,
		NumPts,
		Dim
		);
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
unsigned long* RBC<NumPts, Dim>::Query(DatasetContainer::Pointer QueryPts)
{
	// Resize container for NN search result
	int numQueryPts = QueryPts->Capacity()/Dim;
	if (m_NNIndices->Capacity() != numQueryPts)
	{
		m_NNIndices->SetContainerSize(numQueryPts, 1);
		m_NNIndices->Reserve(numQueryPts*1);
	}

	// Compute the NNs on GPU
	CUDAQueryRBC(
		m_Dataset->GetCudaMemoryPointer(),
		m_Weights->GetCudaMemoryPointer(),
		QueryPts->GetCudaMemoryPointer(),
		m_devReps,
		m_NNIndices->GetCudaMemoryPointer(),
		m_NumReps,
		NumPts,
		Dim
		);

	// Copy computed NN indices from GPU to CPU
	m_NNIndices->SynchronizeHost();

	return m_NNIndices->GetBufferPointer();
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
void RBC<NumPts, Dim>::SetWeights(float Weights[Dim])
{
	// copy weights to our internal container
	std::copy(Weights, Weights+Dim, stdext::checked_array_iterator<float*>(m_Weights->GetBufferPointer(), Dim));

	// synchronize device (copy data to GPU)
	m_Weights->SynchronizeDevice();
}


//----------------------------------------------------------------------------
template<unsigned long NumPts, unsigned long Dim>
void RBC<NumPts, Dim>::SetWeight(unsigned long DimNew, float Weight)
{
	// copy weight to our internal container
	m_Weights->GetBufferPointer()[DimNew] = Weight;

	// synchronize device (copy data to GPU)
	m_Weights->SynchronizeDevice();
}