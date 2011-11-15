#ifndef ICP_H__
#define ICP_H__

#include "RBC.h"

/**	@class		ICP
 *	@brief		ICP algorithm
 *	@author		Dominik Neumann
 *
 *	@details
 *	Implementation of the iterative closest point (ICP) algorithm, partially based on VTK ICP implementation.
 *	Registration of points is always performed in 3-D. If Dim > 3, additional dimensions will be used for NN search only.
 */
template<uint Dim, uint NumPts>
class ICP
{
public:
	// constructor 
	ICP(uint maxIter = 100) : m_MaxIter(maxIter) { std::cout << "ICP()" << std::endl; }

	// destructor
	~ICP() { std::cout << "~ICP()" << std::endl; }

	// set fixed and moving point cloud
	void SetPts(float** fixed, float** moving) { std::cout << "SetPts()" << std::endl; }

	// set maximum number of iterations
	void SetMaxIter(uint iter) { m_MaxIter = iter; }

	// set weights for NN search such that weight[dim<3] = 1.f; weight[dim>=3] = w
	void SetWeight(float w) { float weights[Dim]; for (int i = 0; i < 3; ++i) weights[i] = 1; for (int i = 3; i < Dim; ++i) weights[i] = w; m_RBC.SetWeights(weights); }

private:
	// RBC data structure for efficient NN search
	RBC<Dim, NumPts> m_RBC;

	// pointer to set of fixed points
	float** m_FixedPts;

	// pointer to set of moving points
	float** m_MovingPts;

	// maximum number of ICP iterations
	uint m_MaxIter;
};

#endif // ICP_H__