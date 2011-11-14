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

	// returns pointer to RBC data structure
	RBC<Dim, NumPts>* GetRBC() { return &m_RBC; }

private:
	// RBC data structure for efficient NN search
	RBC<Dim, NumPts> m_RBC;

	//
	float** m_FixedPts;
	float** m_MovingPts;

	// maximum number of ICP iterations
	uint m_MaxIter;
};

#endif // ICP_H__