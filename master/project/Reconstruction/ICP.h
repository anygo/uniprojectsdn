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
template<uint NumPts, uint Dim>
class ICP
{
public:
	/// Constructor 
	ICP(uint maxIter = 100) : m_MaxIter(maxIter) { /* TODO */ }

	/// Destructor
	~ICP() { /* TODO */ }

	/// Set fixed points; indicate whether 'fixed' points to GPU memory using devMem
	void SetFixedPts(float** fixed, bool devMem) { m_FixedPts = fixed; /* copy to GPU, pass to RBC and build RBC; */ }

	/// Set moving points; indicate whether 'moving' points to GPU memory using devMem
	void SetMovingPts(float** moving, bool devMem) { m_MovingPts = moving; /* copy to GPU */ }

	/// Set maximum number of iterations
	void SetMaxIter(uint iter) { m_MaxIter = iter; }

	/// Set weights for NN search such that weight[dim<3] = 1.f and weight[dim>=3] = w
	void SetWeight(float w) { float weights[Dim]; for (int i = 0; i < 3; ++i) weights[i] = 1; for (int i = 3; i < Dim; ++i) weights[i] = w; m_RBC.SetWeights(weights); }

	/// Run ICP and write resulting transformation to 'Mat'
	void Run() { /* ICP; resulting transformation matrix will be written to Mat */ }

	/// Accumulated transformation matrix
	float Mat[4][4];

protected:
	/// RBC data structure for efficient NN search
	RBC<NumPts, Dim> m_RBC;

	/// Pointer to set of fixed points
	float** m_FixedPts;

	/// Pointer to set of moving points
	float** m_MovingPts;

	/// Maximum number of ICP iterations
	uint m_MaxIter;

private:
	/// Purposely not implemented
	ICP(ICP&);

	/// Purposely not implemented
	void operator=(const ICP&); 
};

#endif // ICP_H__