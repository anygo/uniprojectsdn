#ifndef ICP_H__
#define ICP_H__

#include "RBCOneShot.h"

/**	@class		ICP
 *	@brief		ICP algorithm
 *	@author		Dominik Neumann
 *
 *	@details
 *	Implementation of the iterative closest point (ICP) algorithm, based on the VTK implementation
 */
class ICP
{
public:
	// constructor 
	ICP() { std::cout << "ICP()" << std::endl; }

	// destructor
	~ICP() { std::cout << "~ICP()" << std::endl; }

private:
	
};

#endif // ICP_H__