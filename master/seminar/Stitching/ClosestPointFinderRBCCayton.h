#ifndef ClosestPointFinderRBCCAYTON_H__
#define ClosestPointFinderRBCCAYTON_H__

#include "ClosestPointFinder.h"
#include<stdlib.h>
#include<cuda.h>
#include<math.h>
#include "rbc/defsRBC.h"
#include "rbc/utils.h"
#include "rbc/utilsgpu.h"
#include "rbc/rbc.h"
#include "rbc/brute.h"
#include <iostream>


/**	@class		ClosestPointFinderRBCCayton
 *	@brief		RandomBallCover ClosestPointFinder on GPU (CUDA)
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Class that implements the ClosestPointFinder Interface by using
 *	the Random Ball Cover acceleration structure. Uses GPU capabilities.
 */
class ClosestPointFinderRBCCayton : public ClosestPointFinder
{
public:
	ClosestPointFinderRBCCayton(int NrOfPoints, float nrOfRepsFactor) : ClosestPointFinder(NrOfPoints), m_NrOfRepsFactor(nrOfRepsFactor) 
	{ 
		nnsRBC = new unint[m_NrOfPoints];

		initMat( &m_Source, m_NrOfPoints, 6 );
		m_Source.mat = (real*)calloc( sizeOfMat(m_Source), sizeof(*(m_Source.mat)) );
	}

	~ClosestPointFinderRBCCayton() 
	{
		delete[] nnsRBC;
		

		free(m_Source.mat);
	}

	unsigned short* FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors);
	void SetTarget(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors);
	inline bool usesGPU() { return false; }

protected:
	void initRBC();

	matrix m_Target;
	matrix m_Source;
	rbcStruct m_rbcS;
	int m_NrOfReps;
	float m_NrOfRepsFactor;
	unint* nnsRBC;
	real* distsRBC;

};

#endif // ClosestPointFinderRBCCAYTON_H__
