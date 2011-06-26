#include "ClosestPointFinderRBCCayton.h"
#include <QTime>
#include <iostream>


void
ClosestPointFinderRBCCayton::SetTarget(PointCoords* targetCoords, PointColors* targetColors, PointCoords* sourceCoords, PointColors* sourceColors)
{ 
	ClosestPointFinder::SetTarget(targetCoords, targetColors, sourceCoords, sourceColors);
	initRBC();
}



unsigned short*
ClosestPointFinderRBCCayton::FindClosestPoints(PointCoords* sourceCoords, PointColors* sourceColors)
{
	for(unsigned int i = 0; i < m_Target.r; i++)
	{
		m_Source.mat[IDX( i, 0, m_Source.ld )] = sourceCoords[i].x;
		m_Source.mat[IDX( i, 1, m_Source.ld )] = sourceCoords[i].y;
		m_Source.mat[IDX( i, 2, m_Source.ld )] = sourceCoords[i].z;
		m_Source.mat[IDX( i, 3, m_Source.ld )] = sourceColors[i].r;
		m_Source.mat[IDX( i, 4, m_Source.ld )] = sourceColors[i].g;
		m_Source.mat[IDX( i, 5, m_Source.ld )] = sourceColors[i].b;
	}



	//QTime t;
	//t.start();
	//This finds the 32-NNs; if you are only interested in the 1-NN, use queryRBC(..) instead
	queryRBC( m_Source, m_rbcS, nnsRBC, m_Distances );
	//std::cout << "QueryTime() " << t.elapsed() << " ms" << std::endl;

	for(int i = 0; i < m_NrOfPoints; ++i)
	{
		m_Indices[i] = static_cast<unsigned short>(nnsRBC[i]);
	}

	// return the indices which will then be used in the icp algorithm
	return m_Indices;
}
//----------------------------------------------------------------------------

void
ClosestPointFinderRBCCayton::initRBC()
{
	//Build the RBC
	//std::cout << "building the rbc..\n";
	unint numReps = (int)sqrt(static_cast<float>(m_NrOfPoints));
	initMat( &m_Target, m_NrOfPoints, 6 );
	m_Target.mat = (real*)calloc( sizeOfMat(m_Target), sizeof(*(m_Target.mat)) );
	for(unsigned int i = 0; i < m_Target.r; i++)
	{
		m_Target.mat[IDX( i, 0, m_Target.ld )] = m_TargetCoords[i].x;
		m_Target.mat[IDX( i, 1, m_Target.ld )] = m_TargetCoords[i].y;
		m_Target.mat[IDX( i, 2, m_Target.ld )] = m_TargetCoords[i].z;
		m_Target.mat[IDX( i, 3, m_Target.ld )] = m_TargetColors[i].r;
		m_Target.mat[IDX( i, 4, m_Target.ld )] = m_TargetColors[i].g;
		m_Target.mat[IDX( i, 5, m_Target.ld )] = m_TargetColors[i].b;
	}

	QTime t;
	t.start();
	buildRBC( m_Target, &m_rbcS, numReps, numReps );
	std::cout << "buiildRBC(): " << t.elapsed() << std::endl;
}

