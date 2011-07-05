#ifndef ExtendedICPTransform_H__
#define	ExtendedICPTransform_H__

#include <FastStitchingWidget.h>

#include <vtkLandmarkTransform.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkMatrix4x4.h>

#include "defs.h"
#include "ClosestPointFinder.h"

/**	@class		ExtendedICPTransform
 *	@brief		Adaption of vtkIterativeClosestPointTransform
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Class that encapsulates the ICP Algorithm, adapted from VTK
 */
class ExtendedICPTransform
{
public:

	ExtendedICPTransform();
	~ExtendedICPTransform();

	void SetSource(float4* devSource);
	void SetTarget(float4* devTarget);

	// set and get methods
	inline void SetNumLandmarks(int landmarks) { 

		m_NumLandmarks = landmarks;

		if(m_Source) delete[] m_Source;
		if(m_Target) delete[] m_Target;
		if(m_Distances) delete[] m_Distances;
		if(m_Indices) delete[] m_Indices;
		if(m_ClosestP) delete[] m_ClosestP;

		// allocate some points used for icp
		m_Source = new float4[m_NumLandmarks];
		m_Target = new float4[m_NumLandmarks];
		m_ClosestP = new float4[m_NumLandmarks];
		m_Distances = new float[m_NumLandmarks];
		m_Indices = new unsigned int[m_NumLandmarks];
	}

	vtkMatrix4x4* StartICP();

	inline void SetMaxIter(int maxIter) { m_MaxIter = maxIter; }
	inline void SetMaxMeanDist(float maxMean) { m_MaxMeanDist = maxMean; }

	inline void SetClosestPointFinder(ClosestPointFinder* cpf) { m_ClosestPointFinder = cpf; }
	inline void SetNormalizeRGBToDistanceValuesFactor(float factor) { m_NormalizeRGBToDistanceValuesFactor = factor; }

	inline int GetNumIter() { return m_NumIter; }
	inline float GetMeanDist() { return m_MeanDist; }
	inline float GetMeanTargetDistance() { return m_MeanTargetDistance; }

protected:

	vtkMatrix4x4* EstimateTransformationMatrix(float4* source, float4* target);

	float4* m_devSource;
	float4* m_devTarget;

	float4* m_Source;
	float4* m_Target;
	float4* m_ClosestP;

	float* m_Distances;
	unsigned int* m_Indices;

	ClosestPointFinder* m_ClosestPointFinder;

	int m_MaxIter;
	int m_NumLandmarks;
	float m_MaxMeanDist;

	int m_NumIter;
	float m_MeanDist;
	float m_MeanTargetDistance;
	float m_NormalizeRGBToDistanceValuesFactor;

	vtkSmartPointer<vtkTransform> m_Accumulate;

};

#endif // ExtendedICPTransform_H__