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
class ExtendedICPTransform : public vtkLinearTransform
{
public:
	static ExtendedICPTransform *New();
	vtkTypeMacro(ExtendedICPTransform, vtkLinearTransform);
	void PrintSelf(ostream& os, vtkIndent indent);

	void SetSource(vtkPolyData *source);
	void SetTarget(vtkPolyData *target);

	void vtkPolyDataToPointCoordsAndColors(double clipPercentage);

	// virtual methods that have to be defined
	vtkAbstractTransform *MakeTransform();
	void Inverse();

	// set and get methods
	inline void SetMaxIter(int iter) { m_MaxIter = iter; }
	inline int GetMaxIter() { return m_MaxIter; }
	inline void SetNumLandmarks(int landmarks) { 

		m_NumLandmarks = landmarks;

		if(m_SourceCoords) delete[] m_SourceCoords;
		if(m_SourceColors) delete[] m_SourceColors;
		if(m_TargetCoords) delete[] m_TargetCoords;
		if(m_TargetColors) delete[] m_TargetColors;
		if(m_Distances) delete[] m_Distances;

		// allocate some points used for icp
		m_SourceCoords = new PointCoords[m_NumLandmarks];
		m_SourceColors = new PointColors[m_NumLandmarks];
		m_TargetCoords = new PointCoords[m_NumLandmarks];
		m_TargetColors = new PointColors[m_NumLandmarks];

		// for gpu based distance computation
		m_Distances = new float[m_NumLandmarks];
		
		m_Points1 = vtkSmartPointer<vtkPoints>::New();
		m_Points1->SetNumberOfPoints(m_NumLandmarks);
		m_Points2 = vtkSmartPointer<vtkPoints>::New();
		m_Points2->SetNumberOfPoints(m_NumLandmarks);
		m_Closestp = vtkSmartPointer<vtkPoints>::New();
		m_Closestp->SetNumberOfPoints(m_NumLandmarks);

	}
	inline int GetNumLandmarks() { return m_MaxIter; }
	inline void SetMaxMeanDist(float dist) { m_MaxMeanDist = dist; }
	inline int GetMaxMeanDist() { return m_MaxMeanDist; }
	inline void SetOutlierRate(float percentage) { m_OutlierRate = percentage; }
	inline void SetRemoveOutliers(bool remove) { m_RemoveOutliers = remove; }
	inline void SetClosestPointFinder(ClosestPointFinder* cpf) { m_ClosestPointFinder = cpf; }
	inline void SetNormalizeRGBToDistanceValuesFactor(float factor) { m_NormalizeRGBToDistanceValuesFactor = factor; }
	inline void SetPreviousTransformMatrix(vtkMatrix4x4* m) { m_PreviousTransformationMatrix = m; }
	inline void SetApplyPreviousTransform(bool apply) { m_ApplyPreviousTransform = apply; }

	inline int GetNumIter() { return m_NumIter; }
	inline float GetMeanDist() { return m_MeanDist; }
	inline float GetMeanTargetDistance() { return m_MeanTargetDistance; }
	inline vtkLandmarkTransform* GetLandmarkTransform() { return m_LandmarkTransform; }

protected:
	ExtendedICPTransform();
	~ExtendedICPTransform();

	void InternalUpdate();
	unsigned long int GetMTime();
	void vtkPolyDataToPointCoords(vtkSmartPointer<vtkPoints> poly, PointCoords* coords);

	vtkSmartPointer<vtkPolyData> m_Source;
	vtkSmartPointer<vtkPolyData> m_Target;

	PointCoords* m_SourceCoords;
	PointCoords* m_TargetCoords;
	PointColors* m_SourceColors;
	PointColors* m_TargetColors;

	vtkSmartPointer<vtkPoints> m_Points1;
	vtkSmartPointer<vtkPoints> m_Points2;
	vtkSmartPointer<vtkPoints> m_Closestp;

	ClosestPointFinder* m_ClosestPointFinder;
	float* m_Distances;

	int m_MaxIter;
	int m_NumLandmarks;
	float m_MaxMeanDist;
	float m_OutlierRate;
	bool m_RemoveOutliers;
	vtkSmartPointer<vtkMatrix4x4> m_PreviousTransformationMatrix;
	bool m_ApplyPreviousTransform;

	int m_NumIter;
	float m_MeanDist;
	float m_MeanTargetDistance;
	float m_NormalizeRGBToDistanceValuesFactor;
	vtkSmartPointer<vtkLandmarkTransform> m_LandmarkTransform;
	vtkSmartPointer<vtkTransform> m_Accumulate;

private:
	ExtendedICPTransform(const ExtendedICPTransform&);  // Not implemented.
	void operator=(const ExtendedICPTransform&);  // Not implemented.
};

#endif // ExtendedICPTransform_H__