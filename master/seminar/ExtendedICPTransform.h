#ifndef ExtendedICPTransform_H__
#define	ExtendedICPTransform_H__

#include <StitchingWidget.h>

#include <vtkLandmarkTransform.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>

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

	// virtual methods that have to be defined
	vtkAbstractTransform *MakeTransform();
	void Inverse();

	// set and get methods
	inline void SetMaxIter(int iter) { m_MaxIter = iter; }
	inline int GetMaxIter() { return m_MaxIter; }
	inline void SetNumLandmarks(int landmarks) { m_NumLandmarks = landmarks; }
	inline int GetNumLandmarks() { return m_MaxIter; }
	inline void SetMaxMeanDist(float dist) { m_MaxMeanDist = dist; }
	inline int GetMaxMeanDist() { return m_MaxMeanDist; }
	inline void SetOutlierRate(float percentage) { m_OutlierRate = percentage; }
	inline void SetRemoveOutliers(bool remove) { m_RemoveOutliers = remove; }
	inline void SetClosestPointFinder(ClosestPointFinder* cpf) { m_ClosestPointFinder = cpf; }
	inline void SetNormalizeRGBToDistanceValuesFactor(float factor) { m_NormalizeRGBToDistanceValuesFactor = factor; }

	inline int GetNumIter() { return m_NumIter; }
	inline float GetMeanDist() { return m_MeanDist; }
	inline float GetMeanTargetDistance() { return m_MeanTargetDistance; }
	inline vtkLandmarkTransform* GetLandmarkTransform() { return m_LandmarkTransform; }

protected:
	ExtendedICPTransform();
	~ExtendedICPTransform();

	void InternalUpdate();
	unsigned long int GetMTime();
	void vtkPolyDataToPointCoordsAndColors();
	void vtkPolyDataToPointCoords(vtkSmartPointer<vtkPoints> poly, PointCoords* coords);

	vtkSmartPointer<vtkPolyData> m_Source;
	vtkSmartPointer<vtkPolyData> m_Target;

	PointCoords* m_SourceCoords;
	PointCoords* m_TargetCoords;
	PointColors* m_SourceColors;
	PointColors* m_TargetColors;

	ClosestPointFinder* m_ClosestPointFinder;

	int m_MaxIter;
	int m_NumLandmarks;
	float m_MaxMeanDist;
	float m_OutlierRate;
	bool m_RemoveOutliers;

	int m_NumIter;
	float m_MeanDist;
	float m_MeanTargetDistance;
	float m_NormalizeRGBToDistanceValuesFactor;
	vtkSmartPointer<vtkLandmarkTransform> m_LandmarkTransform;

private:
	ExtendedICPTransform(const ExtendedICPTransform&);  // Not implemented.
	void operator=(const ExtendedICPTransform&);  // Not implemented.
};

#endif // ExtendedICPTransform_H__