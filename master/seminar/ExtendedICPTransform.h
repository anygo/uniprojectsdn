#ifndef ExtendedICPTransform_H__
#define	ExtendedICPTransform_H__

#include <StitchingWidget.h>

#include <vtkLandmarkTransform.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>

#include "defs.h"
#include "ClosestPointFinder.h"

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
	inline void SetMaxMeanDist(double dist) { m_MaxMeanDist = dist; }
	inline int GetMaxMeanDist() { return m_MaxMeanDist; }
	inline void SetMetric(int metric) { m_Metric = metric; }
	inline int GetMetric() { return m_Metric; }
	inline void SetClosestPointFinder(ClosestPointFinder* cpf) { m_ClosestPointFinder = cpf; }

	inline int GetNumIter() { return m_NumIter; }
	inline double GetMeanDist() { return m_MeanDist; }
	inline vtkLandmarkTransform* GetLandmarkTransform() { return m_LandmarkTransform; }

protected:
	ExtendedICPTransform();
	~ExtendedICPTransform();

	void InternalUpdate();
	unsigned long int GetMTime();
	void vtkPolyDataToPoint6DArray();
	void vtkPolyDataToPoint6DArray(vtkSmartPointer<vtkPoints> poly, Point6D* point);

	vtkSmartPointer<vtkPolyData> m_Source;
	vtkSmartPointer<vtkPolyData> m_Target;

	Point6D* m_SourcePoints;
	Point6D* m_TargetPoints;

	ClosestPointFinder* m_ClosestPointFinder;

	int m_MaxIter;
	int m_NumLandmarks;
	double m_MaxMeanDist;
	int m_Metric;
	

	int m_NumIter;
	double m_MeanDist;
	vtkSmartPointer<vtkLandmarkTransform> m_LandmarkTransform;

private:
	ExtendedICPTransform(const ExtendedICPTransform&);  // Not implemented.
	void operator=(const ExtendedICPTransform&);  // Not implemented.
};

#endif // ExtendedICPTransform_H__