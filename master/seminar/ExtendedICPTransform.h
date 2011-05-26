#ifndef ExtendedICPTransform_H__
#define	ExtendedICPTransform_H__

#include <StitchingWidget.h>

#include <vtkLandmarkTransform.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>


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
	inline void SetMaxLandmarks(int landmarks) { m_MaxLandmarks = landmarks; }
	inline int GetMaxLandmarks() { return m_MaxIter; }
	inline void SetMaxMeanDist(double dist) { m_MaxMeanDist = dist; }
	inline int GetMaxMeanDist() { return m_MaxMeanDist; }
	inline void SetMetric(int metric) { m_Metric = metric; }
	inline int GetMetric() { return m_Metric; }

	inline int GetNumIter() { return m_NumIter; }
	inline double GetMeanDist() { return m_MeanDist; }
	inline vtkLandmarkTransform* GetLandmarkTransform() { return m_LandmarkTransform; }

	enum ICP_METRIC {
		LOG_ABSOLUTE_DISTANCE,
		ABSOLUTE_DISTANCE,
		SQUARED_DISTANCE
	};

	typedef struct Point6D {
		double x, y, z;
		double r, g, b;
	} Point6D;

protected:
	ExtendedICPTransform();
	~ExtendedICPTransform();

	void InternalUpdate();
	unsigned long int GetMTime();
	void vtkPolyDataToPoint6DArray();
	void vtkPolyDataToPoint6DArray(vtkSmartPointer<vtkPoints> poly, Point6D *point);
	int* FindClosestPoints(Point6D *source, Point6D* target);
	int FindClosestPoint(Point6D source, Point6D* target);

	vtkSmartPointer<vtkPolyData> m_Source;
	vtkSmartPointer<vtkPolyData> m_Target;

	Point6D * m_SourcePoints;
	Point6D * m_TargetPoints;

	int m_MaxIter;
	int m_MaxLandmarks;
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