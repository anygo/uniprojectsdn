#ifndef ExtendedICPTransform_H__
#define	ExtendedICPTransform_H__

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

	inline int GetNumIter() { return m_NumIter; }
	inline int GetMeanDist() { return m_MeanDist; }
	inline vtkLandmarkTransform* GetLandmarkTransform() { return m_LandmarkTransform; }

protected:
	ExtendedICPTransform();
	~ExtendedICPTransform();

	void InternalUpdate();

	vtkSmartPointer<vtkPolyData> m_Source;
	vtkSmartPointer<vtkPolyData> m_Target;
	int m_MaxIter;
	int m_MaxLandmarks;
	double m_MaxMeanDist;
	

	int m_NumIter;
	double m_MeanDist;
	vtkSmartPointer<vtkLandmarkTransform> m_LandmarkTransform;

private:
	ExtendedICPTransform(const ExtendedICPTransform&);  // Not implemented.
	void operator=(const ExtendedICPTransform&);  // Not implemented.
};

#endif // ExtendedICPTransform_H__