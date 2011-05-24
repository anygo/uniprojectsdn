#include "ExtendedICPTransform.h"

#include "vtkCellLocator.h"
#include "vtkDataSet.h"
#include "vtkLandmarkTransform.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkPoints.h"
#include "vtkTransform.h"
#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"

#include <complex>

vtkStandardNewMacro(ExtendedICPTransform);


//----------------------------------------------------------------------------
ExtendedICPTransform::ExtendedICPTransform() : vtkLinearTransform()
{
	m_Source = vtkSmartPointer<vtkPolyData>::New();
	m_Target = vtkSmartPointer<vtkPolyData>::New();
	m_LandmarkTransform = vtkSmartPointer<vtkLandmarkTransform>::New();
	m_MaxIter = 500;
	m_MaxMeanDist = 0.0001;
	m_MaxLandmarks = 1000;

	m_NumIter = 0;
	m_MeanDist = 0.0;
}

ExtendedICPTransform::~ExtendedICPTransform() {}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::Inverse()
{
	vtkSmartPointer<vtkPolyData> tmp = m_Source;
	m_Source = m_Target;
	m_Target = tmp;
	Modified();
}
vtkAbstractTransform *
ExtendedICPTransform::MakeTransform()
{
	return ExtendedICPTransform::New();
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::SetSource(vtkPolyData *source)
{
	m_Source = source;
	Modified();
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::SetTarget(vtkPolyData *target)
{
	m_Target = target;
	Modified();
}
//----------------------------------------------------------------------------
void ExtendedICPTransform::PrintSelf(ostream& os, vtkIndent indent)
{
	os << indent << "PrintSelf() not implemented";
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::InternalUpdate() 
{
	// create locator
	vtkSmartPointer<vtkCellLocator> locator =
		vtkSmartPointer<vtkCellLocator>::New();
	locator->SetDataSet(m_Target);
	locator->SetNumberOfCellsPerBucket(1);
	locator->BuildLocator();

	// create two sets of points to handle iteration
	int step = 1;
	if (m_Source->GetNumberOfPoints() > m_MaxLandmarks)
	{
		step = m_Source->GetNumberOfPoints() / m_MaxLandmarks;
	}

	vtkIdType nb_points = m_Source->GetNumberOfPoints() / step;

	// allocate some points and color arrays
	vtkSmartPointer<vtkPoints> points1 =
		vtkSmartPointer<vtkPoints>::New();
	points1->SetNumberOfPoints(nb_points);
	vtkSmartPointer<vtkDoubleArray> colors1 =
		vtkSmartPointer<vtkDoubleArray>::New();		
	colors1->SetNumberOfComponents(4);

	vtkSmartPointer<vtkPoints> closestp =
		vtkSmartPointer<vtkPoints>::New();
	closestp->SetNumberOfPoints(nb_points);
	vtkSmartPointer<vtkDoubleArray> colorsClosest =
		vtkSmartPointer<vtkDoubleArray>::New();		
	colorsClosest->SetNumberOfComponents(4);

	vtkSmartPointer<vtkPoints> points2 =
		vtkSmartPointer<vtkPoints>::New();
	points2->SetNumberOfPoints(nb_points);
	vtkSmartPointer<vtkDoubleArray> colors2 =
		vtkSmartPointer<vtkDoubleArray>::New();		
	colors2->SetNumberOfComponents(4);

	// fill with initial positions (sample dataset using step)
	vtkSmartPointer<vtkTransform> accumulate =
		vtkTransform::New();
	accumulate->PostMultiply();

	vtkIdType i;
	int j;
	double p1[3], p2[3];

	for (i = 0, j = 0; i < nb_points; i++, j += step)
	{
		points1->SetPoint(i, m_Source->GetPoint(j));
		colors1->InsertTuple(i, m_Source->GetPointData()->GetScalars()->GetTuple(i));
	}

	// go
	vtkSmartPointer<vtkPoints> temp;
	vtkSmartPointer<vtkDoubleArray> tempcolor;
	vtkSmartPointer<vtkPoints> a;
	a->ShallowCopy(points1);
	vtkSmartPointer<vtkDoubleArray> acolor;
	vtkSmartPointer<vtkPoints> b;
	b->ShallowCopy(points2);
	vtkSmartPointer<vtkDoubleArray> bcolor;
	vtkIdType cell_id;
	int sub_id;
	double dist2;
	double outPoint[3];
	double totaldist;

	std::cout << "1" << std::endl;

	m_NumIter = 0;

	while (true)
	{
		// fill points with the closest points to each vertex in input
		for(i = 0; i < nb_points; i++)
		{
			double x[6];
			double *tmp;
			
			std::cout << "2" << std::endl;
			// extract x, y and z coords
			tmp = a->GetPoint(i);
			std::cout << "3" << std::endl;
			x[0] = tmp[0]; x[1] = tmp[1]; x[2] = tmp[2];

			// extract R, G and B (color)
			tmp = acolor->GetTuple(i);
			std::cout << "4" << std::endl;
			x[3] = tmp[0]; x[4] = tmp[1]; x[5] = tmp[2];

			std::cout << "5" << std::endl;
			locator->FindClosestPoint(x, outPoint, cell_id, sub_id, dist2);
			std::cout << "6" << std::endl;
			closestp->SetPoint(i, outPoint);
		}

		// build the landmark transform
		m_LandmarkTransform->SetSourceLandmarks(a);
		m_LandmarkTransform->SetTargetLandmarks(closestp);
		m_LandmarkTransform->Update();

		// concatenate
		accumulate->Concatenate(m_LandmarkTransform->GetMatrix());

		m_NumIter++;
		if (m_NumIter >= m_MaxIter) 
		{
			break;
		}

		// move mesh and compute mean distance
		totaldist = 0.0;

		for(i = 0; i < nb_points; i++)
		{
			a->GetPoint(i, p1);
			m_LandmarkTransform->InternalTransformPoint(p1, p2);
			b->SetPoint(i, p2);

			double absDist = std::abs(p1[0] - p2[0]) + std::abs(p1[1] - p2[1]) + std::abs(p1[2] - p2[2]);
			totaldist += std::log(absDist);
		}

		m_MeanDist = totaldist / (double)nb_points;

		if (m_MeanDist <= m_MaxMeanDist)
		{
			break;
		}

		// swapping
		temp = a;
		a = b;
		b = temp;
	} 

	// now recover accumulated result
	this->Matrix->DeepCopy(accumulate->GetMatrix());
}