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
unsigned long int
ExtendedICPTransform::GetMTime()
{
	unsigned long result = this->vtkLinearTransform::GetMTime();
	unsigned long mtime;

	if (m_Source)
	{
		mtime = m_Source->GetMTime(); 
		if (mtime > result)
		{
			result = mtime;
		}
	}

	if (m_Target)
	{
		mtime = m_Target->GetMTime(); 
		if (mtime > result)
		{
			result = mtime;
		}
	}

	if (m_LandmarkTransform)
	{
		mtime = m_LandmarkTransform->GetMTime();
		if (mtime > result)
		{
			result = mtime;
		}
	}

	return result;
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
void
ExtendedICPTransform::PrintSelf(ostream& os, vtkIndent indent)
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

	// Allocate some points.
	vtkSmartPointer<vtkPoints> points1 =
		vtkSmartPointer<vtkPoints>::New();
	points1->SetNumberOfPoints(nb_points);

	vtkSmartPointer<vtkPoints> closestp =
		vtkSmartPointer<vtkPoints>::New();
	closestp->SetNumberOfPoints(nb_points);

	vtkSmartPointer<vtkPoints> points2 =
		vtkSmartPointer<vtkPoints>::New();
	points2->SetNumberOfPoints(nb_points);

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
	}

	// go
	vtkSmartPointer<vtkPoints> temp;
	vtkSmartPointer<vtkPoints> a = points1;
	vtkSmartPointer<vtkPoints> b = points2;
	vtkIdType cell_id;
	int sub_id;
	double dist2;
	double outPoint[3];
	double totaldist;

	m_NumIter = 0;

	while (true)
	{
		// fill points with the closest points to each vertex in input
		for(i = 0; i < nb_points; i++)
		{
			locator->FindClosestPoint(a->GetPoint(i), outPoint, cell_id, sub_id, dist2);
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

			switch (m_Metric)
			{
			case ExtendedICPTransform::ABSOLUTE_DISTANCE:
				totaldist += std::abs(p1[0] - p2[0]) + std::abs(p1[1] - p2[1]) + std::abs(p1[2] - p2[2]); break;
			case ExtendedICPTransform::LOG_ABSOLUTE_DISTANCE:
				totaldist += std::log(std::abs(p1[0] - p2[0]) + std::abs(p1[1] - p2[1]) + std::abs(p1[2] - p2[2]) + 1.0); break;
			case ExtendedICPTransform::SQUARED_DISTANCE:
				totaldist += vtkMath::Distance2BetweenPoints(p1, p2);
			}

		}

		m_MeanDist = totaldist / (double)nb_points;
		std::cout << "Iteration " << m_NumIter << ":\t mean distance = " << m_MeanDist << std::endl;
			
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