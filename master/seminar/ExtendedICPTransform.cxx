#include "ExtendedICPTransform.h"

#include "vtkCellLocator.h"
#include "vtkDataSet.h"
#include "vtkLandmarkTransform.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkPoints.h"
#include "vtkTransform.h"

#include <complex>

vtkStandardNewMacro(ExtendedICPTransform);

void
ExtendedICPTransform::InternalUpdate() 
{

	// Check source, target

	if (this->Source == NULL || !this->Source->GetNumberOfPoints())
	{
		vtkErrorMacro(<<"Can't execute with NULL or empty input");
		return;
	}

	if (this->Target == NULL || !this->Target->GetNumberOfPoints())
	{
		vtkErrorMacro(<<"Can't execute with NULL or empty target");
		return;
	}

	// Create locator

	this->CreateDefaultLocator();
	this->Locator->SetDataSet(this->Target);
	this->Locator->SetNumberOfCellsPerBucket(1);
	this->Locator->BuildLocator();

	// Create two sets of points to handle iteration

	int step = 1;
	if (this->Source->GetNumberOfPoints() > this->MaximumNumberOfLandmarks)
	{
		step = this->Source->GetNumberOfPoints() / this->MaximumNumberOfLandmarks;
		vtkDebugMacro(<< "Landmarks step is now : " << step);
	}

	vtkIdType nb_points = this->Source->GetNumberOfPoints() / step;

	// Allocate some points.
	// - closestp is used so that the internal state of LandmarkTransform remains
	//   correct whenever the iteration process is stopped (hence its source
	//   and landmark points might be used in a vtkThinPlateSplineTransform).
	// - points2 could have been avoided, but do not ask me why 
	//   InternalTransformPoint is not working correctly on my computer when
	//   in and out are the same pointer.

	vtkPoints *points1 = vtkPoints::New();
	points1->SetNumberOfPoints(nb_points);

	vtkPoints *closestp = vtkPoints::New();
	closestp->SetNumberOfPoints(nb_points);

	vtkPoints *points2 = vtkPoints::New();
	points2->SetNumberOfPoints(nb_points);

	// Fill with initial positions (sample dataset using step)

	vtkTransform *accumulate = vtkTransform::New();
	accumulate->PostMultiply();

	vtkIdType i;
	int j;
	double p1[3], p2[3];

	
	for (i = 0, j = 0; i < nb_points; i++, j += step)
	{
		points1->SetPoint(i, this->Source->GetPoint(j));
	}
	

	// Go

	vtkIdType cell_id;
	int sub_id;
	double dist2, totaldist = 0;
	double outPoint[3];

	vtkPoints *temp, *a = points1, *b = points2;

	this->NumberOfIterations = 0;

	do 
	{
		// Fill points with the closest points to each vertex in input

		for(i = 0; i < nb_points; i++)
		{
			this->Locator->FindClosestPoint(a->GetPoint(i),
				outPoint,
				cell_id,
				sub_id,
				dist2);
			closestp->SetPoint(i, outPoint);
		}

		// Build the landmark transform

		this->LandmarkTransform->SetSourceLandmarks(a);
		this->LandmarkTransform->SetTargetLandmarks(closestp);
		this->LandmarkTransform->Update();

		// Concatenate (can't use this->Concatenate directly)

		accumulate->Concatenate(this->LandmarkTransform->GetMatrix());

		this->NumberOfIterations++;
		vtkDebugMacro(<< "Iteration: " << this->NumberOfIterations);
		if (this->NumberOfIterations >= this->MaximumNumberOfIterations) 
		{
			break;
		}

		// Move mesh and compute mean distance if needed


		totaldist = 0.0;


		for(i = 0; i < nb_points; i++)
		{
			a->GetPoint(i, p1);
			this->LandmarkTransform->InternalTransformPoint(p1, p2);
			b->SetPoint(i, p2);
				
			double absDist = std::abs(p1[0] - p2[0]) + std::abs(p1[1] - p2[1]) + std::abs(p1[2] - p2[2]);
			totaldist += std::log(absDist + 1.0);
			
		}

		this->MeanDistance = totaldist / (double)nb_points;

		vtkDebugMacro("Mean distance: " << this->MeanDistance);
		if (this->MeanDistance <= this->MaximumMeanDistance)
		{
			break;
		}

		temp = a;
		a = b;
		b = temp;

	} 
	while (1);

	// Now recover accumulated result

	this->Matrix->DeepCopy(accumulate->GetMatrix());

	accumulate->Delete();
	points1->Delete();
	closestp->Delete();
	points2->Delete();

}