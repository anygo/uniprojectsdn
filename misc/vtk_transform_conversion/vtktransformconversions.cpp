#include "vtkSmartPointer.h"
#include "vtkTransform.h"
#include "vtkMatrix4x4.h"

vtkSmartPointer<vtkTransform> getVTKTransformFromParams(double transX, double transY, double transZ, double rotX, double rotY, double rotZ)
{
  vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();

  // initialize with identity
  transform->Identity();
  transform->PreMultiply();

  // set translation
  transform->Translate(transX, transY, transZ);
  transform->Update();

  // set rotation (in degrees)
  transform->RotateZ(rotZ);
  transform->RotateX(rotX);
  transform->RotateY(rotY); 
  transform->Update();

  return transform;
}

vtkSmartPointer<vtkTransform> getVTKTransformFromParams(double trans[3], double rot[3])
{
  return getVTKTransformFromParams(trans[0], trans[1], trans[2], rot[0], rot[1], rot[2]);
}

void getParamsFromVTKTransform(vtkTransform* transform, double& transX, double& transY, double& transZ, double& rotX, double& rotY, double& rotZ)
{
  // get translation vector
  transX = transform->GetPosition()[0];
  transY = transform->GetPosition()[1];
  transZ = transform->GetPosition()[2];

  // get rotation angles (in degrees)
  rotX = transform->GetOrientation()[0];
  rotY = transform->GetOrientation()[1];
  rotZ = transform->GetOrientation()[2];
}

void getParamsFromVTKTransform(vtkTransform* transform, double trans[3], double rot[3])
{
  getParamsFromVTKTransform(transform, trans[0], trans[1], trans[2], rot[0], rot[1], rot[2]);
}


// two different ways to use the functions
void test()
{
  // compute transform (angles in degrees, as in VTK)
  vtkSmartPointer<vtkTransform> transform1 = getVTKTransformFromParams(1.0, 2.0, 3.0, 10.0, 20.0, 30.0);

  // get params from transform
  double tx, ty, tz, rx, ry, rz;
  getParamsFromVTKTransform(transform1, tx, ty, tz, rx, ry, rz);

  std::cout << "Matrix:" << std::endl;
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      std::cout << transform1->GetMatrix()->Element[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Translation: " << tx << " " << ty << " " << tz << std::endl;
  std::cout << "Rotation: " << rx << " " << ry << " " << rz << std::endl;

  std::cout << std::endl;

  // compute transform (angles in degrees, as in VTK), this time using arrays (for convenience)
  double transIn[3] = {4.0, 5.0, 6.0};
  double rotIn[3] = {40.0, 50.0, 60.0};
  vtkSmartPointer<vtkTransform> transform2 = getVTKTransformFromParams(transIn, rotIn);

  // get params from transform
  double transOut[3], rotOut[3];
  getParamsFromVTKTransform(transform2, transOut, rotOut);

  std::cout << "Matrix:" << std::endl;
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      std::cout << transform2->GetMatrix()->Element[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Translation: " << transOut[0] << " " << transOut[1] << " " << transOut[2] << std::endl;
  std::cout << "Rotation: " << rotOut[0] << " " << rotOut[1] << " " << rotOut[2] << std::endl;
}