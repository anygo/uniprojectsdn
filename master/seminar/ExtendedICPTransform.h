#ifndef ExtendedICPTransform_H__
#define	ExtendedICPTransform_H__

#include <vtkIterativeClosestPointTransform.h>

class ExtendedICPTransform : public vtkIterativeClosestPointTransform
{
public:
	static ExtendedICPTransform *New();

protected:
	void InternalUpdate();
};

#endif // ExtendedICPTransform_H__