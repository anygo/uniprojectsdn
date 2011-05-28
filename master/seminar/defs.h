#ifndef defs_H__
#define	defs_H__

#include <QListWidget.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <RImageActorPipeline.h>

typedef struct Point6D
{
	double x, y, z;
	double r, g, b;
}
Point6D;

class HistoryListItem : public QListWidgetItem
{
public:
	vtkSmartPointer<ritk::RImageActorPipeline>	m_actor;
	vtkSmartPointer<vtkMatrix4x4>				m_transform;
};

enum ICP_METRIC
{
	LOG_ABSOLUTE_DISTANCE,
	ABSOLUTE_DISTANCE,
	SQUARED_DISTANCE
};


#endif // defs_H__
