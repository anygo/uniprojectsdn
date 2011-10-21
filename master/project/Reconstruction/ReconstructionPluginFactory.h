#ifndef ReconstructionPLUGINFACTORY_H__
#define ReconstructionPLUGINFACTORY_H__

#include <QObject>
#include "ritkPluginFactories.h"

class ReconstructionPluginFactory : public QObject, public ritk::ApplicationPluginFactory
{
	Q_OBJECT
	Q_INTERFACES(ritk::ApplicationPluginFactory)

public:
	/// Constructor
	ReconstructionPluginFactory();
	/// Destructor
	~ReconstructionPluginFactory();

	ritk::ApplicationPlugin* CreateInstance();

	QString GetName();

	QString GetDescription();
};

#endif // ReconstructionPLUGINFACTORY_H__
