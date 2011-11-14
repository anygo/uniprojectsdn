#ifndef RECONSTRUCTIONPLUGINFACTORY_H__
#define RECONSTRUCTIONPLUGINFACTORY_H__

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

#endif // RECONSTRUCTIONPLUGINFACTORY_H__
