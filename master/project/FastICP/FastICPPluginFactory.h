#ifndef FASTICPPLUGINFACTORY_H__
#define FASTICPPLUGINFACTORY_H__

#include <QObject>
#include "ritkPluginFactories.h"

class FastICPPluginFactory : public QObject, public ritk::ApplicationPluginFactory
{
	Q_OBJECT
	Q_INTERFACES(ritk::ApplicationPluginFactory)

public:
	/// Constructor
	FastICPPluginFactory();
	/// Destructor
	~FastICPPluginFactory();

	ritk::ApplicationPlugin* CreateInstance();

	QString GetName();

	QString GetDescription();
};

#endif // FASTICPPLUGINFACTORY_H__
