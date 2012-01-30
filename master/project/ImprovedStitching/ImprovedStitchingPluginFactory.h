#ifndef IMPROVEDSTITCHINGPLUGINFACTORY_H__
#define IMPROVEDSTITCHINGPLUGINFACTORY_H__

#include <QObject>
#include "ritkPluginFactories.h"

class ImprovedStitchingPluginFactory : public QObject, public ritk::ApplicationPluginFactory
{
	Q_OBJECT
	Q_INTERFACES(ritk::ApplicationPluginFactory)

public:
	/// Constructor
	ImprovedStitchingPluginFactory();
	/// Destructor
	~ImprovedStitchingPluginFactory();

	ritk::ApplicationPlugin* CreateInstance();

	QString GetName();

	QString GetDescription();
};

#endif // IMPROVEDSTITCHINGPLUGINFACTORY_H__
