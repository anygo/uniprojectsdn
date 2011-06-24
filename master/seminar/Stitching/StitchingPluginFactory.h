#ifndef StitchingPLUGINFACTORY_H__
#define StitchingPLUGINFACTORY_H__

#include <QObject>
#include "PluginFactories.h"

class StitchingPluginFactory : public QObject, public ritk::ApplicationPluginFactory
{
	Q_OBJECT
	Q_INTERFACES(ritk::ApplicationPluginFactory)

public:
	/// Constructor
	StitchingPluginFactory();
	/// Destructor
	~StitchingPluginFactory();

	ritk::ApplicationPlugin* CreateInstance();

	QString GetName();

	QString GetDescription();
};

#endif // StitchingPLUGINFACTORY_H__
