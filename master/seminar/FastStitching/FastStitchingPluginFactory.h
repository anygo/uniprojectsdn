#ifndef FastStitchingPLUGINFACTORY_H__
#define FastStitchingPLUGINFACTORY_H__

#include <QObject>
#include "PluginFactories.h"

class FastStitchingPluginFactory : public QObject, public ritk::ApplicationPluginFactory
{
	Q_OBJECT
    Q_INTERFACES(ritk::ApplicationPluginFactory)

public:
	/// Constructor
	FastStitchingPluginFactory();
	/// Destructor
	~FastStitchingPluginFactory();

  ritk::ApplicationPlugin* CreateInstance();

	QString GetName();

	QString GetDescription();
};

#endif // FastStitchingPLUGINFACTORY_H__
