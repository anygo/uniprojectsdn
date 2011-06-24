#ifndef CUDARangeToWorldPLUGINFACTORY_H__
#define CUDARangeToWorldPLUGINFACTORY_H__

#include <QObject>
#include "PluginFactories.h"

class CUDARangeToWorldPluginFactory : public QObject, public ritk::ApplicationPluginFactory
{
	Q_OBJECT
    Q_INTERFACES(ritk::ApplicationPluginFactory)

public:
	/// Constructor
	CUDARangeToWorldPluginFactory();
	/// Destructor
	~CUDARangeToWorldPluginFactory();

  ritk::ApplicationPlugin* CreateInstance();

	QString GetName();

	QString GetDescription();
};

#endif // CUDARangeToWorldPLUGINFACTORY_H__
