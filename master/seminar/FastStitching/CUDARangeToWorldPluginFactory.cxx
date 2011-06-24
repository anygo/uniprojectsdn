#include "CUDARangeToWorldPluginFactory.h"
#include "CUDARangeToWorldPlugin.h"

#include <QtPlugin>

CUDARangeToWorldPluginFactory::CUDARangeToWorldPluginFactory()
{
}

CUDARangeToWorldPluginFactory::~CUDARangeToWorldPluginFactory()
{
}

ritk::ApplicationPlugin*
CUDARangeToWorldPluginFactory::CreateInstance()
{
	return new CUDARangeToWorldPlugin();
}

QString 
CUDARangeToWorldPluginFactory::GetName()
{
	return ("CUDARangeToWorld");
}

QString 
CUDARangeToWorldPluginFactory::GetDescription()
{
	return ("Calculates world coords in CUDA");
}


Q_EXPORT_PLUGIN2(CUDARangeToWorldPlugin, CUDARangeToWorldPluginFactory)

