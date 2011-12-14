#include "FastICPPluginFactory.h"
#include "FastICPPlugin.h"

#include <QtPlugin>

FastICPPluginFactory::FastICPPluginFactory()
{
}

FastICPPluginFactory::~FastICPPluginFactory()
{
}

ritk::ApplicationPlugin*
FastICPPluginFactory::CreateInstance()
{
	return new FastICPPlugin();
}

QString 
FastICPPluginFactory::GetName()
{
	return ("FastICP");
}

QString 
FastICPPluginFactory::GetDescription()
{
	return ("Real-time environment reconstruction");
}


Q_EXPORT_PLUGIN2(FastICPPlugin, FastICPPluginFactory)

