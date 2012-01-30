#include "ImprovedStitchingPluginFactory.h"
#include "ImprovedStitchingPlugin.h"

#include <QtPlugin>

ImprovedStitchingPluginFactory::ImprovedStitchingPluginFactory()
{
}

ImprovedStitchingPluginFactory::~ImprovedStitchingPluginFactory()
{
}

ritk::ApplicationPlugin*
ImprovedStitchingPluginFactory::CreateInstance()
{
	return new ImprovedStitchingPlugin();
}

QString 
ImprovedStitchingPluginFactory::GetName()
{
	return ("ImprovedStitching");
}

QString 
ImprovedStitchingPluginFactory::GetDescription()
{
	return ("Real-time environment reconstruction");
}


Q_EXPORT_PLUGIN2(ImprovedStitchingPlugin, ImprovedStitchingPluginFactory)

