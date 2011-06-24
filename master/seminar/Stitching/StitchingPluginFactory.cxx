#include "StitchingPluginFactory.h"
#include "StitchingPlugin.h"

#include <QtPlugin>

StitchingPluginFactory::StitchingPluginFactory()
{
}

StitchingPluginFactory::~StitchingPluginFactory()
{
}

ritk::ApplicationPlugin*
StitchingPluginFactory::CreateInstance()
{
	return new StitchingPlugin();
}

QString 
StitchingPluginFactory::GetName()
{
	return ("Stitching");
}

QString 
StitchingPluginFactory::GetDescription()
{
	return ("Realtime Stitching of 3D-Pointclouds");
}


Q_EXPORT_PLUGIN2(StitchingPlugin, StitchingPluginFactory)

