#include "FastStitchingPluginFactory.h"
#include "FastStitchingPlugin.h"

#include <QtPlugin>

FastStitchingPluginFactory::FastStitchingPluginFactory()
{
}

FastStitchingPluginFactory::~FastStitchingPluginFactory()
{
}

ritk::ApplicationPlugin*
FastStitchingPluginFactory::CreateInstance()
{
	return new FastStitchingPlugin();
}

QString 
FastStitchingPluginFactory::GetName()
{
	return ("FastStitching");
}

QString 
FastStitchingPluginFactory::GetDescription()
{
	return ("Realtime FastStitching of 3D-Pointclouds");
}


Q_EXPORT_PLUGIN2(FastStitchingPlugin, FastStitchingPluginFactory)

