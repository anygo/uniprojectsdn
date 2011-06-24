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
	return ("Fast GPU Stitching based on RBC-ICP");
}


Q_EXPORT_PLUGIN2(FastStitchingPlugin, FastStitchingPluginFactory)

