#!/bin/bash

JAVA="/usr/bin/java"
CLASSPATH="bin:jaxr-*"

wsdlURL=http://`hostname --fqdn`:42042/MWPathService?wsdl
echo $wsdlURL

cd /opt/mwcc
$JAVA -cp $CLASSPATH mw.cache.MWCache &
$JAVA -cp $CLASSPATH mw.path.MWPathServer &
$JAVA -cp $CLASSPATH mw.MWRegistryAccess REGISTER gruppe1 MWPathService $wsdlURL
