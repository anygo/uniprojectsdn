#!/bin/sh

LOCAL_CLASSPATH=.

make
rmiregistry 10412 &
java -cp $LOCAL_CLASSPATH -Djava.security.policy=rmi.policy vs.boardserver

