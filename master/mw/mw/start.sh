#!/bin/bash

rm -rf /proj/i4mw/sidoneum/gruppe1/mw/zab*


ssh faui07f {`killall -9 java & rm -rf /proj/i4mw/sidoneum/gruppe1/mw/zab* && java -cp /proj/i4mw/pub/aufgabe5/jars/zab-dev.jar:/proj/i4mw/pub/aufgabe5/jars/log4j-1.2.15.jar:/proj/i4mw/sidoneum/gruppe1/mw/bin/ mw.zookeeper.MWZooKeeperServer 1 11234 faui07f:11234 faui06e:11235 faui06b:11236`} &
ssh faui06e {`killall -9 java & rm -rf /proj/i4mw/sidoneum/gruppe1/mw/zab* && java -cp /proj/i4mw/pub/aufgabe5/jars/zab-dev.jar:/proj/i4mw/pub/aufgabe5/jars/log4j-1.2.15.jar:/proj/i4mw/sidoneum/gruppe1/mw/bin/ mw.zookeeper.MWZooKeeperServer 2 11235 faui07f:11234 faui06e:11235 faui06b:11236`} &
ssh faui06b {`killall -9 java & rm -rf /proj/i4mw/sidoneum/gruppe1/mw/zab* && java -cp /proj/i4mw/pub/aufgabe5/jars/zab-dev.jar:/proj/i4mw/pub/aufgabe5/jars/log4j-1.2.15.jar:/proj/i4mw/sidoneum/gruppe1/mw/bin/ mw.zookeeper.MWZooKeeperServer 3 11236 faui07f:11234 faui06e:11235 faui06b:11236`} &
