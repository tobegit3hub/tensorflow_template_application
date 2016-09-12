#!/bin/bash 

set -x 
set -e

mvn clean install

mvn compile

mvn exec:java -Dexec.mainClass="com.tobe.Hello"
