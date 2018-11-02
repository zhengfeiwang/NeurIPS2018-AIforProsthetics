#!/bin/bash

if [ $# != 2 ] ; then 
  echo "USAGE: $0 docker-image-name docker-image-tag" 
  exit 1
fi 

NAME=$1
TAG=$2

docker build --no-cache -t login.local.inspir.ai:5000/$NAME:$TAG .
