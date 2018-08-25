#!/bin/bash

if [ $# != 1 ] ; then 
  if [ $# != 3 ] ; then 
    echo "USAGE FOR CLUSTER: $0 docker.uri head_ip.list node_ip.list"
    echo "   OR FOR LOCAL  : $0 docker.uri"
    exit 1
  fi
fi 

DOCKER_URI_FILE=$1
HEAD_IP_LIST_FILE=$2
NODE_IP_LIST_FILE=$3

RUN_USER=cluster
TIMEOUT=600

argument_info="# [The arguments of this time]"
echo -e "\033[35m${argument_info}\033[0m"
# print docker image uri
docker=($(awk '{print $0}' $DOCKER_URI_FILE))
DOCKER_URI=${docker[0]}
echo "docker image : $DOCKER_URI"
if [ $# != 1 ] ; then
  # print head ips
  iplist=($(awk '{print $0}' $HEAD_IP_LIST_FILE))
  for((i=0;i<${#iplist[@]};i++));
  do
    echo "head ip [$i] : ${iplist[$i]}"
  done
  # print node ips
  iplist=($(awk '{print $0}' $NODE_IP_LIST_FILE))
  for((i=0;i<${#iplist[@]};i++));
  do
    echo "node ip [$i] : ${iplist[$i]}"
  done
  # print run user
  echo "run user : $RUN_USER"
fi
argument_info="# [That's all of the argument]"
echo -e "\033[36m${argument_info}\033[0m"

# parallel shutdown docker
if [ $# != 1 ] ; then
  shutdown_info="# [parallel shutdown docker]"
  echo -e "\033[35m${shutdown_info}\033[0m"
  shutdown_cmd="docker kill \$(docker ps | grep $DOCKER_URI | awk '{print \$1}')"
  parallel-ssh -h $NODE_IP_LIST_FILE -P -t $TIMEOUT -l $RUN_USER $shutdown_cmd
  parallel-ssh -h $HEAD_IP_LIST_FILE -P -t $TIMEOUT -l $RUN_USER $shutdown_cmd
  shutdown_info="# [parallel shutdown docker image done]"
  echo -e "\033[36m${shutdown_info}\033[0m"
else
  shutdown_info="# [shutdown docker]"
  echo -e "\033[35m${shutdown_info}\033[0m"
  docker kill $(docker ps | grep $DOCKER_URI | awk '{print $1}')
  if [ $? -ne 0 ]; then 
    error_info="# [shutdown docker fail, exit!]"
    echo -e "\033[31m${error_info}\033[0m"
    exit 1
  fi
  shutdown_info="# [shutdown docker done]"
  echo -e "\033[36m${shutdown_info}\033[0m"
fi
