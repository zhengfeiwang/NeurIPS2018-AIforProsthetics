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
SHM_SIZE=50G
MANAGER_PORT=18076
REDIS_PORT=16379
TB_PORT=6006
WORKERS_NUM=40
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
# print shared memory size in ray
echo "shared memory size : $SHM_SIZE"
# print ray manager port
echo "ray manager port : $MANAGER_PORT"
# print ray redis port
echo "ray redis port : $REDIS_PORT"
# print ray workers number per-node
echo "ray workers number per-node : $WORKERS_NUM"
argument_info="# [That's all of the argument]"
echo -e "\033[36m${argument_info}\033[0m"

# parallel deploy docker image
if [ $# != 1 ] ; then
  deploy_info="# [parallel deploy docker image]"
  echo -e "\033[35m${deploy_info}\033[0m"
  deploy_cmd="docker pull $DOCKER_URI"
  parallel-ssh -h $HEAD_IP_LIST_FILE -P -t $TIMEOUT -l $RUN_USER $deploy_cmd
  if [ $? -ne 0 ]; then 
    error_info="# [parallel deploy docker image to head fail, exit!]"
    echo -e "\033[31m${error_info}\033[0m"
    exit 1
  fi
  parallel-ssh -h $NODE_IP_LIST_FILE -P -t $TIMEOUT -l $RUN_USER $deploy_cmd
  if [ $? -ne 0 ]; then 
    error_info="# [parallel deploy docker image to node fail, exit!]"
    echo -e "\033[31m${error_info}\033[0m"
    exit 1
  fi
  deploy_info="# [parallel deploy docker image done]"
  echo -e "\033[36m${deploy_info}\033[0m"
else
  deploy_info="# [deploy docker image]"
  echo -e "\033[35m${deploy_info}\033[0m"
  docker pull $DOCKER_URI
  if [ $? -ne 0 ]; then 
    error_info="# [deploy docker image to local fail, exit!]"
    echo -e "\033[31m${error_info}\033[0m"
    exit 1
  fi
  deploy_info="# [deploy docker image done]"
  echo -e "\033[36m${deploy_info}\033[0m"
fi

# parallel launch docker
if [ $# != 1 ] ; then
  launch_info="# [parallel launch docker]"
  echo -e "\033[35m${launch_info}\033[0m"
  launch_head_cmd="docker run -d -v /root/logs/:/root/logs/ -p$TB_PORT:$TB_PORT --shm-size=$SHM_SIZE --net=host --runtime=nvidia $DOCKER_URI /ray/start_head.sh $MANAGER_PORT $REDIS_PORT $WORKERS_NUM $TB_PORT"
  parallel-ssh -h $HEAD_IP_LIST_FILE -P -t $TIMEOUT -l $RUN_USER $launch_head_cmd
  if [ $? -ne 0 ]; then 
    error_info="# [parallel launch head fail, exit!]"
    echo -e "\033[31m${error_info}\033[0m"
    exit 1
  fi
  head_ip_list=($(awk '{print $0}' $HEAD_IP_LIST_FILE))
  head_ip=${head_ip_list[0]}
  launch_node_cmd="docker run -d -v /root/logs/:/root/logs/ --shm-size=$SHM_SIZE --net=host --runtime=nvidia $DOCKER_URI /ray/start_node.sh $MANAGER_PORT $head_ip $REDIS_PORT $WORKERS_NUM"
  parallel-ssh -h $NODE_IP_LIST_FILE -P -t $TIMEOUT -l $RUN_USER $launch_node_cmd
  if [ $? -ne 0 ]; then 
    error_info="# [parallel launch node fail, exit!]"
    echo -e "\033[31m${error_info}\033[0m"
    exit 1
  fi
  launch_info="# [parallel launch docker done]"
  echo -e "\033[36m${launch_info}\033[0m"
else
  launch_info="# [launch docker]"
  echo -e "\033[35m${launch_info}\033[0m"
  docker run -d --shm-size=$SHM_SIZE --net=host --runtime=nvidia $DOCKER_URI /ray/start_head.sh $MANAGER_PORT $REDIS_PORT $WORKERS_NUM $TB_PORT
  if [ $? -ne 0 ]; then 
    error_info="# [launch head fail, exit!]"
    echo -e "\033[31m${error_info}\033[0m"
    exit 1
  fi
  launch_info="# [launch docker done]"
  echo -e "\033[36m${launch_info}\033[0m"
fi
