#!/bin/bash

CLUSTER_USER=zechen
CLUSTER_USER_ID=254670
CLUSTER_GROUP_NAME=NLP-StaffU
CLUSTER_GROUP_ID=11131

MY_IMAGE="eric11eca/meta_kg_step2_5"

arg_job_prefix="meta-kg"
arg_job_suffix="proof5-s2"
arg_job_name="$arg_job_prefix-$arg_job_suffix"

command=$1

if [ "$command" == "run" ]; then
	echo "Job [$arg_job_name]"

	runai submit $arg_job_name \
		-i $MY_IMAGE \
		--gpu 1 \
		--node-type G10
		
		# -e CLUSTER_USER=$CLUSTER_USER \
		# -e CLUSTER_USER_ID=$CLUSTER_USER_ID \
		# -e CLUSTER_GROUP_NAME=$CLUSTER_GROUP_NAME \
		# -e CLUSTER_GROUP_ID=$CLUSTER_GROUP_ID \
		
		# --pvc runai-nlp-zechen-nlpdata1:/nlpdata1 \
		# --large-shm \
		# --service-type=nodeport \
		# --port 30012:22
	exit 0
fi

if [ "$command" == "log" ]; then
	runai logs $arg_job_name -f
	exit 0
fi

if [ "$command" == "stat" ]; then
	runai describe job $arg_job_name 
	exit 0
fi

if [ "$command" == "del" ]; then
	runai delete job $arg_job_name
	exit 0
fi



if [ $? -eq 0 ]; then
	runai describe job $arg_job_name
fi
