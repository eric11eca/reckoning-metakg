#!/bin/bash

MY_IMAGE="eric11eca/meta_kg_7"

arg_job_prefix="meta-kg"
arg_job_suffix="clutrr2"
arg_job_name="$arg_job_prefix-$arg_job_suffix"

command=$1

if [ "$command" == "run" ]; then
	echo "Job [$arg_job_name]"

	runai submit $arg_job_name \
		-i $MY_IMAGE \
		--gpu 1 \
		--node-type G10 \
		# --pvc runai-nlp-zechen-nlpdata1:/nlpdata1 \
		# --large-shm
	exit 0
fi

if [ "$command" == "run_bash" ]; then
	echo "Job [$arg_job_name]"

	runai submit $arg_job_name \
		-i $MY_IMAGE \
		--gpu 4 \
		--pvc runai-nlp-zechen-nlpdata1:/nlpdata1 \
		--large-shm \
		--interactive \
		--attach \
		--node-type G10
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
