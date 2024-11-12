#!/bin/bash

all_args=("$@")

config=$1
project=${2:-test}

project_path=${3:-None}

wandb_offline=${4:-"none"} # set to "offline" to disable online wandb sync


config_name=`basename $config`
config_name="${config_name%.*}"
exp_name="${config_name}"
if [ ! -z $post_fix ]; then
    exp_name+="_$post_fix"
fi


echo exp name: $exp_name
echo extra commands ${extra[@]}

if [ $project_path == "None" ]; then
    checkpoint_path=checkpoints/ckpt_${project}/checkpoints_${exp_name}
else
    checkpoint_path=$project_path
fi

mkdir -p $checkpoint_path



if [ "$wandb_offline" == "offline" ]; then
    wandb offine
fi


config_command=""
config_command+=`python3 my_scripts/config_to_command.py --config $config ${extra[@]}`
config_command+=" --save-dir ${checkpoint_path} "
config_command+=" --log-file ${checkpoint_path}/${exp_name}.log "
config_command+=" --wandb-project $project "

echo $config_command

eval $config_command

# sleep 600
