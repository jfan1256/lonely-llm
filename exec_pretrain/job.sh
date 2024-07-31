#!/bin/bash

#####Set Scheduler Configuration Directives#####
# Name the job:
$ -N Mental Health-Data-Analysis

# Send e-mail at beginning/end/suspension of job
$ -m bes

# E-mail address to send to
$ -M weiguo-fan@uiowa.edu

# Start script in current working directory
$ -cwd

#####Resource Selection Directives#####
# Select the queue to run in
$ -q MANSCI-GPU -l gpu=true -l gpu_a40=true

# Request four cores
$ -pe smp 64

#####Execute Pretrain#####
# Cd into python env
cd /old_Users/weigfan/mentalbertenv/bin/

# Activate python env
source activate

# Cd into exec_pretrain
cd /old_Users/weigfan/mental-health-research/lonely-llm/exec_pretrain

# Unset gpu (allows it to recognize all gpus)
unset CUDA_VISIBLE_DEVICES

# Set up executable
chmod +x pretrain.sh
./pretrain.sh