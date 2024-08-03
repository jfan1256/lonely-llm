# Run this in command line for multi-gpu training on Argon HPC Cluster
# qsub -pe smp 128 -q MANSCI-GPU -l gpu=true -l gpu_a40=true job.sh

#!/bin/bash

# Send Email at Beginning/End/Suspension of Job
#$ -m bes

# Email address to send to
#$ -M weiguo-fan@uiowa.edu

# Activate Python Environment
cd /old_Users/weigfan/mentalbertenv/bin/
source activate

# CC into exec_pretrain
cd /old_Users/weigfan/mental-health-research/lonely-llm/exec_pretrain

# Execute pretrain
python -m torch.distributed.run --nproc_per_node=4 pretrain.py
