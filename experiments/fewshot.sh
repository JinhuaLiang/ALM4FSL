#!/bin/bash
#$ -l gpu=1
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -l h_rt=10:0:0
#$ -wd /data/home/eey340/WORKPLACE/ALM4FSL/experiments/python
#$ -j y
#$ -N ALM4FSL
#$ -o /data/home/eey340/WORKPLACE/ALM4FSL/experiments/LOGS/fsdkaggle_sweep.log
#$ -m beas
#$ -l cluster=andrena

module load cudnn/8.1.1-cuda11
source ../../alm/bin/activate

gpu-usage
# for N_SUPPORTS in 1 3 5 10
# do
#     for EPOCH in 10 15 20 25 30
#     do
#         python3 fewshot.py --dataset esc50 --n_supports ${N_SUPPORTS} --fine_tune --train_epochs ${EPOCH} --train_lr 0.001
#     done
# done

for N_SUPPORTS in 1 3 5 10
do
    python3 fewshot.py --dataset fsdkaggle18k --n_class 10 --n_supports ${N_SUPPORTS} --n_queries 60
done

for N_SUPPORTS in 1 3 5 10
do
    for EPOCH in 10 15 20 25 30
    do
        python3 fewshot.py --dataset fsdkaggle18k --n_class 10 --n_supports ${N_SUPPORTS} --n_queries 60 --fine_tune --train_epochs ${EPOCH} --train_lr 0.001
    done
done

# Generated by Job Script Builder on 2022-01-20
# For assistance, please email its-research-support@qmul.ac.uk
#=================================COMMON CMD===================================#
## testify the code by using a short queue (shorter than 1 hour) on dn150
# -l h_rt=1:0:0    # 1 hour runtime
# -l gpu_type=kepler

## select 8 cores per GPU, and 7.5GB per core, e.g.:
# -pe smp 8        # 8 cores
# -l h_vmem=7.5G / 11G  # 7.5 * 8 = 60G total RAM
# -l gpu=1         # request 1 GPU

# -l exclusive     # request exclusive access
