#!/bin/bash
#$ -l gpu=1
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -l h_rt=100:0:0
#$ -wd /data/home/eey340/WORKPLACE/ALM4FSL/experiments/python
#$ -j y
#$ -N ALM4FSL
#$ -o /data/home/eey340/WORKPLACE/ALM4FSL/experiments/LOGS/esc50_fullsize_sweep.log
#$ -m beas
#$ -l cluster=andrena


# Import environments
module load cudnn/8.1.1-cuda11
source ../../alm/bin/activate


# Set variables (The below are receommended settings)
# |            	| esc50      	| fsdkaggle18k 	| fsd_fs     	|
# |------------	|------------	|--------------	|------------	|
# | EXPERIMENT 	| sl_fewshot 	| sl_fewshot   	| ml_fewshot 	|
# | DATABASE   	| esc50      	| fsdkaggle18k 	| fsd_fs     	|
# | N_TASK     	| 100        	| 100          	| 50         	|
# | N_CLASS    	| 15         	| 10           	| 15         	|
# | N_SUPPORTS 	| 10         	| 20           	| 5          	|
# | N_QUERIES  	| 30         	| 50           	| 5          	|
# | N_EPOCHS   	| 20         	| 20           	| 20         	|
STORAGE_DIR=/data/EECS-MachineListeningLab/ # `dataset dir` and `pretrained weight pth` should be under this dir
EXPERIMENT=ml_fewshot  # [sl_fewshot, ml_fewshot]
DATABASE=fsd_fs

N_TASK=50
N_CLASS=15
# N_SUPPORTS=1 
N_QUERIES=5
# N_EPOCHS=20

FINE_TUNE=True
ADAPTER='xattention'

gpu-usage

# for N_SUPPORTS in 1 3 5 10 15 20
# do
#     for N_EPOCHS in 10 15 20 25 30 35 40 45 50
#     do
#         python3 ${EXPERIMENT}.py storage_pth=${STORAGE_DIR} database=${DATABASE} fewshot.n_task=${N_TASK} fewshot.n_class=${N_CLASS} fewshot.n_supports=${N_SUPPORTS} fewshot.n_queries=${N_QUERIES} fewshot.fine_tune=${FINE_TUNE} fewshot.adapter=${ADAPTER} fewshot.train_epochs=${N_EPOCHS}
#     done
# done

# for N_SUPPORTS in 1 3 5 10 15 20
# do
#     for N_EPOCHS in 10 15 20 25 30 35 40 45 50
#     do
#         python3 esc50_fullsize_evaluation.py storage_pth=${STORAGE_DIR} fewshot.fine_tune=True fewshot.n_supports=${N_SUPPORTS} fewshot.adapter=${ADAPTER} fewshot.train_epochs=${N_EPOCHS}
#     done
# done


for N_SUPPORTS in 1 3 5 10 15 20
do
    python3 esc50_fullsize_evaluation.py storage_pth=${STORAGE_DIR} fewshot.fine_tune=False fewshot.n_supports=${N_SUPPORTS}
done
