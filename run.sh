#!/bin/bash
#$ -l gpu=1
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -l h_rt=100:0:0
#$ -wd /data/home/eey340/WORKPLACE/ALM4FSL/experiments/python
#$ -j y
#$ -N ALM4FSL
#$ -o /data/home/eey340/WORKPLACE/ALM4FSL/experiments/LOGS/new_esc_fullsize_sweep.log
#$ -m beas
###£#$ -l cluster=andrena


# Import environments
module load cudnn/8.1.1-cuda11
source ./alm/bin/activate


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
# if EXPERIMENT is esc50_fullsize_evaluation, fsdkaggle18k_fullsize_evaluation, then the following fewshot setting won't work
DATABASE=fsd_fs
N_TASK=50
N_CLASS=15
N_QUERIES=5

EXPERIMENT=esc50_fullsize_evaluation  # [sl_fewshot, ml_fewshot, esc50_fullsize_evaluation, fsdkaggle18k_fullsize_evaluation]
N_SUPPORTS=1,3,5,10,15,20
FINE_TUNE=False
N_EPOCHS=10,15,20,25,30,35,40,45,50
ADAPTER=xattention # match

gpu-usage

python3 ${EXPERIMENT}.py \
storage_pth=${STORAGE_DIR} \
database=${DATABASE} \
fewshot.n_task=${N_TASK} \
fewshot.n_class=${N_CLASS} \
fewshot.n_queries=${N_QUERIES} \
fewshot.fine_tune=True \
fewshot.adapter=${ADAPTER} \
fewshot.n_supports=${N_SUPPORTS} \
fewshot.train_epochs=${N_EPOCHS}
