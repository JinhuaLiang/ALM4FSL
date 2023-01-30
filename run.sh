#!/bin/bash
#$ -l gpu=1
#$ -pe smp 8
#$ -l h_vmem=11G
#$ -l h_rt=100:0:0
#$ -wd /data/home/eey340/WORKPLACE/ALM4FSL/experiments/python
#$ -j y
#$ -N ALM4FSL
#$ -o /data/home/eey340/WORKPLACE/ALM4FSL/experiments/LOGS/test.log
#$ -m beas
###£#$ -l cluster=andrena


# Import environments
module load cudnn/8.1.1-cuda11
source ../../alm/bin/activate
# gpu-usage


STORAGE_DIR=/data/EECS-MachineListeningLab/ # `dataset dir` and `pretrained weight pth` should be under this dir
# Set variables (The below are receommended settings)
# |            	| esc50      	| fsdkaggle18k 	| fsd_fs     	|
# |------------	|------------	|--------------	|------------	|
# | EXPERIMENT 	| sl_fewshot 	| sl_fewshot   	| ml_fewshot 	|
# | DATABASE   	| esc50      	| fsdkaggle18k 	| fsd_fs     	|
# | N_TASK     	| 100        	| 100          	| 50         	|
# | N_CLASS    	| 15         	| 10           	| 15         	|
# | N_QUERIES  	| 30         	| 50           	| 5          	|
# | N_SUPPORTS 	| 10         	| 20           	| 5          	|
# | N_EPOCHS   	| 20         	| 20           	| 20         	|
# EXPERIMENT=ml_fewshot  # [sl_fewshot, ml_fewshot]
# DATABASE=fsd_fs
# N_TASK=50
# N_CLASS=15
# N_QUERIES=5

# N_SUPPORTS=1,3,5,10,15,20
# FINE_TUNE=True
# N_EPOCHS=0,1,3,5,10,15,20,25,30
# ADAPTER=xattention # match

# python3 ${EXPERIMENT}.py \
# storage_pth=${STORAGE_DIR} \
# database=${DATABASE} \
# fewshot.n_task=${N_TASK} \
# fewshot.n_class=${N_CLASS} \
# fewshot.n_queries=${N_QUERIES} \
# fewshot.fine_tune=True \
# fewshot.adapter=${ADAPTER} \
# fewshot.n_supports=${N_SUPPORTS} \
# fewshot.train_epochs=${N_EPOCHS}


### fullsize experiment
# EXPERIMENT=esc50_fullsize_evaluation  # [esc50_fullsize_evaluation, fsdkaggle18k_fullsize_evaluation]
# N_SUPPORTS=32  #1,3,5,10,15,20,25,32
# FINE_TUNE=True
# N_EPOCHS=50  #10,15,20,25,30,35,40,45,50,60,70,80,100
# ADAPTER=xattention # match, xattention
# LEARNING_RATE=0.00001,0.0005,0.0001,0.0002

# python3 ${EXPERIMENT}.py \
# storage_pth=${STORAGE_DIR} \
# fewshot.fine_tune=True \
# fewshot.adapter=${ADAPTER} \
# fewshot.n_supports=${N_SUPPORTS} \
# fewshot.train_epochs=${N_EPOCHS} \
# fewshot.learning_rate=${LEARNING_RATE}

python3 finetune.py storage_pth=${STORAGE_DIR}
