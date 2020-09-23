#!/bin/bash -f

#=====================================
# XIfontGAN
# MC-GAN
# Train and Test conditional GAN Glyph network
# By Samaneh Azadi
#=====================================


#=====================================
## Set Parameters
#=====================================

# input dataset folder when running
DATASET=$1
FILE_DIR=$(pwd)
PC_NAME=$(hostname)
XI_DIR=${FILE_DIR%XIcodes*}
PROJECT_DIR=${FILE_DIR%xifontgan*}
MODEL="LSTMcycleGAN"

EXE_PATH="${PROJECT_DIR}xifontgan/exe/${MODEL}/XItrain_${MODEL}.py"
#
#

DATA_LOADER='extended_half_t'
#BLANKS=0.7
# set pretrained glyphnet model
EXPERIMENT_DIR="${DATASET}_${MODEL}"
# set model parameters

MODEL_G=unet_64
MODEL_D=n_layers
n_layers_D=1
NORM=batch
PRENET=2_layers
FINESIZE=64
LOADSIZE=64
LAM_A=100
#NITER=500
#NITERD=100
NITER=80
NITERD=20
#batch size
BATCHSIZE=100
CUDA_ID=0
#LOAD_EPOCH='latest'
LOAD_EPOCH='100'
PHASE=train

# build checkpoint folder
if [ ! -d "${PROJECT_DIR}xifontgan/checkpoints/${experiment_dir}" ]; then
	mkdir "${PROJECT_DIR}xifontgan/checkpoints/${experiment_dir}" -p
fi

TRAINLOG="${PROJECT_DIR}xifontgan/checkpoints/${experiment_dir}/log_${PHASE}.txt"
#A_LOG="${PROJECT_DIR}xifontgan/checkpoints/${experiment_dir}/output.txt"
#if [ -f "$LOG" ]; then
#	rm "$LOG"
#fi

exec &> >(tee -a "$TRAINLOG")

CUDA_VISIBLE_DEVICES=${CUDA_ID} python "${EXE_PATH}" --dataset "${DATASET}" --experiment_dir "${experiment_dir}" --model ${MODEL} \
--which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D}  --n_layers_D ${n_layers_D} --which_model_preNet ${PRENET} \
--norm ${NORM} --str_input "${STR_INPUT}" --fineSize ${FINESIZE} --loadSize ${LOADSIZE} \
--lambda_A ${LAM_A} --align_data --use_dropout --display_id 0 --niter ${NITER} --niter_decay ${NITERD} \
--batchSize ${BATCHSIZE} --conditional --save_epoch_freq 10 \
--print_freq 500 --conv3d --data_loader ${DATA_LOADER} \
--blanks ${BLANKS} --which_epoch ${LOAD_EPOCH} --continue_train


