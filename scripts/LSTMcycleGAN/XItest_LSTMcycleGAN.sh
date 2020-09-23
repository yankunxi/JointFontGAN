#!/bin/bash -f

#=====================================
# MC-GAN
# Train and Test conditional GAN Glyph network
# By Samaneh Azadi
#=====================================


#=====================================
## Set Parameters
#=====================================
DATASET=$1
FILE_DIR=$(pwd)
XI_DIR=${FILE_DIR%XIcodes*}
PROJECT_DIR=${FILE_DIR%xifontgan*}
MODEL="LSTMcycleGAN"

EXE_PATH="${PROJECT_DIR}xifontgan/exe/${MODEL}/XItest_${MODEL}.py"

DATA_LOADER='extended_half_t'
#BLANKS=0.7
Base_DIR="/mnt/Auxiliary/XIauxiliary/XIdataset/font/English/${DATASET}/BASE"
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
EPOCH=100 #test at which epoch?
#EPOCH=latest #test at which epoch?
#EPOCH=1000 #test at which epoch?
CUDA_ID=0
PHASE=test

# build checkpoint folder
if [ ! -d "${PROJECT_DIR}xifontgan/checkpoints/${EXPERIMENT_DIR}" ]; then
	mkdir "${PROJECT_DIR}xifontgan/checkpoints/${EXPERIMENT_DIR}" -p
fi
# clear old logs
TESTLOG="${PROJECT_DIR}xifontgan/checkpoints/${EXPERIMENT_DIR}/result_${PHASE}.txt"
#A_LOG="${PROJECT_DIR}xifontgan/checkpoints/${EXPERIMENT_DIR}/output.txt"
if [ -f "$TESTLOG" ]; then
	rm "$TESTLOG"
fi

exec &> >(tee -a "$TESTLOG")


# =======================================
## Test Glyph Network on font dataset
# =======================================
CUDA_VISIBLE_DEVICES=${CUDA_ID} python "${EXE_PATH}" --dataset \
"${DATASET}" --experiment_dir "${EXPERIMENT_DIR}" --model ${MODEL} \
--which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} \
--n_layers_D ${n_layers_D} --which_model_preNet ${PRENET} --norm \
${NORM} --str_input "${STR_INPUT}" --str_output "${STR_OUTPUT}" \
--loadSize ${FINESIZE} --fineSize ${LOADSIZE} --display_id 0 \
--conditional --which_epoch ${EPOCH} --conv3d \
--align_data --data_loader ${DATA_LOADER} --phase ${PHASE} \
