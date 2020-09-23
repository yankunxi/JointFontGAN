#!/bin/bash -f

#=====================================
# MC-GAN
# Train and Test conditional GAN Glyph network
# By Samaneh Azadi
#=====================================


#=====================================
## Set Parameters
#=====================================
DATASET="$1"
PHASE="$2"
STR_INPUT="$3"
FILE_DIR=$(pwd)
PC_NAME=$(hostname)
XI_DIR=${FILE_DIR%XIcodes*}
PROJECT_DIR=${FILE_DIR%xifontgan*}
MODEL='skGAN'

EXE_PATH="${PROJECT_DIR}xifontgan/exe/${MODEL}/XItest_${MODEL}.py"
DATA_LOADER='skeleton'
#BLANKS=0.7
# set pretrained glyphnet model
EXPERIMENT_DIR="${DATASET}_${MODEL}"
# set model parameters
MODEL_G=resnet_6blocks
MODEL_D=n_layers
n_layers_D=1
NORM=batch
PRENET=2_layers
FINESIZE=64
LOADSIZE=64
LAM_A=30
NITER=500
NITERD=100
EPOCH=600 #test at which epoch?
CUDA_ID=0
DOWNSAMPLING_0_N=2
DOWNSAMPLING_0_MULT=3

SKMODE=1

# build checkpoint folder
if [ ! -d "${PROJECT_DIR}xifontgan/checkpoints/${EXPERIMENT_DIR}" ]; then
	mkdir "${PROJECT_DIR}xifontgan/checkpoints/${EXPERIMENT_DIR}" -p;
fi
# clear old logs
if [ $EPOCH \> $NITER ]; then
  TESTLOG="${PROJECT_DIR}xifontgan/checkpoints/${EXPERIMENT_DIR}/result_${PHASE}_${NITER}+$((EPOCH-NITER))@${NITERD}.txt";
else
  TESTLOG="${PROJECT_DIR}xifontgan/checkpoints/${EXPERIMENT_DIR}/result_${PHASE}_${EPOCH}.txt";
fi

if [ -f "$TESTLOG" ]; then
	rm "$TESTLOG"
fi
echo ${TESTLOG}

# =======================================
## Test Glyph Network on font dataset
## =======================================
{
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u "${EXE_PATH}" --dataset \
"${DATASET}" --experiment_dir "${EXPERIMENT_DIR}" --model ${MODEL} \
--which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D}  --n_layers_D ${n_layers_D} --which_model_preNet ${PRENET} \
--norm ${NORM} --str_input "${STR_INPUT}" --loadSize \
${FINESIZE} --fineSize ${LOADSIZE} --display_id 0 --niter ${NITER} --niter_decay ${NITERD} \
--conditional --which_epoch ${EPOCH} --conv3d  --phase ${PHASE} \
--align_data --data_loader ${DATA_LOADER} --skmode ${SKMODE} \
--downsampling_0_n ${DOWNSAMPLING_0_N} --downsampling_0_mult ${DOWNSAMPLING_0_MULT}
} 2>&1 | tee -a "${TESTLOG}"


