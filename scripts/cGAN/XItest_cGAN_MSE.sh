#!/bin/bash -f

#=============================
# JointFontGAN
# Modified from https://github.com/azadis/MC-GAN
# By Yankun Xi
#=============================


#=====================================
## Set Parameters
#=====================================
DATASET="$1"
PHASE="$2"
STR_INPUT="$3"
STR_OUTPUT=$4
FILE_DIR=$(pwd)
PC_NAME=$(hostname)
XI_DIR=${FILE_DIR%XIcodes*}
PROJECT_DIR=${FILE_DIR%JointFontGAN*}
MODEL=cGAN

EXE_PATH="${PROJECT_DIR}JointFontGAN/exe/${MODEL}/XItest_${MODEL}.py"
DATA_LOADER='base'
#BLANKS=0.85
# set pretrained glyphnet model
EXPERIMENT_DIR="${DATASET}_${MODEL}_MSE"
# set model parameters
MODEL_G=resnet_6blocks
MODEL_D=n_layers
n_layers_D=1
NORM=batch
PRENET=2_layers
FINESIZE=64
LOADSIZE=64
LAM_A=100
NITER=500
NITERD=100
EPOCH=600 #test at which epoch?
CUDA_ID=0
DOWNSAMPLING_0_N=2
DOWNSAMPLING_0_MULT=3

# build checkpoint folder
if [ ! -d "${PROJECT_DIR}JointFontGAN/checkpoints/${EXPERIMENT_DIR}" ]; then
	mkdir "${PROJECT_DIR}JointFontGAN/checkpoints/${EXPERIMENT_DIR}" -p;
fi
# clear old logs
if [ $EPOCH \> $NITER ]; then
  TESTLOG="${PROJECT_DIR}JointFontGAN/checkpoints/${EXPERIMENT_DIR}/result_${PHASE}_${NITER}+$((EPOCH-NITER))@${NITERD}.txt";
else
  TESTLOG="${PROJECT_DIR}JointFontGAN/checkpoints/${EXPERIMENT_DIR}/result_${PHASE}_${EPOCH}.txt";
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
--norm ${NORM} --str_input "${STR_INPUT}" --str_output "${STR_OUTPUT}" --loadSize \
${FINESIZE} --fineSize ${LOADSIZE} --display_id 0 --niter ${NITER} --niter_decay ${NITERD} \
--conditional --which_epoch ${EPOCH} --conv3d  --phase ${PHASE} \
--align_data --data_loader ${DATA_LOADER} \
--downsampling_0_n ${DOWNSAMPLING_0_N} --downsampling_0_mult ${DOWNSAMPLING_0_MULT} --stack_result
} 2>&1 | tee -a "${TESTLOG}"



