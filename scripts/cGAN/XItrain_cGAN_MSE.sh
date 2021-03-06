#!/bin/bash -f

#=====================================
# XIfontGAN
#=====================================


#=====================================
## Set Parameters
#=====================================

# input dataset folder when running
DATASET=$1
FILE_DIR=$(pwd)
PC_NAME=$(hostname)
XI_DIR=${FILE_DIR%XIcodes*}
PROJECT_DIR=${FILE_DIR%JointFontGAN*}
MODEL=cGAN

EXE_PATH="${PROJECT_DIR}JointFontGAN/exe/${MODEL}/XItrain_${MODEL}.py"
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
#NITER=500
#NITERD=100
NITER=500
NITERD=100
#batch size
BATCHSIZE=150
CUDA_ID=0
LOAD_EPOCH=516
DOWNSAMPLING_0_N=2
DOWNSAMPLING_0_MULT=3
DIFF_LOSS0="MSE"


# build checkpoint folder
if [ ! -d "${PROJECT_DIR}JointFontGAN/checkpoints/${EXPERIMENT_DIR}" ]; then
	mkdir "${PROJECT_DIR}JointFontGAN/checkpoints/${EXPERIMENT_DIR}" -p
fi

TRAINLOG="${PROJECT_DIR}JointFontGAN/checkpoints/${EXPERIMENT_DIR}/log_train.txt"
#A_LOG="${PROJECT_DIR}JointFontGAN/checkpoints/${EXPERIMENT_DIR}/output.txt"
#if [ -f "$LOG" ]; then
#	rm "$LOG"
#fi

# =======================================
## Train Glyph Network on font dataset
# =======================================
{
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u "${EXE_PATH}" --dataset "${DATASET}" --experiment_dir "${EXPERIMENT_DIR}" --model ${MODEL} \
--which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D}  --n_layers_D ${n_layers_D} --which_model_preNet ${PRENET} \
--norm ${NORM} --str_input "${STR_INPUT}" --fineSize ${FINESIZE} --loadSize ${LOADSIZE} \
--lambda_A ${LAM_A} --align_data --use_dropout --display_id 0 --niter ${NITER} --niter_decay ${NITERD} \
--batchSize ${BATCHSIZE} --conditional --downsampling_0_n ${DOWNSAMPLING_0_N} \
--downsampling_0_mult ${DOWNSAMPLING_0_MULT} --diff_loss0 ${DIFF_LOSS0} \
--save_epoch_freq 100 --print_freq 1 --data_loader ${DATA_LOADER} \
--conv3d --which_epoch ${LOAD_EPOCH} --stack_result --continue_latest
} 2>&1 | tee -a "${TRAINLOG}"



# =======================================
## Train on RGB inputs to generate RGB outputs; Image Translation in the paper
# =======================================
# CUDA_VISIBLE_DEVICES=2 python ~/AdobeFontDropper/train.py --dataset ../datasets/Capitals_colorGrad64/ --name "${experiment_dir}"\
						 # --model cGAN --which_model_netG resnet_6blocks --which_model_netD n_layers --n_layers_D 1 --which_model_preNet 2_layers \
						 # --norm batch --input_nc 78 --output_nc 78 --fineSize 64 --loadSize 64 --lambda_A 100 --align_data --use_dropout \
						 # --display_id 0 --niter 500 --niter_decay 1000 --batchSize 100 --conditional --save_epoch_freq 20 --display_freq 2 --rgb

# =======================================
## Consider input as tiling of input glyphs rather than a stack
# =======================================

# CUDA_VISIBLE_DEVICES=2 python ~/AdobeFontDropper/train.py --dataset ../datasets/Capitals64/ --name "${experiment_dir}" \
				# --model cGAN --which_model_netG resnet_6blocks --which_model_netD n_layers  --n_layers_D 1 --which_model_preNet 2_layers\
				# --norm batch --input_nc 1 --output_nc 1 --fineSize 64 --loadSize 64 --lambda_A 100 --align_data --use_dropout\
				# --display_id 0 --niter 500 --niter_decay 2000 --batchSize 5 --conditional --save_epoch_freq 10 --display_freq 5 --print_freq 100 --flat



