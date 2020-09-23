#!/bin/bash -f

#=====================================
# MC-GAN
# Train and Test End-to-End network
# By Samaneh Azadi
#=====================================


#=====================================
## Set Parameters
#=====================================
COLLECTION=$1
DATA=$2
DATASET="${COLLECTION}/${DATA}/"
FILE_DIR=$(pwd)
PC_NAME=$(hostname)
XI_DIR=${FILE_DIR%XIcodes*}
PROJECT_DIR=${FILE_DIR%xifontgan*}
MODEL="StackGAN_EcGAN"
EXE_PATH="${PROJECT_DIR}xifontgan/exe/${MODEL}/XItest_${MODEL}.py"
base_dir="Capitals64/BASE/"
DATA_LOADER='Estack'
#BLANKS=0.85
# set pretrained glyphnet model
EXPERIMENT_DIR="${DATASET}${MODEL}/"
# set model parameters
MODEL_G=resnet_6blocks
MODEL_D=n_layers
n_layers_D=1
NORM=batch
IN_NC=26
O_NC=26
IN_NC_1=3
O_NC_1=3
PRENET=2_layers
LR=0.002
FINESIZE=64
LOADSIZE=64
LAM_A=300
LAM_C=10
NITER=400
NITERD=300
BATCHSIZE=$3
EPOCH=700
EPOCH1="400+0"
CUDA_ID=0


if [ ! -d "./checkpoints/${experiment_dir}" ]; then
	mkdir "./checkpoints/${experiment_dir}"
fi
LOG="./checkpoints/${experiment_dir}/test.txt"
if [ -f $LOG ]; then
	rm $LOG
fi

# =======================================
##COPY pretrained network from its corresponding directory
# =======================================
#model_1_pretrained="./checkpoints_64_small_${EPOCH}/GlyphNet_pretrain"
#if [ ! -f "./checkpoints/${experiment_dir}/${EPOCH}_net_G.pth" ]; then
#    cp "${model_1_pretrained}/${EPOCH}_net_G.pth" "./checkpoints/${experiment_dir}/"
#    cp "${model_1_pretrained}/${EPOCH}_net_G_3d.pth" "./checkpoints/${experiment_dir}/"
#fi


exec &> >(tee -a "$LOG")
# =======================================
## Test End-2-End model
# =======================================
#CUDA_VISIBLE_DEVICES=${CUDA_ID} python test_Stack.py --dataset ${DATASET} --name "${experiment_dir}" --model ${MODEL}\
#								 --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D} --grps ${GRP}\
#								 --norm ${NORM} --input_nc ${IN_NC} --output_nc ${O_NC} --input_nc_1 ${IN_NC_1} --output_nc_1 ${O_NC_1}\
#								 --which_model_preNet ${PRENET} --fineSize ${FINESIZE} --loadSize ${LOADSIZE} --display_id 0\
#								 --batchSize 1 --conditional --rgb_out --partial --align_data --which_epoch ${EPOCH} --which_epoch1 ${EPOCH1}\
#								 --blanks 0 --conv3d  --base_root ${base_dir}

CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u "${EXE_PATH}" --dataset ${DATASET} --experiment_dir "${EXPERIMENT_DIR}" --model ${MODEL} \
							  --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D} \
							  --norm ${NORM} --input_nc ${IN_NC} --output_nc ${O_NC} --input_nc_1 ${IN_NC_1} --output_nc_1 ${O_NC_1}\
							  --which_model_preNet ${PRENET} --fineSize ${FINESIZE} --loadSize ${LOADSIZE} \
							  --display_id 0 --niter ${NITER} --niter_decay ${NITERD}\
							  --batchSize 1 --conditional --rgb_out --partial --align_data --which_epoch ${EPOCH} --which_epoch1 ${EPOCH1}\
							  --data_loader ${DATA_LOADER} --blanks 0 --conv3d  --base_root ${base_dir}



# =======================================
## test only the second network for clean b/w glyphs
# =======================================

# CUDA_VISIBLE_DEVICES=${CUDA_ID} python test_Stack.py --dataset ${DATASET} --name "${experiment_dir}" --model ${MODEL}\
								 # --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D} --norm ${NORM} \
								 # --input_nc ${IN_NC} --output_nc ${O_NC} --input_nc_1 ${IN_NC_1} --output_nc_1 ${O_NC_1} --which_model_preNet ${PRENET}\
								 #  --fineSize ${FINESIZE} --loadSize ${LOADSIZE} --align_data --display_id 0 \
								 #  --batchSize 1 --conditional --rgb_out --partial  --which_epoch ${EPOCH} --which_epoch1 ${EPOCH1} --blanks 0\
								 #   --conv3d --no_Style2Glyph --orna --base_root ${base_dir}




