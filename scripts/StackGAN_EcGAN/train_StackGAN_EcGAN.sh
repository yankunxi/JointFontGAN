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
EXE_PATH="${PROJECT_DIR}xifontgan/exe/${MODEL}/XItrain_${MODEL}.py"
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
EPOCH=0
EPOCH1="400+0"
CUDA_ID=0


if [ ! -d "./checkpoints/${experiment_dir}" ]; then
    mkdir "./checkpoints/${experiment_dir}"
fi
LOG="./checkpoints/${experiment_dir}/output.txt"
if [ -f $LOG ]; then
    rm $LOG
fi

# =======================================
##COPY pretrained network from its corresponding directory
# =======================================
FROM="../../../../../../../../../XIauxiliary/XIcodes/Python/python3.6/pytorch1.2/XIfontGAN_yankunxi/XIFONTGAN_05/xifontgan/checkpoints/Capitals64_EcGAN/"
TO="../../../../../../../../../XIauxiliary/XIcodes/Python/python3.6/pytorch1.2/XIfontGAN_yankunxi/XIFONTGAN_05/xifontgan/checkpoints/${EXPERIMENT_DIR}"
if [ ! -d ${TO} ]; then
    mkdir -p ${TO}
fi
if [ ! -f "${TO}${EPOCH1}_net_G.pth" ]; then
    cp "${FROM}${EPOCH1}_net_G.pth" "${TO}"
    cp "${FROM}${EPOCH1}_net_G_3d.pth" "${TO}"
fi


exec &> >(tee -a "$LOG")

# =======================================
## Train End-2-End model
# =======================================
echo "TRAIN MODEL WITH REAL TRAINING DATA" 

CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u "${EXE_PATH}" --dataset ${DATASET} --experiment_dir "${EXPERIMENT_DIR}" --model ${MODEL} \
							  --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D} \
							  --norm ${NORM} --input_nc ${IN_NC} --output_nc ${O_NC} --input_nc_1 ${IN_NC_1} --output_nc_1 ${O_NC_1}\
							  --which_model_preNet ${PRENET} --fineSize ${FINESIZE} --loadSize ${LOADSIZE} --lambda_A ${LAM_A}\
							  --lambda_C ${LAM_C} --align_data --use_dropout --display_id 0 --niter ${NITER} --niter_decay ${NITERD}\
							  --batchSize ${BATCHSIZE} --conditional --save_epoch_freq 700 --rgb_out --partial \
							  --display_freq 70 --print_freq 70 --blanks 0 --conv3d --base_root ${base_dir} \
							  --data_loader ${DATA_LOADER} --which_epoch ${EPOCH} --which_epoch1 ${EPOCH1} #--base_font


# =======================================
## BASELINE: train only the second network on top of clean b/w glyphs
# =======================================

# CUDA_VISIBLE_DEVICES=${CUDA_ID} python train_Stack.py --dataset ${DATASET}  --name ${NAME} --model ${MODEL}
                                # --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D} --norm ${NORM}
                                # --input_nc ${IN_NC} --output_nc ${O_NC} --input_nc_1 ${IN_NC_1} --output_nc_1 ${O_NC_1} 
                                # --which_model_preNet ${PRENET} --fineSize ${FINESIZE} --loadSize ${LOADSIZE} --lambda_A ${LAM_A} 
                                # --lambda_C ${LAM_C} --align_data --use_dropout --display_id 0 --niter ${NITER} --niter_decay ${NITERD} 
                                # --batchSize ${BATCHSIZE} --conditional --save_epoch_freq 100 --rgb_out --partial --which_epoch ${EPOCH}
                                # --display_freq 5 --print_freq 5 --blanks 0 --base_font --conv3d --no_Style2Glyph --orna










