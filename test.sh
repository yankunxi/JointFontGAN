#!/bin/bash -f


FILE_DIR=$(pwd)
printf "${FILE_DIR} \n"
PC_NAME=$(hostname)
printf "${PC_NAME} \n"
XI_DIR=${FILE_DIR%XIcodes*}
printf "${XI_DIR} \n"
PROJECT_DIR=${FILE_DIR%xifontgan*}
printf "${PROJECT_DIR} \n"
EXE_PATH="${PROJECT_DIR}xifontgan/exe/LSTMcycleGAN/XItrain_LSTMcycleGAN.py"
printf "${EXE_PATH} \n"
CONFIG_DEVICE="config/device"
printf "${CONFIG_DEVICE} \n"

CODING_ROOT=$(awk "/^\[${PC_NAME}\]/{f=1}f==1&&/^auxiliary_root\
/{print \$3; exit}" "${CONFIG_DEVICE}")
PROJECT_RELATIVE=$(awk "/^\[${PC_NAME}\]/{f=1}f==1&&/^\
project_relative/{print \$3; exit}" "${CONFIG_DEVICE}")
USE_AUXILIARY=$(awk "/^\[${PC_NAME}\]/{f=1}f==1&&/^use_auxiliary\
/{print \$3; exit}" "${CONFIG_DEVICE}")
AUXILIARY_ROOT=$(awk "/^\[${PC_NAME}\]/{f=1}f==1&&/^auxiliary_root\
/{print \$3; exit}" "${CONFIG_DEVICE}")
USE_BUFFER=$(awk "/^\[${PC_NAME}\]/{f=1}f==1&&/^use_buffer\
/{print \$3; exit}" "${CONFIG_DEVICE}")
BUFFER_ROOT=$(awk "/^\[${PC_NAME}\]/{f=1}f==1&&/^buffer_root\
/{print \$3; exit}" "${CONFIG_DEVICE}")



/mnt/Files/XIauxiliary/XIdataset/font/English/${DATASET}"

AAA=1

printf "${PROJECT_RELATIVE} \n"

if [ "${USE_AUXILIARY}" = True ] ; then
  printf "${AUXILIARY_ROOT} \n"
fi

if [ "${USE_BUFFER}" = True ] ; then
  printf "${BUFFER_ROOT} \n"
fi


