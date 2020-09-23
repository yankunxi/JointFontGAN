#!/bin/bash -f

#DATASET=$1
#PHASE=$2
#STR_INPUT=$3
#STR_OUTPUT=$4
FILE_DIR=$(pwd)
PC_NAME=$(hostname)
XI_DIR=${FILE_DIR%XIcodes*}
PROJECT_DIR=${FILE_DIR%xifontgan*}
MODEL=""

EXE_PATH="${PROJECT_DIR}xifontgan/plot/show_paper.py"

{
python -u "${EXE_PATH}"  --grps 60 --blanks 0.85 --dataset "SandunLK10k64" --model "cGAN" "sk1GAN" "EskGAN_mixed" \
--string  --font "300 Trojans Leftalic" \
"A850-Roman-Medium" "Aargau-Medium" "Action Man Bold" "Airacobra Expanded" "Aldos Moon" "Algol VII" \
"Armor Piercing 2.0 BB" "B691-Sans-HeavyItalic" "bad robot italic laser" "Berthside" "BiergДrten Laser" \
"Book-BoldItalic" "Cartel" "DirtyBakersDozen" "Former Airline" "Funky Rundkopf NF" "Gamera" "genotype H BRK" \
"GlasgowAntique-Bold" "HydrogenWhiskey Ink" "Iconian Italic" "Jackson" "Johnny Fever" "Lincoln Lode" \
"Ocean View MF Initials" "QuickQuick Condensed" "Ribbon" "Saturn" "SF Chromium 24 SC Bold" "SF Comic Script Extended" \
"SF Telegraphic Light" "Underground NF" "VariShapes Solid" "Xerography" "Zado" "Commonwealth Expanded Italic"
}



