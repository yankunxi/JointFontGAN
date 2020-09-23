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

EXE_PATH="${PROJECT_DIR}xifontgan/plot/show_dataset.py"

{
python -u "${EXE_PATH}" --blanks 0.85 --dataset "SandunLK10k64" --model "cGAN" --font "300 Trojans Leftalic" \
"A850-Roman-Medium" "Aargau-Medium" "Action Man Bold_" "Airacobra Expanded" "Aldos Moon" "Algol VII" \
"Armor Piercing 2.0 BB" "B691-Sans-HeavyItalic" "bad robot italic laser" "Berthside" "BiergДrten Laser" \
"Book-BoldItalic_" "Cartel" "DirtyBakersDozen" "Former Airline" "Funky Rundkopf NF" "Gamera" "genotype H BRK" \
"GlasgowAntique-Bold" "HydrogenWhiskey Ink" "Iconian Italic" "Jackson" "Johnny Fever" "Lincoln Lode" \
"Ocean View MF Initials" "QuickQuick Condensed" "Ribbon" "Saturn" "SF Chromium 24 SC Bold" "SF Comic Script Extended" \
"SF Telegraphic Light" "Underground NF" "VariShapes Solid" "Xerography" "Zado" "Commonwealth Expanded Italic"
}



#\
#"DIGITALDREAMNARROW.0.0" "gather.0.0" "gosebmp2.0.0" "HoltwoodOneSC.0.0" "IMPOS0__.0.0" "inglobalbi.0.0" \
#"JandaManateeBubble.0.0" "Kabinett-Fraktur-Halbfett.0.0" "keyrialt.0.0" "kimberley bl.0.0" "KOMIKABG.0.0" \
#"KR Cloud Nine.0.0" "KR Rachel's Chalkboard.0.0" "lakeshor.0.0" "Lord-Juusai-Reigns.0.0" "nymonak.0.0" "Overdose.0.0" \
#"pindownp.0.0" "SFAvondaleCond-Italic.0.0" "SFRetroesqueFX-Oblique.0.0" "Walbaum-Fraktur-Bold.0.0" "Walbaum-Fraktur.0.0"