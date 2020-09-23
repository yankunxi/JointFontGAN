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
python -u "${EXE_PATH}" --model "zi2zi" "cGAN" "EskGAN" --font "1980 portable.0.0" "advanced_led_board-7.0.0" \
"AntiqueNo14.0.0" "Belgrano-Regular.0.0" "Bitstream Vera Sans Bold Oblique.0.0" "Bitstream Vera Sans Mono Roman.0.0" \
"Bitstream Vera Sans Oblique.0.0" "BuriedBeforeBugsBB_Reg.0.0" "capacitor.0.0" "chatteryt.0.0" "ChockABlockNF.0.0" \
"DIGITALDREAMNARROW.0.0" "gather.0.0" "gosebmp2.0.0" "HoltwoodOneSC.0.0" "IMPOS0__.0.0" "inglobalbi.0.0" \
"JandaManateeBubble.0.0" "Kabinett-Fraktur-Halbfett.0.0" "keyrialt.0.0" "kimberley bl.0.0" "KOMIKABG.0.0" \
"KR Cloud Nine.0.0" "KR Rachel's Chalkboard.0.0" "lakeshor.0.0" "Lord-Juusai-Reigns.0.0" "nymonak.0.0" "Overdose.0.0" \
"pindownp.0.0" "SFAvondaleCond-Italic.0.0" "SFRetroesqueFX-Oblique.0.0" "Walbaum-Fraktur-Bold.0.0" "Walbaum-Fraktur.0.0"
}



