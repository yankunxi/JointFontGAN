import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import json
import pickle
import collections
from model.preprocessing_helper import draw_single_char_by_font, \
    draw_example, CHAR_SIZE, CANVAS_SIZE
from package import save_train_valid_data

source_dir = './data/SandunLK10k64'
base_font = 'A750-Sans-Cd-Medium.png'

source_train = os.path.join(source_dir, 'train')
source_test = os.path.join(source_dir, 'test')

w = 64
h = 64
ch = 60

# fonts = ["1980 portable.0.0", "advanced_led_board-7.0.0",
#          "AntiqueNo14.0.0", "Belgrano-Regular.0.0",
#          "Bitstream Vera Sans Bold Oblique.0.0",
#          "Bitstream Vera Sans Mono Roman.0.0",
#          "Bitstream Vera Sans Oblique.0.0",
#          "BuriedBeforeBugsBB_Reg.0.0", "capacitor.0.0",
#          "chatteryt.0.0", "ChockABlockNF.0.0",
#          "DIGITALDREAMNARROW.0.0", "gather.0.0", "gosebmp2.0.0",
#          "HoltwoodOneSC.0.0", "IMPOS0__.0.0", "inglobalbi.0.0",
#          "JandaManateeBubble.0.0", "Kabinett-Fraktur-Halbfett.0.0",
#          "keyrialt.0.0", "kimberley bl.0.0", "KOMIKABG.0.0",
#          "KR Cloud Nine.0.0", "KR Rachel's Chalkboard.0.0",
#          "lakeshor.0.0", "Lord-Juusai-Reigns.0.0", "nymonak.0.0",
#          "Overdose.0.0", "pindownp.0.0", "SFAvondaleCond-Italic.0.0",
#          "SFRetroesqueFX-Oblique.0.0", "Walbaum-Fraktur-Bold.0.0",
#          "Walbaum-Fraktur.0.0"]

fonts = ["300 Trojans Leftalic", "A850-Roman-Medium",
         "Aargau-Medium", "Action Man Bold", "Airacobra Expanded",
         "Aldos Moon", "Algol VII", "Armor Piercing 2.0 BB",
         "B691-Sans-HeavyItalic", "bad robot italic laser",
         "Berthside", "BiergДrten Laser", "Book-BoldItalic",
         "Cartel", "DirtyBakersDozen", "Former Airline",
         "Funky Rundkopf NF", "Gamera", "genotype H BRK",
         "GlasgowAntique-Bold", "HydrogenWhiskey Ink",
         "Iconian Italic", "Jackson", "Johnny Fever", "Lincoln Lode",
         "Ocean View MF Initials", "QuickQuick Condensed", "Ribbon",
         "Saturn", "SF Chromium 24 SC Bold",
         "SF Comic Script Extended", "SF Telegraphic Light",
         "Underground NF", "VariShapes Solid", "Xerography",
         "Zado", "Commonwealth Expanded Italic"]

base_img_s = []
base_img = Image.open(os.path.join(source_train, base_font))
for j in range(ch):
    base_img_ = base_img.crop((j * w, 0, (j + 1) * w, h))
    base_img_s += [base_img_]

dict_inds = pickle.load(open('dict.pkl', 'rb'))

i = 0
for root, _, fnames in sorted(os.walk(source_train)):
    for fname in fnames:
        if i == 5:
            break
        img = Image.open(os.path.join(root, fname))
        for j in range(ch):
            img_ = img.crop((j * w, 0, (j + 1) * w, h))
            img_d = Image.new(img.mode, (2 * w, h),
                              color=(255, 255, 255))
            img_d.paste(img_, (0, 0))
            img_d.paste(base_img_s[j], (w, 0))
            img_d.save(os.path.join(source_dir, 'paired_images_train',
                                    '%d_%0.4d.png' %
                                    (i, j)))
        i += 1
        print("Number of training fonts:", i)

i = 0
for root, _, fnames in sorted(os.walk(source_test)):
    for fname in fnames:
        font = fname.rpartition('.')[0]
        if font in fonts:
            img = Image.open(os.path.join(root, fname))
            for j in range(8):
                l = dict_inds[fname][-8:][j]
                img_ = img.crop((l * w, 0, (l + 1) * w, h))
                img_d = Image.new(img.mode, (2 * w, h),
                                      color=(255, 255, 255))
                img_d.paste(img_, (0, 0))
                img_d.paste(base_img_s[l], (w, 0))
                img_d.save(os.path.join(source_dir,
                                        'paired_images_test',
                                        '%d_%0.4d.png' % (i, j)))
            i += 1
            print("Number of testing fonts:", i)


# save_train_valid_data(save_dir='experiments/data', sample_dir=os.path.join(source_dir, 'paired_images_train'), split_ratio=0.01)
save_train_valid_data(save_dir='experiments_finetune/data',
                      sample_dir=os.path.join(source_dir,
                                              'paired_images_test'), split_ratio=0)

