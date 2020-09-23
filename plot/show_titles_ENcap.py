import os, sys
from PIL import Image, ImageEnhance, ImageDraw, ImageChops
if __package__ is None:
    this_path = os.path.dirname(os.path.realpath(__file__))
    project_root = this_path.rpartition("xifontgan")[0]
    sys.path.insert(0, project_root)
    from xifontgan.util.indexing import str2index
else:
    from ..util.indexing import str2index

# results_dir = "/mnt/Auxiliary/XIauxiliary/XIcodes/Python/python3.6" \
#               "/pytorch1.2/XIfontGAN_yankunxi/XIFONTGAN_05" \
#               "/xifontgan/results/Capitals64_cGAN/test_500+100" \
#               "@100_BEAUTY AND THE/_/images"
# results__dir = "/mnt/Auxiliary/XIauxiliary/XIcodes/Python/python3.6" \
#                "/pytorch1.2/XIfontGAN_yankunxi/XIFONTGAN_05" \
#                "/xifontgan/results/Capitals64_cGAN/test_500+100" \
#                "@100_BEAUTY AND THE/BEAUTY AND THE BEAST"\


# input_str = '"One, remember'
# output_str = '"One, remember to look up at the stars '\
#              'and not down at your feet. Two, never '\
#              'give up work. Work gives you meaning and '\
#              'purpose and life is empty without it. '\
#              'Three, if you are lucky enough to find '\
#              'love, remember it is there and don\'t '\
#              'throw it away.” by Stephen Hawking '
# input_n = 14

# input_str = '"The impor'
# output_str = '"The important thing is not to stop questioning. ' \
#              'Curiosity has its own reason for existence. One ' \
#              'cannot help but be in awe when he contemplates the ' \
#              'mysteries of eternity, of life, of the marvelous ' \
#              'structure of reality. It is enough if one tries ' \
#              'merely to comprehend a little of this mystery each ' \
#              'day." by Albert Einstein '
# input_n = 10

# # input_str = 'I THINK THEREF'
# output_str = 'I THINK THEREFORE I AM '
# input_n = 14

# # input_str = 'KNOWLEDG'
# output_str = 'KNOWLEDGE MAKES A MAN UNFIT TO BE A SLAVE '
# input_n = 8

# # input_str = 'THE TRUTH IS F'
# output_str = 'THE TRUTH IS FOUND WHEN MEN ARE FREE TO PURSUE IT '
# input_n = 14

# # input_str = 'THE PRICE O'
# output_str = 'THE PRICE OF GREATNESS IS RESPONSIBILITY '
# input_n = 11

# # input_str = 'SCIENCE WITH'
# output_str = 'SCIENCE WITHOUT RELIGION IS LAME RELIGION WITHOUT SCIENCE IS BLINK '
# input_n = 12

# # input_str = 'WHERE THERE IS A'
# output_str = 'WHERE THERE IS A MONSTER THERE IS A MIRACLE '
# input_n = 16

# input_str = 'ONE CANNOT STEP'
output_str = 'ONE CANNOT STEP TWICE IN THE SAME RIVER '
input_n = 15

# input_n = len(input_str)
input_str = output_str[0:input_n]

results_dir = '/mnt/Auxiliary/XIauxiliary/XIcodes/Python/python3.6' \
              '/pytorch1.2/XIfontGAN_yankunxi/XIFONTGAN_05' \
              '/xifontgan/results/Capitals64_EskGAN' \
              '/test_500+100@100_'+input_str+'/_/images'
results__dir = '/mnt/Auxiliary/XIauxiliary/XIcodes/Python/python3.6' \
               '/pytorch1.2/XIfontGAN_yankunxi/XIFONTGAN_05' \
               '/xifontgan/results/Capitals64_EskGAN' \
               '/test_500+100@100_'+input_str+'/string100'
if not os.path.isdir(results__dir):
    os.mkdir(results__dir)

IMG_EXTENSIONS = ['.png']
w = 64
h = 64
p = 2
row_max = 4500
trans = True
red_box = False

fonts = ["1980 portable.0.0", "advanced_led_board-7.0.0",
         "AntiqueNo14.0.0", "Belgrano-Regular.0.0",
         "Bitstream Vera Sans Bold Oblique.0.0",
         "Bitstream Vera Sans Mono Roman.0.0",
         "Bitstream Vera Sans Oblique.0.0",
         "BuriedBeforeBugsBB_Reg.0.0", "capacitor.0.0",
         "chatteryt.0.0", "ChockABlockNF.0.0",
         "DIGITALDREAMNARROW.0.0", "gather.0.0", "gosebmp2.0.0",
         "HoltwoodOneSC.0.0", "IMPOS0__.0.0", "inglobalbi.0.0",
         "JandaManateeBubble.0.0", "Kabinett-Fraktur-Halbfett.0.0",
         "keyrialt.0.0", "kimberley bl.0.0", "KOMIKABG.0.0",
         "KR Cloud Nine.0.0", "KR Rachel's Chalkboard.0.0",
         "lakeshor.0.0", "Lord-Juusai-Reigns.0.0", "nymonak.0.0",
         "Overdose.0.0", "pindownp.0.0", "SFAvondaleCond-Italic.0.0",
         "SFRetroesqueFX-Oblique.0.0", "Walbaum-Fraktur-Bold.0.0",
         "Walbaum-Fraktur.0.0"]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def trim(im, p, min_):
    w, h = im.size
    bc = im.getpixel((0, 0))
    bg = Image.new(im.mode, im.size, bc)
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -5)
    bbox = diff.getbbox()
    if bbox:
        x1 = bbox[0]
        x2 = bbox[2]
        im_ = im.crop((x1, 0, x2, h))
        if (x2 - x1) > min_:
            min_ = x2 - x1
    output = Image.new(im.mode, (min_+2*p, h), bc)
    if bbox:
        output.paste(im_, (int((min_-x2+x1)/2)+p, 0))
    return output


def t_trans(img):
    img = img.convert("RGBA")
    pixdata = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            # print(pixdata[x,y])
            r, g, b, _ = pixdata[x, y]
            pixdata[x, y] = (r, g, b, 255-int((r+g+b)/3))
    return img


for root, _, fnames in sorted(os.walk(results_dir)):
    for fname in fnames:
        font = fname.partition('_')[0]
        if font in fonts:
        # if True:
            img = Image.open(os.path.join(root, fname))
            # l = str2index("BEAUTI AND THE BEAST", 'ENcap')
            # l = str2index('"Science without religion is lame. Religion '
            #               'wihout sicence is blind." Albert Einstein ',
            #               'ENfull')

            # l = str2index('"The important thing is not to stop '
            #               'questioning. Curiosity has its own reason '
            #               'for existence. One cannot help but be in awe '
            #               'when he contemplates the mysteries of '
            #               'eternity, of life, of the marvelous '
            #               'structure of reality. It is enough if one '
            #               'tries merely to comprehend a little of this '
            #               'mystery each day." Albert Einstein ',
            #               'ENfull')

            l = str2index(output_str, 'ENcap')
            new_img = Image.new('RGBA', (w * len(l), h),
                                color=(255, 255, 255))
            gt_l = []
            op_l = []
            w_gt_min = 64
            w_gt_min_ = 0
            w_op_min = 64
            w_op_min_ = 0
            for i in range(27):
                if i == 26:
                    w_gt_min_ = w_gt_min
                    w_op_min_ = w_op_min
                gt_ = img.crop((i * w, h, (i + 1) * w, 2 * h))
                # gt_ = trim(gt_, p, w_gt_min_)
                # gt_ = t_trans(gt_)
                gt_l += [gt_]
                gt_ = gt_.size[0]
                if i < 26:
                    if gt_ < w_gt_min:
                        w_gt_min = gt_
                op_ = img.crop((i * w, 2 * h, (i + 1) * w, 3 * h))
                # op_ = trim(op_, p, w_op_min_)
                # op_ = t_trans(op_)
                op_l += [op_]
                op_ = op_.size[0]
                if i < 26:
                    if op_ < w_op_min:
                        w_op_min = op_
            total_w_gt = [0, 0, 0]
            total_w_op = [0, 0, 0]
            total_h_gt = h
            total_h_op = h
            x_gt_l = [[], []]
            x_op_l = [[], []]
            for grp in l:
                if grp > -1:
                    x_gt_l[0] += [(total_w_gt[0], total_h_gt - h)]
                    x_op_l[0] += [(total_w_op[0], total_h_op - h)]
                    a = gt_l[grp].size[0]
                    total_w_gt[0] += a
                    if a < w_gt_min:
                        w_gt_min = a
                    a = op_l[grp].size[0]
                    total_w_op[0] += a
                    if a < w_op_min:
                        w_op_min = a
                else:
                    x_gt_l[0] += [(total_w_gt[0], total_h_gt - h)]
                    x_op_l[0] += [(total_w_op[0], total_h_op - h)]
                    total_w_gt[0] += w_gt_min_
                    total_w_op[0] += w_op_min_
                    if total_w_gt[0] < row_max:
                        total_w_gt[1] = total_w_gt[0]
                        x_gt_l[1] += x_gt_l[0]
                        x_gt_l[0] = []
                    else:
                        total_h_gt += h
                        total_w_gt[0] = total_w_gt[0] - total_w_gt[1]
                        x_gt_l[1] += [(x - total_w_gt[1], y + h)
                                      for x, y in x_gt_l[0]]
                        x_gt_l[0] = []
                    if total_w_op[0] < row_max:
                        total_w_op[1] = total_w_op[0]
                        x_op_l[1] += x_op_l[0]
                        x_op_l[0] = []
                    else:
                        total_h_op += h
                        total_w_op[0] = total_w_op[0] - total_w_op[1]
                        x_op_l[1] += [(x - total_w_op[1], y + h)
                                      for x, y in x_op_l[0]]
                        x_op_l[0] = []
                    if total_w_gt[1] > total_w_gt[2]:
                        total_w_gt[2] = total_w_gt[1]
                    if total_w_op[1] > total_w_op[2]:
                        total_w_op[2] = total_w_op[1]
            gt_img = Image.new('RGBA', (total_w_gt[2], total_h_gt),
                               color=(255, 255, 255))
            if trans:
                op_img = Image.new('RGBA', (total_w_gt[2],
                                            total_h_gt),
                                   color=(255, 255, 255))
            else:
                op_img = Image.new('RGBA', (total_w_op[2],
                                            total_h_op),
                               color=(255, 255, 255))
            n = 0
            for grp in l:
                if grp > -1:
                    gt_img.paste(gt_l[grp], x_gt_l[1][n])
                    if trans:
                        op_img.paste(op_l[grp], x_gt_l[1][n])
                    else:
                        op_img.paste(op_l[grp], x_op_l[1][n])
                n += 1

            if red_box:
                img_draw = ImageDraw.Draw(gt_img)
                img_draw.rectangle(
                    [0, 0, x_gt_l[1][input_n][0] + 1, h],
                    outline=(175, 0, 0, 255), width=2)


            gt_img.save(os.path.join(results__dir, font + '_gt.png'))
            op_img.save(
                os.path.join(results__dir, font + '_output.png'))
            print(font)





