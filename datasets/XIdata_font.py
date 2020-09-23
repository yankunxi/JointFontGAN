import argparse
import sys
import random
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pickle
import collections

PUNC_CHARSET = None
EN_CHARSET = None
ENc_CHARSET = None
ENl_CHARSET = None
PUNCf_CHARSET = None
ZH_CHARSET = None
ZHt_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None


DEFAULT_CHARSET = "./char.pkl"
DEFAULT_FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"


def load_global_charset():
    global EN_CHARSET, ENc_CHARSET, ENl_CHARSET
    global ZH_CHARSET, ZHt_CHARSET, JP_CHARSET, KR_CHARSET
    global PUNC_CHARSET, PUNCf_CHARSET
    charset = pickle.load(open(DEFAULT_CHARSET, 'rb'))
    PUNC_CHARSET = charset['PUNC']
    EN_CHARSET = charset['EN']
    ENc_CHARSET = charset['ENc']
    ENl_CHARSET = charset['ENl']
    PUNCf_CHARSET = charset['PUNCf']
    ZH_CHARSET = charset['ZH']
    ZHt_CHARSET = charset['ZHt']
    JP_CHARSET = charset['JP']
    KR_CHARSET = charset['KR']
    # cjk_ = {'PUNC':PUNC_CHARSET, 'PUNCf': PUNCf_CHARSET,
    #         'EN': EN_CHARSET, 'ENc': ENc_CHARSET, 'ENl': ENl_CHARSET,
    #         'KR': KR_CHARSET, 'JP': JP_CHARSET, 'ZH': ZH_CHARSET,
    #         'ZHt': ZHt_CHARSET}
    # pickle.dump(cjk_, open(DEFAULT_CHARSET,'wb'))


def draw_single_char(ch, font, canvas_x=64, canvas_y=64, x_offset=0,
                     y_offset=0):
    img = Image.new("RGB", (canvas_x, canvas_y), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x, y = draw.textsize(ch, font=font)
    draw.text(((canvas_x - x) / 2 + x_offset, y_offset), ch,
              (0, 0, 0), font=font)
    return img, y


def draw_row(charset, basefont, font, canvas_x=64, canvas_y=64,
             x_offset=0, y_offset=0, replace_hashes=False):
    l = len(charset)
    img = Image.new("RGB", (canvas_x * l, canvas_y),
                    (255, 255, 255))
    if replace_hashes:
        filter_hashes, y_max = \
            filter_recurring_hash(charset, font, canvas_x=canvas_x,
                                  canvas_y=canvas_y)
        filter_hashes = set(filter_hashes)
        # print("filter hashes -> %s" % (
        #     ",".join([str(h) for h in filter_hashes])))
    for i in range(l):
        temp_img, _ = draw_single_char(charset[i], font,
                                       canvas_x=canvas_x,
                                       canvas_y=canvas_y,
                                       x_offset=x_offset,
                                       y_offset=y_offset+(
                                                canvas_y-y_max)/2)
        if replace_hashes:
            temp_hash = hash(temp_img.tobytes())
            if temp_hash in filter_hashes:
                temp_img, _ = draw_single_char(charset[i], basefont,
                                               canvas_x=canvas_x,
                                               canvas_y=canvas_y,
                                               x_offset=x_offset,
                                               y_offset=y_offset + (
                                                       canvas_y -
                                                       y_max) / 2)
        img.paste(temp_img, (canvas_x * i, 0))
    return img


def filter_recurring_hash(charset, font, canvas_x=56, canvas_y=64,
                          x_offset=0, y_offset=0):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = random.sample(charset, len(charset))
    sample = _charset[:]
    hash_count = collections.defaultdict(int)
    y_max = 0
    for c in sample:
        img, y = draw_single_char(c, font, canvas_x=canvas_x,
                               canvas_y=canvas_y, x_offset=x_offset,
                               y_offset=y_offset)
        hash_count[hash(img.tobytes())] += 1
        if y > y_max:
            y_max = y
    recurring_hashes = [d for d in list(hash_count.items()) if
                        d[1] > 2]
    return [rh[0] for rh in recurring_hashes], y_max


def fonts2dataset(base_dir, fonts, charset, data_dir,
                  char_size=48, canvas_x=64, canvas_y=64,
                  x_offset=0, y_offset=0, replace_hashes=False):
    basefont = ImageFont.truetype(base_dir, size=char_size)
    for font_dir in fonts:
        font = ImageFont.truetype(font_dir, size=char_size)
        img = draw_row(charset, basefont, font, canvas_x=canvas_x,
                       canvas_y=canvas_y, x_offset=x_offset,
                       y_offset=y_offset, replace_hashes=replace_hashes)
        img.save(os.path.join(data_dir, os.path.basename(
            font_dir).replace('.ttf', '.png')))


def split(path, input, height, width, k):
    target = os.path.basename(path)
    im = Image.open(input)
    imgwidth, imgheight = im.size
    img_list = []
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            defaultbox = (0,0,64,64)
            a = im.crop(box)
            o = a.crop(defaultbox)
            o.save(os.path.join(path,target + "_%s.png" % k))
            k +=1
            img_list
    im.close()
    return img_list


def group(input_img, target, indices):
    im = Image.open(input_img)
    imgwidth, imgheight = im.size
    imBtest = Image.new("RGB", (imgwidth, imgheight), (255, 255, 255))
    imBtest.paste(im)
    imBtest.save(os.path.join("../datasets/public_web_fonts/", target, "B", "test", target + ".png"))
    imAtest = Image.new("RGB", (imgwidth, imgheight), (255,255,255))
    for index in indices:
        imBtrain = Image.open(os.path.join("../datasets/public_web_fonts/", target, target + "_%s.png" % index))
        imBtrain.save(os.path.join("../datasets/public_web_fonts/", target, "B", "train", target + "_%s.png" % index))
        imAtest.paste(imBtrain, (index*64, 0, index*64+64, 64))
    imAtest.save(os.path.join("../datasets/public_web_fonts/", target, "A", "test", target + ".png"))
    im_cover = Image.new("RGB", (imgheight, imgheight), (255, 255, 255))
    for index in indices:
        imAtrain = imAtest.copy()
        imAtrain.paste(im_cover, (index*64, 0,index*64+64,64))
        imAtrain.save(os.path.join("../datasets/public_web_fonts/", target, "A", "train", target + "_%s.png" % index))


load_global_charset()
parser = argparse.ArgumentParser(description='Convert font to dataset')
parser.add_argument('--base_font', default= "/usr/share/fonts/truetype/arphic/uming.ttc", help='path of the base font')
# parser.add_argument('--font', required=True, help='path of the target font')
parser.add_argument('--no_replace', action='store_true', help='no replace blanks')
parser.add_argument('--charset', dest='charset', type=str, default='EN', help='charset, can be either: EN, ZH, ZHt JP, KR or a one line file')
parser.add_argument('--sample_count', type=int, default=60, help='number of characters to draw; 0: no limit')
parser.add_argument('--sample_random', type=int, default=0, help='0: no random; 1:sample in order, shuffle; 2: random sample, keep order; 3: all random')
parser.add_argument('--char_size', type=int, default=48,
                    help='character size')
parser.add_argument('--canvas_x', type=int, default=64, help='canvas width')
parser.add_argument('--canvas_y', type=int, default=64, help='canvas height')
parser.add_argument('--x_offset', type=int, default=0, help='x offset')
parser.add_argument('--y_offset', type=int, default=0, help='y_offset')
# parser.add_argument('--dataset_dir', dest='sample_dir', help='directory to save examples')
parser.add_argument('--punctuation', action='store_true', help='includes punctuation')

args = parser.parse_args()

if __name__ == '__main__':
    datasets_dir = "/mnt/Auxiliary/XIauxiliary/XIdataset/font" \
                   "/English/"
    input_dir = os.path.join(datasets_dir,
                             "10 K FONTS For Designers By SandunLK")
    dataset_name = "SandunLK10k"

    fonts = []
    for r, d, f in os.walk(input_dir):
        for file in f:
            if '.TTF' in file:
                os.rename(os.path.join(r, file), os.path.join(r, file.replace('.TTF', '.ttf')))
            if ".ttf" in file:
                fonts.append(os.path.join(r, file))
    l = len(fonts)
    l_ = list(range(l))
    random.shuffle(l_)
    tests = [fonts[i] for i in l_[:int(l/10)]]
    trains = [fonts[i] for i in l_[int(l / 10):]]
    vals = [trains[i] for i in range(int(l / 10))]

    dataset_dir = os.path.join(datasets_dir, dataset_name)
    dir_test = os.path.join(dataset_dir, "test")
    dir_dicts = os.path.join(dataset_dir, "dicts")
    dir_train = os.path.join(dataset_dir, "train")
    dir_val = os.path.join(dataset_dir, "val")
    dirs = [dir_test, dir_dicts, dir_train, dir_val]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    if args.charset in ['EN', 'ENc', 'ENl']:
        charset = locals().get("%s_CHARSET" % args.charset)
        if args.punctuation:
            charset = charset + PUNC_CHARSET
    elif args.charset in ['ZH', 'ZHt', 'JP', 'KR']:
        charset = locals().get("%s_CHARSET" % args.charset)
        if args.punctuation:
            charset = charset + PUNCf_CHARSET
    else:
        charset = [c for c in open(args.charset).readline()[:-1].decode("utf-8")]
    l = len(charset)

    if args.sample_random > 1:
        if args.sample_count:
            l_ = random.sample(range(l), args.sample_count)
        if args.sample_random == 2:
            l_ = sorted(l_)
    else:
        if args.sample_count:
            l_ = range(min(l, args.sample_count))
        if args.sample_random == 2:
            l_ = random.sample(l_, args.sample_count)
    charset = [charset[i] for i in l_]

    fonts2dataset(DEFAULT_FONT, tests, charset, dir_test,
                  char_size=args.char_size,
                  canvas_x=args.canvas_x, canvas_y=args.canvas_y,
                  x_offset=args.x_offset, y_offset=args.y_offset,
                  replace_hashes=not args.no_replace)
    fonts2dataset(DEFAULT_FONT, trains, charset, dir_train,
                  char_size=args.char_size,
                  canvas_x=args.canvas_x, canvas_y=args.canvas_y,
                  x_offset=args.x_offset, y_offset=args.y_offset,
                  replace_hashes=not args.no_replace)
    fonts2dataset(DEFAULT_FONT, vals, charset, dir_val,
                  char_size=args.char_size,
                  canvas_x=args.canvas_x, canvas_y=args.canvas_y,
                  x_offset=args.x_offset, y_offset=args.y_offset,
                  replace_hashes=not args.no_replace)

