import os
import pickle
import argparse
import random

IMG_EXTENSIONS = ['.png']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


parser = argparse.ArgumentParser(description='Creating test pickles')
parser.add_argument('--data_dir', type=str,
                    default="/mnt/Auxiliary/XIauxiliary/XIdataset"
                            "/font/English/", help='dataset folder')
parser.add_argument('--dataset', default='SandunLK10k64',
                    help='dataset to images, Capitals64, SandunLK10k64')
parser.add_argument('--grps', type=int,
                    default=60, help='number of characters')
args = parser.parse_args()

data_dir = os.path.join(args.data_dir, args.dataset)
test_dir = os.path.join(data_dir, 'test')
dict_dir = os.path.join(data_dir, 'test_dict')
# os.system("mkdir %s" % dict_dir)
dict_inds = {}
for root, _, fnames in sorted(os.walk(data_dir)):
    for fname in fnames:
        if is_image_file(fname):
            path = os.path.join(root, fname)
            r = [x for x in range(60)]
            random.shuffle(r)
            dict_inds[fname] = r
with open(os.path.join(dict_dir, 'dict.pkl'), 'wb') as f:
    pickle.dump(dict_inds, f, protocol=0)


for font in dict_inds_:
    dict_inds_[font] = list(dict_inds_[font])