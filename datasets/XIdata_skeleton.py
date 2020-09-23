from PIL import Image
import os
import random
from skimage import io
from skimage.morphology import skeletonize
from skimage.util import invert

test_dir = "/mnt/Auxiliary/XIauxiliary/XIdataset/font/English" \
           "/Capitals64/val"
# train_dir = "/mnt/Files/XIremote/OneDrive - Wayne State University/XIdataset/" \
#            "font/English/Capitals64/train"
# val_dir = "/mnt/Files/XIremote/OneDrive - Wayne State University/XIdataset/" \
#            "font/English/Capitals64/val"
testSKdir = "/mnt/Auxiliary/XIauxiliary/XIdataset/font/English" \
            "/Capitals64/valSK0"

imgs = os.listdir(test_dir)

mode = 0
finesize = 64
for png in imgs:
    png_dir = os.path.join(test_dir, png)
    pngSKdir = os.path.join(testSKdir, png)
    image = io.imread(png_dir, as_gray=True)
    image = image > 0
    if mode:
        image = invert(image)
    width, height = image.shape

    # img_matrix = []
    for i in range(0, height, finesize):
        img_list = []
        for j in range(0, width, finesize):
            # box = (j, i, j + finesize, i + finesize)
            a = image[j:(j+finesize), i:(i+finesize)]
            b = skeletonize(a)
            image[j:(j + finesize), i:(i + finesize)] = b
            # defaultbox = (0,0,64,64)
            # o = a.crop(defaultbox)
            # img_list = img_list + [a]
        # img_matrix = img_matrix + [img_list]
    image = invert(image)
    im = Image.fromarray(image)
    im.save(pngSKdir)

