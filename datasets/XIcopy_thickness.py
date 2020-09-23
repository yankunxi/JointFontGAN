from PIL import Image
import os
import random

test_dir = "/mnt/Files/XIremote/OneDrive - Wayne State University/XIdataset/" \
           "font/English/Capitals64_thin/test"
train_dir = "/mnt/Files/XIremote/OneDrive - Wayne State University/XIdataset/" \
           "font/English/Capitals64_thin/train"
val_dir = "/mnt/Files/XIremote/OneDrive - Wayne State University/XIdataset/" \
           "font/English/Capitals64_thin/val"
files = os.listdir(test_dir)
rate = 0
num = 0
for png in files:
    png = os.path.join(test_dir, png)
    image = Image.open(png)
    width, height = image.size
    bg = 255
    bg_max = 200
    bg_max_count = 0
    for n, c in image.getcolors():
        if c < bg_max:
            bg_max_count += n
    image.close()
    bg_max_rate = bg_max_count * 1.0 / (width * height)
    if bg_max_rate > 0.2:
        os.remove(png)
    rate += bg_max_rate
    num += 1
print(rate / num)
files = os.listdir(train_dir)
rate = 0
num = 0
for png in files:
    png = os.path.join(train_dir, png)
    image = Image.open(png)
    width, height = image.size
    bg = 255
    bg_max = 200
    bg_max_count = 0
    for n, c in image.getcolors():
        if c < bg_max:
            bg_max_count += n
    image.close()
    bg_max_rate = bg_max_count * 1.0 / (width * height)
    if bg_max_rate > 0.2:
        os.remove(png)
    rate += bg_max_rate
    num += 1
print(rate / num)
files = os.listdir(val_dir)
rate = 0
num = 0
for png in files:
    png = os.path.join(val_dir, png)
    image = Image.open(png)
    width, height = image.size
    bg = 255
    bg_max = 200
    bg_max_count = 0
    for n, c in image.getcolors():
        if c < bg_max:
            bg_max_count += n
    image.close()
    bg_max_rate = bg_max_count * 1.0 / (width * height)
    if bg_max_rate > 0.2:
        os.remove(png)
    rate += bg_max_rate
    num += 1
print(rate / num)
