from PIL import Image
import os


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


if __name__ == '__main__':
    # input image
    input_img = "../datasets/EN_fonts/test/Bauchaomaicha.0.0.png"
    # dataset name
    target = os.path.splitext(os.path.basename(input_img))[0]
    #target = "1Ichiro"
    #name = os.path.splitext(os.path.basename(input_img))[0]
    IMG_EXTENSIONS = ['.png']
    path = os.path.join("../datasets/public_web_fonts/", target)
    pathA = os.path.join(path, "A")
    pathAtrain = os.path.join(pathA, "train")
    pathAtest = os.path.join(pathA, "test")
    pathB = os.path.join(path, "B")
    pathBtrain = os.path.join(pathB, "train")
    pathBtest = os.path.join(pathB, "test")

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(pathA):
        os.makedirs(pathA)

    if not os.path.exists(pathAtrain):
        os.makedirs(pathAtrain)

    if not os.path.exists(pathAtest):
        os.makedirs(pathAtest)

    if not os.path.exists(pathB):
        os.makedirs(pathB)

    if not os.path.exists(pathBtrain):
        os.makedirs(pathBtrain)

    if not os.path.exists(pathBtest):
        os.makedirs(pathBtest)

    split(path, input_img, 64, 64, 0)

    group(input_img, target, (1,6,9,13,19,21,24))

