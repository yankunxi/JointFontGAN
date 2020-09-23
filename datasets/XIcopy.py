from PIL import Image
import os

if __name__ == '__main__':
    source = "../datasets/Capitals64/test/1_Ichiro_0_0.png"
    IMG_EXTENSIONS = ['.png']
    name = os.path.splitext(os.path.basename(source))[0]
    path = os.path.join("../datasets/public_web_fonts/", name)
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

    split(path, input_img, 64, 64, 0, name)

    group(name, (1,5,6,16,23))