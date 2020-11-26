#=====================================
# JointFontGAN
# By Yankun Xi
#=====================================


import os


def get_root():
    this_path = os.path.dirname(os.path.realpath(__file__))
    print(this_path)
    # root = os.path.dirname(this_path) # relative path
    root = "".join(this_path.rpartition("JointFontGAN")[0:2]) # partitioned path
    project_root = this_path.rpartition("JointFontGAN")[0]
    # parent path for better module structure
    print(project_root)
    return project_root
