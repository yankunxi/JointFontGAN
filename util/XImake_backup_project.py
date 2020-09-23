if __package__ is None:
    import os
    import sys
    this_path = os.path.dirname(os.path.realpath(__file__))
    project_root = this_path.rpartition("xifontgan")[0]
    sys.path.insert(0, project_root)
    from xifontgan.get_root import get_root
    import shutil
else:
    from ..get_root import get_root
    import shutil


print(__name__)
print(__package__)

print(get_root())
# shutil.make_archive()
