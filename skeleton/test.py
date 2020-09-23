from skimage.morphology import skeletonize
from skimage import data
from skimage import io
import matplotlib.pyplot as plt
from skimage.util import invert

# Invert the horse image
image = invert(io.imread('../../ab.png', as_gray=True))
# image = invert(data.horse())

# perform skeletonization
skeleton = skeletonize(image)

# plt.savefig("temp.png")

# display results
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(100, 20),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.savefig("temp.png")
