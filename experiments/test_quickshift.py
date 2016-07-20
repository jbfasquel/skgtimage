import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti
import os
import skimage; from skimage import segmentation

def compute_gradient(image):
    dx=sp.ndimage.filters.sobel(image, 0)
    dy=sp.ndimage.filters.sobel(image, 1)
    gradient=np.sqrt(dx**2+dy**2)
    return gradient



truth_dir="Database/image04/truth/"
image_rgb=sp.misc.imread(os.path.join(truth_dir,"image.png"))
image=skgti.utils.rgb2gray(image_rgb)
image_chsv=skgti.utils.rgb2chsv(image_rgb)
result=skimage.segmentation.quickshift(image_rgb,ratio=0.5)
plt.imshow(result);plt.show();quit()
