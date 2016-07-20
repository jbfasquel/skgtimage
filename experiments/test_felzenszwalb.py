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
#result=skimage.segmentation.slic(image_rgb,n_segments=100,compactness=0.1,multichannel=True)
result=skimage.segmentation.felzenszwalb(image_rgb,scale=15)
#result=skimage.segmentation.slic(image_chsv,n_segments=100,compactness=0.1,multichannel=True)
plt.imshow(result);plt.show();quit()
#gradient=np.round(sp.ndimage.filters.gaussian_gradient_magnitude(image,0)).astype(np.uint16)

gx,gy=np.gradient(image)
gradient=np.sqrt(gx**2+gy**2)
#gradient=compute_gradient(image)
#plt.imshow(gradient,"gray");plt.show()
#print(gradient)
gradient=np.round(gradient).astype(np.uint16)

#print(gradient)
#plt.imshow(gradient);plt.show()

#plt.imshow(np.where(gradient>100,255,0),"gray");plt.show()

not_th_gradient=np.where(gradient<10,1,0)
#plt.imshow(not_th_gradient,"gray");plt.show();quit()
#print(not_th_gradient)
markers,nb=sp.ndimage.measurements.label(not_th_gradient)

print(nb)
#print(markers)
#plt.imshow(markers,"gray");plt.show();quit()
#print(gradient)
w_image=sp.ndimage.measurements.watershed_ift(gradient, markers)
gx,gy=np.gradient(w_image)
diff_w_image=np.round(np.sqrt(gx**2+gy**2)).astype(np.uint8)

#gradient=np.round(compute_gradient(w_image)).astype(np.uint8)
diff_w_image=np.where(gradient > 10, 255,0)

print(w_image)

plt.subplot(121)
plt.imshow(w_image,"gray")
plt.subplot(122)
plt.imshow(diff_w_image,"gray")
plt.show()