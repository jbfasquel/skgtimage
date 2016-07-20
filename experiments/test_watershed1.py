import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti
import skimage; from skimage import exposure,segmentation

image=np.array([    [0, 0,  0,  0,  0,  0,   0,  0,  0,  0, 0],
                    [0, 0,  0,  0,  0,  0,   0,  0,  0,  0, 0],
                    [0, 0, 10, 10, 10,  0,  10, 10, 10,  0, 0],
                    [0, 0, 10,  8, 10,  0,  10,  9, 10,  0, 0],
                    [0, 0, 10,  8, 10,  0,  10,  9, 10,  0, 0],
                    [0, 0, 10, 10, 10,  0,  10, 10, 10,  0, 0],
                    [0, 0,  0,  0,  0,  0,   0,  0,  0,  0, 0],
                    [0, 0,  0,  0,  0,  0,   0,  0,  0,  0, 0]])



#gradient=np.round(sp.ndimage.filters.gaussian_gradient_magnitude(image,0)).astype(np.uint16)
#gx,gy=np.gradient(image)
dx=sp.ndimage.filters.sobel(image, 0,mode='nearest')
dy=sp.ndimage.filters.sobel(image, 1,mode='nearest')
gradient=np.sqrt(dx**2+dy**2)
print(gradient)

#a=skimage.feature.peak_local_max(gradient)
#a=skimage.exposure.histogram(gradient)
#print(a)
#quit()

gradient=np.round(gradient).astype(np.uint16)
print(gradient)
#plt.imshow(gradient);plt.show()

markers=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 2, 0, 3, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.int8)

not_th_gradient=np.where(gradient>10,0,1)
print(not_th_gradient)
markers,_=sp.ndimage.measurements.label(not_th_gradient)
print(markers)

#print(gradient)
w_image=sp.ndimage.measurements.watershed_ift(gradient, markers)
dx=sp.ndimage.filters.sobel(w_image, 0,mode='nearest')
dy=sp.ndimage.filters.sobel(w_image, 1,mode='nearest')
dx,dy=np.gradient(w_image)

gradient=np.round(np.sqrt(dx**2+dy**2)).astype(np.uint8)
print("grad:",dx)
diff_w_image=np.where(gradient != 0, 255,0)
print(diff_w_image)
print(w_image)

plt.subplot(131)
plt.imshow(image)
plt.subplot(132)
plt.imshow(markers)
plt.subplot(133)
plt.imshow(diff_w_image)
plt.show()