import os,time
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti
#import skimage; from skimage import filters
import helper

def median_filtering(image,size=3):
    image_r=image[:,:,0]
    image_r=sp.ndimage.filters.median_filter(image_r, size)
    #image_r=sp.ndimage.filters.uniform_filter(image_r.astype(np.float), size).astype(np.uint8)
    image_g=image[:,:,1]
    image_g=sp.ndimage.filters.median_filter(image_g, size)
    #image_g=sp.ndimage.filters.uniform_filter(image_g.astype(np.float), size).astype(np.uint8)
    image_b=image[:,:,2]
    image_b=sp.ndimage.filters.median_filter(image_b, size)
    #image_b=sp.ndimage.filters.uniform_filter(image_b.astype(np.float), size).astype(np.uint8)
    #image=np.concatenate((image_r,image_g,image_b), axis=2)
    image=np.dstack((image_r,image_g,image_b))
    return image

truth_dir="../../Database/image03/pose_1/downsampled3/truth/"
image_rgb=sp.misc.imread(os.path.join(truth_dir,"image.png"))
root_save_dir="../../Database/image03/results_pose1/Levelsets_Region_A/";
if not os.path.exists(root_save_dir): os.mkdir(root_save_dir)

#image=scipy.misc.imread(os.path.join(segmentation_dir,root_name+"_filtered.jpeg"))
#tmp_image=median_filtering(image_rgb, 3)
#image=skgti.utils.rgb2gray(tmp_image)
image=skgti.utils.rgb2gray(image_rgb)
#Noise removal
image_smooth = ndimage.gaussian_filter(image.astype(np.float), sigma=3.0)
#plt.imshow(image_smooth);plt.show();quit()

Iy, Ix = np.gradient(image_smooth)
f = Ix ** 2 + Iy ** 2
# Level set parameters
g = 1. / (1. + f)  # edge indicator function.
epsilon = 1.5  # the paramater in the definition of smoothed Dirac function 1.5
timestep = 10  # time step 5
mu = 0.2 / timestep  # coefficient of the internal (penalizing) energy term P(\phi) - Note: the product timestep*mu must be less than 0.25 for stability!
lam = 5  # coefficient of the weighted length term Lg(\phi) 5
alf = 3  # coefficient of the weighted area term Ag(\phi); 3 ; # Note: Choose a positive(negative) alf if the initial contour is outside(inside) the object.
# Level set initialization: "contour" initial
c0 = 4
initialLSF = c0 * np.ones(image.shape)
w = 10
initialLSF[w:image.shape[0]-w, w:image.shape[1]-w]=-c0
u = initialLSF

#DEFORMATION
plt.ion()
for n in range(4000):
    u=skgti.utils.evolution(u, g ,lam, mu, alf, epsilon, timestep, 1)
    if np.mod(n,200)==0:
        print(n)
        #plot_u(u)
        plt.imshow(image_rgb)
        plt.hold(True)
        cs = plt.contour(u,0, colors='r',linewidths=2.0)
        plt.draw()
        #plt.show();quit()
        time.sleep(1)
        plt.hold(False)
        plt.savefig(os.path.join(root_save_dir,"image_contour_"+str(n)+".svg"),format="svg",bbox_inches='tight')
        segmented_image=np.where(u>=0.0,0,255).astype(np.uint8)
        sp.misc.imsave(os.path.join(root_save_dir,"image_segmentation.png"),segmented_image)