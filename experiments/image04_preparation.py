import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

truth_dir="Database/image04/truth2/"
save_dir="Database/image04/meanshift/"


tmp=sp.misc.imread(truth_dir+"region_car.png")
for r in ['glass', 'glassboundary']:
    tmp-=sp.misc.imread(truth_dir+"region_"+r+".png")

#plt.imshow(tmp);plt.show();quit()

