import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti


root_name="image01"
input_dir="Database/image01_FULL/"
output_dir="Database/image01/"


image=sp.misc.imread(os.path.join(input_dir,root_name+".png"))
downsampling=3
image=image[::downsampling,::downsampling]
sp.misc.imsave(os.path.join(output_dir,root_name+".png"),image)

for i in range(0,5):
    region=sp.misc.imread(os.path.join(input_dir,root_name+"_region"+str(i)+".png"))
    region=region[::downsampling,::downsampling]
    sp.misc.imsave(os.path.join(output_dir,root_name+"_region"+str(i)+".png"),region)

