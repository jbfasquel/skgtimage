import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

######################################
#CROP
######################################
'''
input="Database/image02/rawdata_nocrop/"
output="Database/image02/truth/"

ref_roi=np.load(os.path.join(input,"region_liver.npy"))
image=np.load(os.path.join(input,"image_filtered.npy"))
image=skgti.utils.extract_subarray(image,ref_roi)
np.save(os.path.join(output,"image_filtered.npy"),image)
image=np.load(os.path.join(input,"image.npy"))
image=skgti.utils.extract_subarray(image,ref_roi)
np.save(os.path.join(output,"image.npy"),image)

for i in ['liver','vessel','tumor']:
    image=np.load(os.path.join(input,"region_"+i+".npy"))
    image=skgti.utils.extract_subarray(image,ref_roi)
    np.save(os.path.join(output,"region_"+i+".npy"),image)
'''


input="Database/image02/truth_filled/"
output="Database/image02/truth/"

liver=np.load(os.path.join(input,"region_liver.npy")).astype(np.uint8)
np.save(os.path.join(output,"roi.npy"),liver)
tumors=np.load(os.path.join(input,"region_tumor.npy")).astype(np.uint8)
vessels=255*(np.load(os.path.join(input,"region_vessel.npy")).astype(np.uint8))
print(np.min(tumors),np.max(tumors))
print(np.min(vessels),np.max(vessels))
print(np.min(liver),np.max(liver))
liver=liver-tumors-vessels
np.save(os.path.join(output,"region_liver.npy"),liver)
np.save(os.path.join(output,"region_vessel.npy"),vessels)
np.save(os.path.join(output,"region_tumor.npy"),tumors)

