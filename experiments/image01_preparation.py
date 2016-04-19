import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti


input_dir="Database/image01/FullDetails/"
output_dir="Database/image01/truth"


image=sp.misc.imread(os.path.join(input_dir,"image.png"))
ref_roi=sp.misc.imread(os.path.join(input_dir,"region_file.png"))
ref_roi=255*sp.ndimage.morphology.binary_dilation(ref_roi,iterations=10).astype(np.uint8)

image=skgti.utils.extract_subarray(image,ref_roi)
sp.misc.imsave(os.path.join(output_dir,"image.png"),image)

for i in ["file","text","paper"]:
    region=sp.misc.imread(os.path.join(input_dir,"region_"+str(i)+".png"))
    region=skgti.utils.extract_subarray(region,ref_roi)
    sp.misc.imsave(os.path.join(output_dir,"region_"+str(i)+".png"),region)

