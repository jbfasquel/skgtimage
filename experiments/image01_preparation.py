import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti


input_dir="Database/image01/FullDetails/"
output_dir="Database/image01/truth"


image=sp.misc.imread(os.path.join(input_dir,"image.png"))
roi=sp.misc.imread(os.path.join(input_dir,"region_file.png"))
ref_roi=255*sp.ndimage.morphology.binary_dilation(roi,iterations=10).astype(np.uint8)


r_file=sp.misc.imread(os.path.join(input_dir,"region_file.png"))
r_text=sp.misc.imread(os.path.join(input_dir,"region_text.png"))
r_paper=sp.misc.imread(os.path.join(input_dir,"region_paper.png"))

image=skgti.utils.extract_subarray(image,ref_roi)
sp.misc.imsave(os.path.join(output_dir,"image.png"),image)
roi=skgti.utils.extract_subarray(roi,ref_roi)
sp.misc.imsave(os.path.join(output_dir,"roi.png"),roi)


new_r=r_file-r_paper
new_r=skgti.utils.extract_subarray(new_r,ref_roi)
sp.misc.imsave(os.path.join(output_dir,"region_file.png"),new_r)
new_r=r_paper-r_text
new_r=skgti.utils.extract_subarray(new_r,ref_roi)
sp.misc.imsave(os.path.join(output_dir,"region_paper.png"),new_r)
new_r=r_text
new_r=skgti.utils.extract_subarray(new_r,ref_roi)
sp.misc.imsave(os.path.join(output_dir,"region_text.png"),new_r)


'''
for i in ["file","text","paper"]:
    region=sp.misc.imread(os.path.join(input_dir,"region_"+str(i)+".png"))
    region=skgti.utils.extract_subarray(region,ref_roi)
    sp.misc.imsave(os.path.join(output_dir,"region_"+str(i)+".png"),region)
'''
