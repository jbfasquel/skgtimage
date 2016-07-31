import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

######################################
#NEW CROP
######################################
input="../../Database/image02/rawdata_nocrop/"
output="../../Database/image02/truth/"

ref_roi=np.load(os.path.join(input,"region_liver.npy"))
ref_roi=sp.ndimage.morphology.binary_dilation(ref_roi,iterations=5).astype(np.uint8)
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



######################################
#CROP
######################################
'''
input="../../Database/image02/rawdata_nocrop/"
output="../../Database/image02/truth/"

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
'''
truth_dir="Database/image02/truth/"
image=np.load(os.path.join(truth_dir,"image_filtered.npy"))
t_desc="tumor,vessel<liver"
p_desc="tumor<liver<vessel"

t_graph,p_graph=skgti.io.from_dir(t_desc,p_desc,image,truth_dir)
#plt.imshow(p_graph.get_image());plt.show()
skgti.io.plot_graph_histogram(t_graph,p_graph,True)#;plt.show()
plt.savefig(truth_dir+"histograms.svg",format="svg",bbox_inches='tight')
plt.savefig(truth_dir+"histograms.png",format="png",bbox_inches='tight')
plt.gcf().clear()
