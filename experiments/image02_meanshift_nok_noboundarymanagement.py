import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti

truth_dir="../../Database/image02/truth/"
save_dir="../../Database/image02/meanshift_bandwidth15_nok_noboundarymanagement/"


# A PRIORI KNOWLEDGE
t_desc="tumor,vessel<liver"
p_desc="tumor<liver<vessel"

# LOAD DATA
image=np.load(truth_dir+"image_filtered.npy")
roi=np.load(truth_dir+"roi.npy")


# SEGMENTATION MEANSHIFT
label=skgti.utils.mean_shift(image,15,roi,False,True)
# RECOGNITION: no boundary management
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=False,verbose=True)
skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,True,slices=[45])

# EVALUATION VS TRUTH
import helper
classif,region2sim=helper.compared_with_truth(image,t_desc,p_desc,truth_dir,save_dir+"06_final",save_dir+"07_eval_classif/",slices=[45])
print("Evaluation of all regions vs truth: GCR = ", classif, " ; Similarities = " , region2sim)

# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
#helper.influence_of_commonisos(matcher,image,t_desc,p_desc,truth_dir,save_dir,slices=[45])
