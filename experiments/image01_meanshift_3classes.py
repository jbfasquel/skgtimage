import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti

truth_dir="../../Database/image01/truth/"
save_dir="../../Database/image01/meanshift_3classes_nobound/"


t_desc="text<paper<file"
p_desc="text<file<paper"

image=sp.misc.imread(truth_dir+"image.png")
roi=sp.misc.imread(truth_dir+"roi.png")

# SEGMENTATION MEANSHIFT

label=skgti.utils.mean_shift(image,15,roi,False,True)
# RECOGNITION
#id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=True,thickness=2,filtering=0,verbose=True)
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=False,thickness=2,filtering=0,verbose=True)

skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
skgti.io.pickle_matcher(matcher,save_dir+"matcher.pkl")


# EVALUATION
import helper
classif,region2sim=helper.compared_with_truth(image,t_desc,p_desc,truth_dir,save_dir+"06_final",save_dir+"07_eval_classif/")
print("Evaluation of all regions vs truth: GCR = ", classif, " ; Similarities = " , region2sim)
# EVALUATION VS RAWSEGMENTATION
region2segmentintensities={'text':255,'paper':127,'file':0}
classif_result,classif_rawsegmentation=helper.compared_with_rawsegmentation(save_dir+"00_context/labelled_image.png",t_desc,p_desc,image,region2segmentintensities,
                                                                                         save_dir+"06_final/",truth_dir,save_dir+"07_eval_vs_raw_seg/")
print("Raw segmentation vs truth: ",classif_rawsegmentation, "(proposed method GCR=",classif_result,")")

# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
helper.influence_of_commonisos(matcher,image,t_desc,p_desc,truth_dir,save_dir)



