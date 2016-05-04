import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti

truth_dir="Database/image01/truth/"
save_dir="Database/image01/meanshift_4classes_vs_3expected/"


t_desc="text<paper<file"
p_desc="text<file<paper"

image=sp.misc.imread(truth_dir+"image.png")
roi=sp.misc.imread(truth_dir+"roi.png")

# SEGMENTATION MEANSHIFT

label=skgti.utils.mean_shift(image,10,roi,False,True)
# RECOGNITION
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=True,thickness=2,filtering=False,verbose=True)

skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
skgti.io.pickle_matcher(matcher,save_dir+"matcher.pkl")


# EVALUATION
import helper
classif,region2sim=helper.compared_with_truth(image,t_desc,p_desc,truth_dir,save_dir+"06_final",save_dir+"07_eval_classif/")
print("Evaluation of all regions vs truth: GCR = ", classif, " ; Similarities = " , region2sim)

# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
helper.influence_of_commonisos_refactorying(matcher,image,t_desc,p_desc,truth_dir,save_dir)
