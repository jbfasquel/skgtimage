import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

#########
# MISC INFORMATIONS
#########
truth_dir="../../Database/image03/truth_top/"
save_dir="../../Database/image03/top_meanshift_5classes_nobounds/"

#########
# A PRIORI KNOWLEDGE
#########
t_desc="C,D<B<A;E,F<D;G<C;H<E"
p_desc="G=B<E=F<H=D<A=C"

#########
# IMAGE: COLOR AND GRAY
image_rgb=sp.misc.imread(os.path.join(truth_dir,"image.png"))
roi=sp.misc.imread(os.path.join(truth_dir,"roi.png"))
image=skgti.utils.rgb2gray(image_rgb)

#########
# MEANSHIFT ON COLOR IMAGE
image_chsv=skgti.utils.rgb2chsv(image_rgb)
label=skgti.utils.mean_shift(image_chsv,0.1,roi,True,True) #0.1 OK


t_graph,p_graph=skgti.core.from_labelled_image(image,label)

# RECOGNITION (with filtering==1 -> pb on eie -> the initial matching criteria lead to worst result than other ones)
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=False,thickness=2,filtering=4,verbose=True)
skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
skgti.io.pickle_matcher(matcher,save_dir+"matcher.pkl")

# EVALUATION VS TRUTH
import helper
classif,region2sim=helper.compared_with_truth(image,t_desc,p_desc,truth_dir,save_dir+"06_final",save_dir+"07_eval_classif/")
print("Evaluation of all regions vs truth: GCR = ", classif, " ; Similarities = " , region2sim)

# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
helper.influence_of_commonisos(matcher,image,t_desc,p_desc,truth_dir,save_dir)
