import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

#########
# MISC INFORMATIONS
#########
truth_dir="Database/image03/truth_top/"
save_dir="Database/image03/top_meanshift_5classes_eie2_filtering1/"

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


# RECOGNITION (with filtering==1 -> pb on eie -> the initial matching criteria lead to worst result than other ones)
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=True,thickness=2,filtering=1,verbose=True)
skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
skgti.io.pickle_matcher(matcher,save_dir+"matcher.pkl")
'''
matcher=skgti.io.unpickle_matcher(save_dir+"matcher.pkl")
eie_per_iso=[]
for c_iso in matcher.common_isomorphisms:
    eie_per_iso+=[skgti.core.costiso(matcher.query_p_graph,matcher.ref_p_graph,c_iso)]
print(eie_per_iso)
quit()
'''
# EVALUATION VS TRUTH
import helper
classif,region2sim=helper.compared_with_truth(image,t_desc,p_desc,truth_dir,save_dir+"06_final",save_dir+"07_eval_classif/")
print("Evaluation of all regions vs truth: GCR = ", classif, " ; Similarities = " , region2sim)

# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
helper.influence_of_commonisos_refactorying(matcher,image,t_desc,p_desc,truth_dir,save_dir)
