import os
import numpy as np
import scipy as sp;from scipy import misc
import skgtimage as skgti
import matplotlib.pyplot as plt


truth_dir="../../Database/image03/truth_bottom/"
save_dir="../../Database/image03/bottom_meanshift_5classes/"

# KNOWLEDGE
t_desc="E<D;G<F;D,F,H,I<C<B<A"
p_desc="B=F<D=H<I=E<C=A=G"
# IMAGE: COLOR AND GRAY
image_rgb=sp.misc.imread(os.path.join(truth_dir,"image.png"))
roi=sp.misc.imread(os.path.join(truth_dir,"roi.png"))
image=skgti.utils.rgb2gray(image_rgb)

# SEGMENTATION
image_chsv=skgti.utils.rgb2chsv(image_rgb)
label=skgti.utils.mean_shift(image_chsv,0.1,roi,True,True) #0.1 OK

# RECOGNITION
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,filtering=1,verbose=True)
skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
skgti.io.pickle_matcher(matcher,save_dir+"matcher.pkl")
matcher=skgti.io.unpickle_matcher(save_dir+"matcher.pkl")

# EVALUATION VS TRUTH
import helper
classif,region2sim=helper.compared_with_truth(image,t_desc,p_desc,truth_dir,save_dir+"06_final",save_dir+"07_eval_classif/")
print("Evaluation of all regions vs truth: GCR = ", classif, " ; Similarities = " , region2sim)
# EVALUATION VS RAWSEGMENTATION
region2segmentintensities={'A':0,'B':63,'C':0,'H':255,'D':127,'E':191,'F':63,'G':0,'I':191}
classif_result,classif_rawsegmentation=helper.compared_with_rawsegmentation(save_dir+"00_context/labelled_image.png",t_desc,p_desc,image,region2segmentintensities,save_dir+"06_final/",truth_dir,save_dir+"07_eval_vs_raw_seg/")
print("Raw segmentation vs truth: ",classif_rawsegmentation, "(proposed method GCR=",classif_result,")")
# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
helper.influence_of_commonisos(matcher,image,t_desc,p_desc,truth_dir,save_dir)
