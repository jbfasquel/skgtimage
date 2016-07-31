import os,pickle
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti


#########
# MISC INFORMATIONS
#########
truth_dir="../../Database/image03/truth_top/"
save_dir="../../Database/image03/top_meanshift_4classes/"

#########
# A PRIORI KNOWLEDGE
#########
t_desc="C,D<B<A;E,F<D;G<C;H<E"
p_desc="G=B<E=F<H=D<A=C"

#########
# IMAGE: COLOR AND GRAY
#########
#COLOR IMAGE
image_rgb=sp.misc.imread(os.path.join(truth_dir,"image.png"))
#ROI
roi=sp.misc.imread(os.path.join(truth_dir,"roi.png"))
#GRAYSCALE IMAGE
image=skgti.utils.rgb2gray(image_rgb)

# MEANSHIFT ON COLOR IMAGE
image_chsv=skgti.utils.rgb2chsv(image_rgb)
label=skgti.utils.mean_shift(image_chsv,0.12,roi,True,True) #0.1 OK
################
# RECOGNITION
################
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,filtering=1,verbose=True)
skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
skgti.io.pickle_matcher(matcher,save_dir+"matcher.pkl")


# EVALUATION VS TRUTH
import helper
classif,region2sim=helper.compared_with_truth(image,t_desc,p_desc,truth_dir,save_dir+"06_final",save_dir+"07_eval_classif/")
print("Evaluation of all regions vs truth: GCR = ", classif, " ; Similarities = " , region2sim)

# EVALUATION VS RAWSEGMENTATION
region2segmentintensities={'A':170,'B':85,'D':0,'C':170,'G':85,'E':85,'F':255,'H':0}
classif_result,classif_rawsegmentation=helper.compared_with_rawsegmentation(save_dir+"00_context/labelled_image.png",t_desc,p_desc,image,region2segmentintensities,save_dir+"06_final/",truth_dir,save_dir+"07_eval_vs_raw_seg/")
print("Raw segmentation vs truth: ",classif_rawsegmentation, "(proposed method GCR=",classif_result,")")


# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
helper.influence_of_commonisos(matcher,image,t_desc,p_desc,truth_dir,save_dir)
