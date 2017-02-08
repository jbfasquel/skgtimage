import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti
import skimage; from skimage import filters
import helper

#########
# MISC INFORMATIONS
#########
root_dir="../../Database/image03/pose_4/"
input_dir=root_dir+"downsampled3/truth/"
seg_para=0.09
save_dir=root_dir+"downsampled3/results/Improvement"+str(seg_para)+"/"
if not os.path.exists(root_dir+"downsampled3/results/"): os.mkdir(root_dir+"downsampled3/results/")
if not os.path.exists(save_dir): os.mkdir(save_dir)

#########
# A PRIORI KNOWLEDGE
#########
t_desc="1C,1D<1B<1A;1F<1C;1G,1E<1D;1H<1E"
p_desc="1G=1B=1E=1F<1H=1D<1A=1C"

#########
# IMAGE + ROI
#########
#plt.imshow(roi);plt.show();quit()
image=sp.misc.imread(os.path.join(input_dir,"image.png"))
roi=skgti.core.fill_region(sp.misc.imread(os.path.join(input_dir,"region_1A.png")))

#########
# OVERSEGMENTATION
#########
label=skgti.utils.mean_shift(image,seg_para,roi,True,True,sigma=None,rgb_convert=True) #0.09 OK
t,p=skgti.core.from_labelled_image(skgti.utils.rgb2gray(image),label,roi)


t_ref=skgti.core.from_string(t_desc)
p_ref=skgti.core.from_string(p_desc)
cisos=skgti.core.common_subgraphisomorphisms_optimized_v2([t,p],[t_ref,p_ref])
print(cisos)
skgti.io.save_graph(t,name='t',directory=save_dir+'g0')
skgti.io.save_graph(p,name='p',directory=save_dir+'g0')
skgti.io.save_graphregions(t,directory=save_dir+'g0')
skgti.io.save_intensities(p,directory=save_dir+'g0/')
skgti.io.with_graphviz.__save_image2d__(label,save_dir+'g0/label.png')
skgti.io.save_image2d_boundaries(image,label,save_dir,"before")

#######
#RAG
label=skgti.core.rag_merge(skgti.utils.rgb2gray(image),label,58,roi)
t,p=skgti.core.from_labelled_image(skgti.utils.rgb2gray(image),label,roi)
cisos=skgti.core.common_subgraphisomorphisms_optimized_v2([t,p],[t_ref,p_ref])
print(cisos)

skgti.io.save_graph(t,name='t',directory=save_dir+'g1')
skgti.io.save_graph(p,name='p',directory=save_dir+'g1')
skgti.io.save_graphregions(t,directory=save_dir+'g1')
skgti.io.save_intensities(p,directory=save_dir+'g1/')
skgti.io.with_graphviz.__save_image2d__(label,save_dir+'g1/label.png')
skgti.io.save_image2d_boundaries(image,label,save_dir,"after")

#sp.misc.imsave(save_dir+'g0/label.png',label)


#plt.imshow(label);plt.show();quit()
#########
# RECOGNITION
#########

'''
try :
    #id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,verbose=True,background=False,mc=True) #7 labels
    id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,verbose=True,background=False,mc=True) #7 labels
    #skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False,mc=True)
    skgti.io.save_matcher_result(matcher,image,label,roi,save_dir,mc=True)
except skgti.core.matcher_exception as m_e:
    print('Matcher exception:', m_e.matcher)
    print("impossible to finalize")
    quit()
# EVALUATION VS TRUTH
import helper
truth_dir=root_dir+"downsampled3/truth/"
classif,region2sim=helper.compared_with_truth(image,t_desc,p_desc,truth_dir,save_dir+"06_final",save_dir+"07_eval_classif/",mc=True)
print("Evaluation of all regions vs truth: GCR = ", classif, " ; Similarities = " , region2sim)

# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
#helper.influence_of_commonisos(matcher,image,t_desc,p_desc,truth_dir,save_dir,mc=True)
'''