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
# IMAGE
image=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 0.0],
                [0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.1, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

label=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 2, 2, 2, 0],
                [0, 1, 3, 3, 3, 3, 3, 2, 0],
                [0, 1, 1, 1, 1, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]])

roi=np.where(label>=1,255,0)


root_dir="../../Database/image03/pose_4/"
seg_para=0
save_dir=root_dir+"downsampled3/results/Improvement"+str(seg_para)+"/"
if not os.path.exists(root_dir+"downsampled3/results/"): os.mkdir(root_dir+"downsampled3/results/")
if not os.path.exists(save_dir): os.mkdir(save_dir)



image=np.dstack((image,image,image))

#label=skgti.utils.mean_shift(image,seg_para,roi,True,True,sigma=None,rgb_convert=True) #0.09 OK
t,p=skgti.core.from_labelled_image(skgti.utils.rgb2gray(image),label,roi)


#######
#RAG
from skimage.future import graph
#label=label+1
#label=label.filled(0)
#plt.imshow(label);plt.show()
gray=skgti.utils.rgb2gray(image)
#gray=np.dstack((gray,gray,gray))
gray=np.dstack((gray,np.zeros(gray.shape),np.zeros(gray.shape)))
#g = graph.rag_mean_color(image, label)
g = graph.rag_mean_color(gray, label,mode='distance')
#tmp=graph.draw_rag(label,g,image);plt.imshow(tmp);plt.show()

label = graph.cut_threshold(label, g, 0.1)  # 30 OK, 50 mieux
label=np.ma.masked_array(label,mask=np.logical_not(roi))
t,p=skgti.core.from_labelled_image(skgti.utils.rgb2gray(image),label,roi)
skgti.io.save_graph(t,name='t',directory=save_dir+'g1')
skgti.io.save_graphregions(t,directory=save_dir+'g1')
skgti.io.save_intensities(p,directory=save_dir+'g1/')
skgti.io.with_graphviz.__save_image2d__(label,save_dir+'g1/label.png')

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