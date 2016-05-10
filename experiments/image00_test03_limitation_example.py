import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

truth_dir="Database/image00/test03/truth/"
save_dir="Database/image00/test03/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

# IMAGE
image=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 1.1, 1.1, 1.1, 0.0],
                [0.0, 1.0, 1.4, 1.0, 0.0, 1.1, 3.9, 1.1, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 1.1, 1.1, 1.1, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

label=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 2, 2, 2, 0],
                [0, 1, 4, 1, 0, 2, 3, 2, 0],
                [0, 1, 1, 1, 0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]])

# A PRIORI KNOWLEDGE
t_desc="B,C<A;D<B"
p_desc="A<B=C<D"

# TRUTH
B=np.where(label==1,1,0)
C=np.where(label==2,1,0)+np.where(label==3,1,0)
D=np.where(label==4,1,0)
A=np.ones(label.shape)-np.where(label>0,1,0)
if not os.path.exists(truth_dir) : os.mkdir(truth_dir)
sp.misc.imsave(truth_dir+"image.png",image)
sp.misc.imsave(truth_dir+"region_A.png",A)
sp.misc.imsave(truth_dir+"region_B.png",B)
sp.misc.imsave(truth_dir+"region_C.png",C)
sp.misc.imsave(truth_dir+"region_D.png",D)

# RECOGNITION
matcher=skgti.core.matcher_factory(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False)
matcher.compute_maching(True)
#skgti.io.save_matcher_details(matcher,image,label,None,save_dir,False)
#id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False,verbose=False)
skgti.io.save_matcher_details(matcher,image,label,None,save_dir,False)

# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
import helper
helper.influence_of_commonisos_refactorying(matcher,image,t_desc,p_desc,truth_dir,save_dir)
