import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

truth_dir="Database/image00/test02/truth/"
save_dir="Database/image00/test02/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

# IMAGE
image=np.array([[0.0, 0.2, 0.0, 0.0, 0.0],
                [0.2, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.1, 1.4, 0.9, 0.0],
                [0.0, 1.0, 1.3, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]])

label=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 4, 1, 0],
                [0, 1, 5, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])


t_desc="B<A"
p_desc="A<B"

rA=np.ones((6,5))*255
#rA[0,0]=0
rB=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])
if not os.path.exists(truth_dir) : os.mkdir(truth_dir)
sp.misc.imsave(truth_dir+"image.png",image)
sp.misc.imsave(truth_dir+"region_A.png",rA)
np.save(truth_dir+"region_A.npy",rA)
sp.misc.imsave(truth_dir+"region_B.png",rB)

##########
# VERSION 1
##########
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=1,verbose=False)
#id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=False,thickness=2,filtering=False,verbose=False)
skgti.io.save_matcher_details(matcher,image,label,None,save_dir,False)
#print(id2r['A'])
#print(np.array_equal(id2r['A'],rA))
#print(np.array_equal(id2r['B'],rB))

##########
# PLOT
##########
#skgti.io.plot_graph_with_regions_refactorying(matcher.relabelled_final_t_graph);plt.show()
#skgti.io.plot_graph_links(matcher.built_t_graph,matcher.ref_t_graph,[skgti.io.matching2links(matcher.matching),matcher.ordered_merges],['red','green']);plt.show()


# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
import helper
helper.influence_of_commonisos_refactorying(matcher,image,t_desc,p_desc,truth_dir,save_dir)
