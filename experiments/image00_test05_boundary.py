import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

truth_dir="../../Database/image00/test05/truth/"
save_dir="../../Database/image00/test05/"


# IMAGE
image=np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.1, 1.1, 1.1, 0.0],
                [0.5, 1.1, 1.2, 1.1, 0.0],
                [0.0, 1.1, 1.1, 1.1, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]])

label=np.array([[0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0],
                [1, 2, 3, 2, 0],
                [0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0]])

# A PRIORI KNOWLEDGE
t_desc="C<B<A"
p_desc="A<B<C"

t_graph,p_graph=skgti.core.from_labelled_image(image,label)
###############
#No topo iso
###############
isos=skgti.core.find_subgraph_isomorphims(t_graph,skgti.core.from_string(t_desc))
print(isos)
###############
#Merge the two most similar heads
###############
remaining_nodes=skgti.core.search_head_nodes(t_graph)
ordered_merging_candidates,d2m=skgti.core.cost2merge(t_graph,p_graph,remaining_nodes,remaining_nodes)
print(d2m)
skgti.io.plot_graph_with_regions(t_graph)
plt.show()
merge=ordered_merging_candidates[0]
skgti.core.merge_nodes_photometry(p_graph, merge[0], merge[1])
skgti.core.merge_nodes_topology(t_graph, merge[0], merge[1])
###############
#One topo iso found (after merge)
###############
isos=skgti.core.find_subgraph_isomorphims(t_graph,skgti.core.from_string(t_desc))
print(isos)
remaining_nodes=skgti.core.search_head_nodes(t_graph)
ordered_merging_candidates,d2m=skgti.core.cost2merge(t_graph,p_graph,remaining_nodes,remaining_nodes)
print(ordered_merging_candidates)
skgti.io.plot_graph_with_regions(t_graph)
plt.show()
quit()


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


tg,pg=skgti.io.from_dir(t_desc,p_desc,image,truth_dir)


# RECOGNITION
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False,verbose=False)
skgti.io.save_matcher_details(matcher,image,label,None,save_dir,False)

# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
import helper
helper.influence_of_commonisos_refactorying(matcher,image,t_desc,p_desc,truth_dir,save_dir)

# IMAGE
save_dir="Database/image00/test04/v2/"
image=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.9, 0.9, 0.9, 0.0],
                [0.0, 1.0, 1.4, 1.0, 0.0, 0.9, 1.2, 0.9, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.9, 0.9, 0.9, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False,verbose=False)
skgti.io.save_matcher_details(matcher,image,label,None,save_dir,False)
# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
import helper
helper.influence_of_commonisos_refactorying(matcher,image,t_desc,p_desc,truth_dir,save_dir)
