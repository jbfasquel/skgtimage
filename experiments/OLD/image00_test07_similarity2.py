import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

truth_dir="../../Database/image00/test07_bis/truth/"
save_dir="../../Database/image00/test07_bis/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

# IMAGE
image=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                [0.0, 1.0, 1.0, 1.0, 0.1, 0.0 ],
                [0.0, 1.0, 1.4, 1.0, 1.5, 0.0 ],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0 ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]])

label=np.array([[0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 4, 0],
                [0, 1, 3, 1, 2, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0]])

# A PRIORI KNOWLEDGE
t_desc="B,C<A;D<B";ref_t_graph=skgti.core.from_string(t_desc)
p_desc="A<B<C=D";ref_p_graph=skgti.core.from_string(p_desc)
#skgti.io.plot_graph(ref_t_graph)
#skgti.io.plot_graph(ref_p_graph)
#plt.show()

# TRUTH
'''
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
'''
# BUILD GRAPH FROM LABELLED IMAGE
t_graph,p_graph=skgti.core.from_labelled_image(image,label)

#skgti.io.plot_graph_with_regions(t_graph)
#skgti.io.plot_graph_with_regions(p_graph)
#plt.show()

# FIND COMMON ISO BRUTE FORCE
common_isomorphisms,isomorphisms_per_graph=skgti.core.common_subgraphisomorphisms([t_graph,p_graph],[ref_t_graph,ref_p_graph])
print("**** COMMON ISO ****")
print(common_isomorphisms)
print("**** T ISO ****")
print(isomorphisms_per_graph[0])
print("**** P ISO ****")
print(isomorphisms_per_graph[1])

# FIND COMMON ISO OPTIMIZED
common_isomorphisms2=skgti.core.common_subgraphisomorphisms_optimized([t_graph,p_graph],[ref_t_graph,ref_p_graph])
print("**** COMMON ISO ****")
print(common_isomorphisms2)

#id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,verbose=True,bf=True)
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,verbose=True,bf=False)
skgti.io.save_matcher_details(matcher,image,label,None,save_dir,False)


equality=True
for e in common_isomorphisms:
    if e not in common_isomorphisms2: equality=False
print("Equality: ",equality)