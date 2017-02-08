import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

truth_dir="../../Database/image00/test06/truth/"
save_dir="../../Database/image00/test06/"


# IMAGE
image=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.1, 1.1, 1.1, 0.0],
                [0.5, 0.0, 1.1, 1.2, 1.1, 0.0],
                [0.0, 0.0, 1.1, 1.1, 1.1, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

label=np.array([[0, 0, 0, 0, 0, 0],
                [0, 0, 2, 2, 2, 0],
                [1, 0, 2, 3, 2, 0],
                [0, 0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0]])

# A PRIORI KNOWLEDGE
t_desc="C<B<A"
p_desc="A<B<C"
ref_t_graph = skgti.core.from_string(t_desc)
ref_p_graph = skgti.core.from_string(p_desc)


t_graph,p_graph=skgti.core.from_labelled_image(image,label)
'''
skgti.io.plot_graph(ref_p_graph)
plt.show()

skgti.io.plot_graph_with_regions(t_graph)
plt.show()

isos=skgti.core.find_subgraph_isomorphims(p_graph,skgti.core.from_string(p_desc))
print(isos)
skgti.io.plot_graph_links(p_graph,ref_p_graph,[skgti.io.matching2links(isos[0]),skgti.io.matching2links(isos[1])],['red','blue'])
plt.show();quit()
'''

#skgti.io.plot_graph_with_regions(p_graph)
#plt.show()
matching, common_isomorphisms, t_isomorphisms, p_isomorphisms, eie = skgti.core.best_common_subgraphisomorphism(t_graph,ref_t_graph,p_graph,ref_p_graph)

isos=skgti.core.find_subgraph_isomorphims(t_graph,skgti.core.from_string(t_desc))
print(isos)
# RECOGNITION
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False,verbose=False)
skgti.io.save_matcher_details(matcher,image,label,None,save_dir,False)

'''
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
skgti.core.merge_nodes_photometry(t_graph, merge[0], merge[1])
skgti.core.merge_nodes_topology(p_graph, merge[0], merge[1])
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
'''


