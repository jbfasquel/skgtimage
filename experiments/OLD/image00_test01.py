import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

save_dir="Database/image00/test01/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

# IMAGE
image=np.array([[0.0, 0.2, 0.0, 0.0, 0.0],
                [0.2, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.1, 1.4, 0.9, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]])

label=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 4, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])


roi=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])

t_desc="C<B<A"
p_desc="A<B<C"

rA=np.ones((5,5))
rB=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])

##########
# VERSION 1
##########
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False,verbose=False)
#id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=False,thickness=2,filtering=False,verbose=False)
skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
print(id2r['A'])
print(np.array_equal(id2r['A'],rA))
print(np.array_equal(id2r['B'],rB))

##########
# PLOT
##########
skgti.io.plot_graph_with_regions(matcher.relabelled_final_t_graph);plt.show()
#skgti.io.plot_graph_links(matcher.built_t_graph,matcher.ref_t_graph,[skgti.io.matching2links(matcher.matching),matcher.ordered_merges],['red','green']);plt.show()
