import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

save_dir="Database/image00/test_refactorying/"
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

t_desc="B<A"
p_desc="A<B"

##########
# VERSION 1
##########
#id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False,verbose=False)
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=False,thickness=2,filtering=False,verbose=False)
##########
# VERSION 2
##########
'''
#Prepare matcher
#tmp_matcher=skgti.core.matcher_factory(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=True)
tmp_matcher=skgti.core.matcher_factory(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False)
#Matching
tmp_matcher.compute_maching(True)
#io
skgti.io.pickle_matcher(tmp_matcher,save_dir+"test.pkl")
matcher=skgti.io.unpickle_matcher(save_dir+"test.pkl")
#Merge
matcher.compute_merge()
#Final graph
matcher.update_final_graph()
id2r=matcher.get_id2regions()
'''
##########
# SAVING DETAILS
##########
skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)

##########
# PLOT
##########
#skgti.io.plot_graph_links(matcher.built_t_graph,matcher.ref_t_graph,[skgti.io.matching2links(matcher.matching),matcher.ordered_merges],['red','green']);plt.show()
#skgti.io.plot_graph_refactorying(m.ref_t_graph);plt.show()
#skgti.io.plot_graph_with_regions_refactorying(m.ref_t_graph);plt.show()
#skgti.io.plot_graph_with_regions_refactorying(m.built_t_graph,2);plt.show()
