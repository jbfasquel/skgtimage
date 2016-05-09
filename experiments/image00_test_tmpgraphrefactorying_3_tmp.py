import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

truth_dir="Database/image00/test_tmp/truth/"
save_dir="Database/image00/test_tmp/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

# IMAGE
image=np.array([[1,1,1,1,1,1,1],
                [1,2,2,2,1,1,1],
                [1,2,3,2,1,3.1,1],
                [1,2,3,2,1,3.1,1],
                [1,2,2,2,1,1,1],
                [1,1,1,1,1,1,1]],np.float)
label=np.array([[1,1,1,1,1,1,1],
                [1,2,2,2,1,1,1],
                [1,2,3,2,1,4,1],
                [1,2,3,2,1,4,1],
                [1,2,2,2,1,1,1],
                [1,1,1,1,1,1,1]],np.float)
# KNOWLEDGE
t_desc="C<B<A;D<A"
p_desc="A<B<C=D"

#skgti.io.plot_graph_refactorying(skgti.core.from_string(t_desc));plt.show();quit()
#skgti.io.plot_graph_refactorying(skgti.core.from_string(p_desc));plt.show();quit()
'''
built_t_graph,res=skgti.core.topological_graph_from_labels(label.astype(np.uint8))
built_p_graph=skgti.core.photometric_graph_from_residues_refactorying(image,res)
plt.subplot(121)
skgti.io.plot_graph_refactorying(built_t_graph)
plt.subplot(122)
skgti.io.plot_graph_refactorying(built_p_graph);plt.show();quit()
'''
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False,verbose=True)
skgti.io.plot_graph_with_regions(matcher.relabelled_final_t_graph);plt.show()
