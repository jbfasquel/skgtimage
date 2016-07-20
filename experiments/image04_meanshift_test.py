import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

truth_dir="Database/image04/truth/"
save_dir="Database/image04/meanshift_test/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

#Initial a priori knowledge
t_desc="glass<glassboundary<carboundary;rims,car<carboundary<background"
p_desc="glassboundary=carboundary<background<car<glass=rims"



image_rgb=sp.misc.imread(os.path.join(truth_dir,"image.png"))
image=skgti.utils.rgb2gray(image_rgb)
sp.misc.imsave("image_gray.png",image)
image_chsv=skgti.utils.rgb2chsv(image_rgb)
#label=skgti.utils.mean_shift(image_chsv,0.12,None,True,True) #0.1 OK
label=skgti.utils.mean_shift(image_chsv,0.05,None,True,True) #0.1 OK

'''
t_graph,p_graph=skgti.io.from_dir2(truth_dir,True)
#skgti.io.plot_graph_histogram(t_graph,p_graph);plt.show()
skgti.io.save_graph(t_graph,name="topological",directory=save_dir+"02_context/")
skgti.io.save_graph(p_graph,name="photo",directory=save_dir+"02_context/")
skgti.io.save_graphregions(t_graph,directory=save_dir+"02_context/")
plt.savefig(save_dir+"02_context/"+"histograms.svg",format="svg",bbox_inches='tight');plt.gcf().clear()
#skgti.io.plot_graph(t_graph);plt.show()
plt.imshow(label)
plt.savefig(save_dir+"02_context/"+"label.png",format="png",bbox_inches='tight');plt.gcf().clear()
'''

t_graph,p_graph=skgti.core.from_labelled_image(image,label)


id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=10,verbose=True)
skgti.io.save_matcher_details(matcher,image,label,None,save_dir,False)


'''
skgti.io.save_graph(t_graph,name="topological",directory=save_dir+"03_filtered_built_topology/")
skgti.io.save_graph(p_graph,name="photo",directory=save_dir+"03_filtered_built_topology/")
skgti.io.save_graphregions(t_graph,directory=save_dir+"03_filtered_built_topology/")
skgti.io.plot_graph(t_graph);plt.show()

'''