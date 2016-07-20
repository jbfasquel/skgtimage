import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

truth_dir="Database/image04/truth2/"
save_dir="Database/image04/meanshift_topolimitation/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

image_rgb=sp.misc.imread(os.path.join(truth_dir,"image.png"))
image_gray=skgti.utils.rgb2gray(image_rgb)
sp.misc.imsave("image_gray.png",image_gray)
image_chsv=skgti.utils.rgb2chsv(image_rgb)
#label=skgti.utils.mean_shift(image_chsv,0.12,None,True,True) #0.1 OK
label=skgti.utils.mean_shift(image_chsv,0.16,None,True,True) #0.1 OK


t_graph,p_graph=skgti.io.from_dir2(truth_dir,True)
#skgti.io.plot_graph_histogram(t_graph,p_graph);plt.show()
skgti.io.save_graph(t_graph,name="topological",directory=save_dir+"02_context/")
skgti.io.save_graph(p_graph,name="photo",directory=save_dir+"02_context/")
skgti.io.save_graphregions(t_graph,directory=save_dir+"02_context/")
plt.savefig(save_dir+"02_context/"+"histograms.svg",format="svg",bbox_inches='tight');plt.gcf().clear()
#skgti.io.plot_graph(t_graph);plt.show()


'''
skgti.io.save_intensities(p_graph,save_dir)
skgti.io.plot_graph(t_graph);plt.show()
skgti.io.plot_graph(p_graph);plt.show()
'''



t_graph,p_graph=skgti.core.from_labelled_image(image_gray,label)
skgti.io.save_graph(t_graph,name="topological",directory=save_dir+"03_filtered_built_topology/")
skgti.io.save_graph(p_graph,name="photo",directory=save_dir+"03_filtered_built_topology/")
skgti.io.save_graphregions(t_graph,directory=save_dir+"03_filtered_built_topology/")
skgti.io.plot_graph(t_graph);plt.show()


'''
#sp.misc.imsave(truth_dir+"image_gray.png",image_ndg)
tmp=np.ones(image_rgb.shape[0:2])
for r in ['glassredcar', 'glassyellowcar', 'redcar', 'rims', 'wheels', 'yellowcar']:
    tmp-=sp.misc.imread(truth_dir+"region_"+r+".png")

sp.misc.imsave(truth_dir+"region_background.png",tmp)
'''
