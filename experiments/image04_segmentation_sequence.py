import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

root_name="image04"
dir="Database/image04_seg"


#########
# KNOWLEDGE
#########
tp_model=skgti.core.TPModel()
tp_model.set_topology("0;1;2;3;4;5;7,8,9,10,11<6")
#skgti.io.plot_graph(tp_model.t_graph);plt.show();quit()
#tp_model.set_photometry(["1=7<3=8<4=9<0=6;2;5;10;11","2=10;2<0;0=1=3=4=6=7=8=9;5;11","5=11;11<0;0=1=2=3=4=6=7=8=9=10"])
tp_model.set_photometry(["1=7<3=8<4=9<0=6;2=5=10=11","2=10=5=11;2<0;0=1=3=4=6=7=8=9","5=11;11<0;0=1=2=3=4=6=7=8=9=10"])
#skgti.io.plot_graph(tp_model.p_graphs[0]);plt.show();quit()
#skgti.io.plot_graph(tp_model.p_graphs[1]);plt.show();quit()
#skgti.io.plot_graph(tp_model.p_graphs[2]);plt.show();quit()
image=sp.misc.imread(os.path.join(dir,root_name+"_filtered.jpeg"))

image_hsv=skgti.utils.rgb2hsv(image)
tp_model.set_image(image_hsv)
#plt.imshow(image);plt.show();quit()

#########
# CONTEXT
#########
for r in ['0','1','2','4','5','6']: tp_model.set_region(r,sp.misc.imread(os.path.join(dir,root_name+"_region"+r+".png")))
#skgti.io.plot_model(tp_model);plt.show()

#########
# SEGMENTATION
#########
tp_model.set_targets(['7'])
l_image=tp_model.roi_image() #;print(l_image[:,:,0]);print(l_image[:,:,1]);print(l_image[:,:,2])
nb=tp_model.number_of_clusters() #;print(nb)
intervals=tp_model.intervals_for_clusters()
# KMEANS SEGMENTATION (try with intervals=None for unconstrained-based seeding -> conforme with paper)
labelled_image=skgti.utils.kmeans(l_image,nb,n_seedings=1,intervals=intervals,fct=skgti.utils.hsv2chsv,mc=True) #;print(labelled_image);quit()

##############################################
# PLOT OBTAINED LABELLED IMAGE
##############################################
#plt.imshow(labelled_image);plt.show()

##############################################
# PLOT OBTAINED TOPOLOGICAL GRAPH ONLY
##############################################
g,new_residues=skgti.core.topological_graph_from_labels(labelled_image)
#skgti.io.plot_graph(g);plt.show();quit()

##############################################
# PLOT TOPOLOGICAL GRAPH + OBTAINED REGIONS
##############################################
regions=skgti.core.regions_from_residues(g,new_residues)
for i in range(0,len(regions)):
    g.set_region(i,regions[i])
tmp_model=skgti.core.TPModel()
tmp_model.t_graph=g
tmp_model.set_image(image)
skgti.io.plot_model(tmp_model);plt.show();quit()
#skgti.core.topological_graph_from_residues()

# IDENTIFICATION
tp_model.identify_from_labels(labelled_image)

# PLOT
skgti.io.plot_model(tp_model);plt.show()

'''
#########
# STEP 1
#########
tp_model.set_region("pen",sp.misc.imread(os.path.join(dir,root_name+"_region0.png")))
tp_model.set_region("clear_file",sp.misc.imread(os.path.join(dir,root_name+"_region1.png")))
tp_model.set_region("file",sp.misc.imread(os.path.join(dir,root_name+"_region2.png")))
#for r in ['0','1','2']: tp_model.set_region(r,sp.misc.imread(os.path.join(dir,root_name+"_region"+r+".png")))

# PLOT
#skgti.io.plot_model(tp_model);plt.show()

#########
# STEP 2
#########
tp_model.set_targets(['text','paper'])
# PARAMETERS FROM KNOWLEDGE : ROI, NB CLUSTERS, INTERVALS
l_image=tp_model.roi_image()
nb=tp_model.number_of_clusters()
intervals=tp_model.intervals_for_clusters()

# KMEANS SEGMENTATION
labelled_image=skgti.utils.kmeans(l_image,nb,n_seedings=30,intervals=intervals)
#plt.imshow(labelled_image,"gray");plt.show();quit()

# IDENTIFICATION
tp_model.identify_from_labels(labelled_image)

# PLOT
skgti.io.plot_model(tp_model);plt.show()
'''