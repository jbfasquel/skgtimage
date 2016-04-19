import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

root_name="image03"
dir="Database/image03"
save_dir="Database/image03/meanshift_down3_test1/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)


#########
# KNOWLEDGE
#########
t_graph=skgti.core.graph_factory("E<D;G<F;D,F,H,I<C<B<A")
p_graph=skgti.core.graph_factory("B=F<D=H<I=E<C=A=G")
print("Number of automorphisms:",skgti.core.nb_automorphisms([t_graph,p_graph]))
skgti.io.plot_graph(t_graph);plt.title("A priori T knowledge");plt.savefig(save_dir+'01_A_priori_topo.png');plt.gcf().clear()
skgti.io.plot_graph(p_graph);plt.title("A priori P knowledge");plt.savefig(save_dir+'01_A_priori_photo.png');plt.gcf().clear()


#########
# IMAGE: COLOR AND GRAY
#########
downsampling=2

image_rgb=sp.misc.imread(os.path.join(dir,"image03.jpeg")).astype(np.float)
for i in range(0,3):
    image_rgb[:,:,i]=sp.ndimage.filters.median_filter(image_rgb[:,:,i], 3)

image_rgb=image_rgb[740:1360,150:1700,:] #Manual crop
roi=sp.misc.imread(os.path.join(dir,"image03_region1.png"))
roi=roi[740:1360,150:1700] #Manual crop

image_rgb=image_rgb[::downsampling,::downsampling,:]
roi=roi[::downsampling,::downsampling]

image_hsv=skgti.utils.rgb2hsv(image_rgb)
image_chsv=skgti.utils.hsv2chsv(image_hsv)
image=skgti.utils.rgb2gray(image_rgb)
sp.misc.imsave(os.path.join(save_dir,"01_image03_gray.png"),image)
sp.misc.imsave(os.path.join(save_dir,"01_roi.png"),roi)
#plt.imshow(image,"gray");plt.show()

#########
# MEANSHIFT ON COLOR IMAGE -> DOES NOT WORK AND GRAY -> NOT ENOUGH DISCRIMINATION
#########
nb_classes=4
bandwidth=0.1 #on obtient -> 5 classes (au lieu de 4 attendues) ; avec bandwidth de 0.15, on obtient 4 classes mais résultat moins bien visuellement
roi_mask=np.dstack(tuple([roi for i in range(0,3)]))
l_image_chsv=np.ma.array(image_chsv, mask=np.logical_not(roi_mask))
labelled_image=skgti.utils.mean_shift(l_image_chsv,bandwidth=0.1,spatial_dim=2,n_features=3,verbose=True) #0.1 OK

#l_image=np.ma.array(image, mask=np.logical_not(roi))
#labelled_image=skgti.utils.mean_shift(l_image,bandwidth=8,spatial_dim=2,n_features=1,verbose=True)
sp.misc.imsave(os.path.join(save_dir,"01_labelled_meanshift.png"),(labelled_image+1).filled(0).astype(np.uint8))
plt.imshow((labelled_image+1).filled(0),"gray");
plt.savefig(save_dir+'01_labelled_meanshift_visu.png');plt.gcf().clear()


#########
# SAVE INITIAL LABEL
#########

#########
# RECOGNITION-GRAPH
#########
#downsampling=5
image=sp.misc.imread(os.path.join(save_dir,"01_image03_gray.png"))
segmentation=sp.misc.imread(os.path.join(save_dir,"01_labelled_meanshift.png"))
roi=sp.misc.imread(os.path.join(save_dir,"01_roi.png"))

l_image=np.ma.array(image, mask=np.logical_not(roi))
l_segmentation=np.ma.array(segmentation, mask=np.logical_not(roi))
###########################################
# BUILDING GRAPHS
###########################################
residues=skgti.core.residues_from_labels(l_segmentation)
built_t_graph,new_residues=skgti.core.topological_graph_from_residues(residues)
built_p_graph=skgti.core.photometric_graph_from_residues(l_image,new_residues)

#PLOT
skgti.io.plot_graphs_regions_new([built_t_graph,built_p_graph])
plt.savefig(save_dir+'02_built_graphs.png');plt.gcf().clear()

filled_new_residues=skgti.core.fill_regions(new_residues)
skgti.io.plot_graphs_regions([built_t_graph,built_p_graph],filled_new_residues)
plt.savefig(save_dir+'02_built_graphs_regions.png');plt.gcf().clear()
#SAVE DETAILED
nodes=built_t_graph.nodes()
for i in range(0,len(nodes)):
    built_t_graph.set_region(nodes[i],skgti.core.fill_region(new_residues[i]))
skgti.io.save_graph("topo",built_t_graph,nodes=None,tree=True,directory=save_dir+"built_t_graph",save_regions=True)

###########################################
# MATCHING
###########################################
matching,(common_isomorphisms,(t_isomorphisms,p_isomorphisms))=skgti.core.recognize_version1([built_t_graph,built_p_graph],[t_graph,p_graph],True)

skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
#skgti.io.plot_graphs_matching([built_t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'04_matching_initial.png');plt.gcf().clear()

###########################################
# MATCHING REFINEMENT (FILTERING)
###########################################
skgti.core.t_filtering_v1(matching,new_residues,built_t_graph,built_p_graph)

skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'04_matching_filtered1.png');plt.gcf().clear()

skgti.core.p_filtering(matching,image,new_residues,built_t_graph,built_p_graph)

skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'04_matching_filtered2.png');plt.gcf().clear()
###########################################
# UPDATING MATCHINGS
###########################################
skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'05_final_matching.png');plt.gcf().clear()

skgti.core.update_graphs([t_graph,p_graph],new_residues,matching)

#Save details
skgti.io.save_graph("topo",t_graph,nodes=None,tree=True,directory=save_dir+"final_t_graph",save_regions=True)


skgti.io.plot_graphs_regions_new([t_graph,p_graph],nb_rows=1)
plt.savefig(save_dir+'06_result_graph_regions.png');plt.gcf().clear()


result=skgti.utils.combine(image.shape,t_graph,['A','B'],[1,2])
plt.imshow(result,cmap="gray",vmin=np.min(result),vmax=np.max(result),interpolation="nearest");plt.axis('off');
plt.savefig(save_dir+'06_result_single_image.png');plt.gcf().clear()

#print(nodes_with_ambiguity)
