import os,pickle
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

root_name="image03"
dir="Database/image03"
save_dir="Database/image03/meanshift_down3_region_top_test1_input_filtered_newfiltering2/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)


#########
# KNOWLEDGE
#########
#A:contour_blanc ; B:bandeau noir ; C: texte_blanc_du_noir ; D:fond_jaune ; E:texte_du_fond_jaune ; F:fleurs
#G:trous_noirs_du_texte_de_C ; H: fond_jaune_du_text_sur_jaune_E
t_graph=skgti.core.graph_factory("C,D<B<A;E,F<D;G<C;H<E")
p_graph=skgti.core.graph_factory("G=B<E=F<H=D<A=C")
print("Number of automorphisms:",skgti.core.nb_automorphisms([t_graph,p_graph]))
skgti.io.plot_graph(t_graph);plt.title("A priori T knowledge");plt.savefig(save_dir+'01_A_priori_topo.png');plt.gcf().clear()
skgti.io.plot_graph(p_graph);plt.title("A priori P knowledge");plt.savefig(save_dir+'01_A_priori_photo.png');plt.gcf().clear()


#########
# IMAGE: COLOR AND GRAY
#########
downsampling=3

image_rgb=sp.misc.imread(os.path.join(dir,"image03.jpeg"))
#image_rgb=sp.misc.imread(os.path.join(dir,"image03.jpeg")).astype(np.float)
for i in range(0,3): image_rgb[:,:,i]=sp.ndimage.filters.median_filter(image_rgb[:,:,i], 5)

image=skgti.utils.rgb2gray(image_rgb)
#image=sp.misc.imread(os.path.join(dir,"image03_ndg.jpeg"))
roi=sp.misc.imread(os.path.join(dir,"image03_region0.png"))

#CROP
image_rgb=skgti.utils.extract_subarray_rgb(image_rgb,roi)
image=skgti.utils.extract_subarray(image,roi)
roi=skgti.utils.extract_subarray(roi,roi)

#DOWNSAMPLING
image_rgb=image_rgb[::downsampling,::downsampling,:]
image=image[::downsampling,::downsampling]
roi=roi[::downsampling,::downsampling]

#DOWNSAMPLING
sp.misc.imsave(os.path.join(save_dir,"01_image03_gray.png"),image)
sp.misc.imsave(os.path.join(save_dir,"01_roi.png"),roi)

#########
# MEANSHIFT ON COLOR IMAGE -> DOES NOT WORK AND GRAY -> NOT ENOUGH DISCRIMINATION
#########
nb_classes=4
#bandwidth=0.1 #on obtient -> 5 classes (au lieu de 4 attendues) ; avec bandwidth de 0.15, on obtient 4 classes mais résultat moins bien visuellement
bandwidth=0.09 #pour filtrage -> 5 classes, sinon erreur pour 4 classes
image_hsv=skgti.utils.rgb2hsv(image_rgb)
image_chsv=skgti.utils.hsv2chsv(image_hsv)
roi_mask=np.dstack(tuple([roi for i in range(0,3)]))
l_image_chsv=np.ma.array(image_chsv, mask=np.logical_not(roi_mask))
labelled_image=skgti.utils.mean_shift(l_image_chsv,bandwidth=bandwidth,spatial_dim=2,n_features=3,verbose=True) #0.1 OK

sp.misc.imsave(os.path.join(save_dir,"01_labelled_meanshift.png"),(labelled_image+1).filled(0).astype(np.uint8))
plt.imshow((labelled_image+1).filled(0),"gray");
plt.savefig(save_dir+'01_labelled_meanshift_visu.png');plt.gcf().clear()
#quit()

#########
# RECOGNITION-GRAPH
#########
#downsampling=5
image=sp.misc.imread(os.path.join(save_dir,"01_image03_gray.png"))
segmentation=sp.misc.imread(os.path.join(save_dir,"01_labelled_meanshift.png"))
roi=sp.misc.imread(os.path.join(save_dir,"01_roi.png"))
segmentation=skgti.core.manage_boundaries(segmentation,roi)


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


for i in range(0,len(new_residues)):
    r=new_residues[i]
    stat=skgti.core.region_stat(image,r,fct=np.mean,mc=False)
    print("Index ", i , " mean: ",stat)


###########################################
# MATCHING
###########################################
'''
matching,(common_isomorphisms,(t_isomorphisms,p_isomorphisms))=skgti.core.recognize_version1([built_t_graph,built_p_graph],[t_graph,p_graph],True)
print("matching before save:" , matching)
#Save
tmp=[matching,(common_isomorphisms,(t_isomorphisms,p_isomorphisms))]
matching_file=open('data.pkl', 'wb');pickle.dump(tmp,matching_file);matching_file.close()
'''
#Load
matching_file=open('data.pkl', 'rb');
[matching,(common_isomorphisms,(t_isomorphisms,p_isomorphisms))]=pickle.load(matching_file)
print("matching after save:" , matching)

for i in range(0,len(common_isomorphisms)):
    iso=common_isomorphisms[i]
    skgti.io.plot_graphs_matching([t_graph],[built_t_graph],iso,titles=["Topology"])
    plt.savefig(save_dir+'03_common_iso_t_'+str(i)+'.png');plt.gcf().clear()

eies=[]
for i in range(0,len(common_isomorphisms)):
    iso=common_isomorphisms[i]
    eies+=[skgti.core.energie_dist(built_p_graph,p_graph,iso)]
print("Energies regarding distances: ", eies)

eies=[]
for i in range(0,len(common_isomorphisms)):
    iso=common_isomorphisms[i]
    eies+=[skgti.core.energie_sim(built_p_graph,p_graph,iso)]
print("Energies regarding brother sim: ", eies)

#new_matching,new_common_isomorphisms=skgti.core.filtered_common_subgraph_isomorphisms_v1(matching,common_isomorphisms)
new_matching=skgti.core.filtered_common_subgraph_isomorphisms_v2(matching,common_isomorphisms)
print("new matching:" , new_matching)

skgti.io.plot_graphs_matching([t_graph],[built_t_graph],matching,titles=["Topology"])
#skgti.io.plot_graphs_matching([built_t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'04_matching_initial_t.png');plt.gcf().clear()
skgti.io.plot_graphs_matching([p_graph],[built_p_graph],matching,titles=["Photometry"])
#skgti.io.plot_graphs_matching([built_t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'04_matching_initial_p.png');plt.gcf().clear()

###########################################
# MATCHING REFINEMENT (FILTERING)
###########################################

skgti.io.plot_graph_matchings(built_t_graph,t_graph,common_isomorphisms)
plt.savefig(save_dir+'05_t_common_iso_filtered.png');plt.gcf().clear()
skgti.io.plot_graph_matching(built_t_graph,t_graph,new_matching)
plt.savefig(save_dir+'05_t_matching_filtered.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_p_graph,p_graph,common_isomorphisms)
plt.savefig(save_dir+'05_p_common_iso_filtered.png');plt.gcf().clear()
skgti.io.plot_graph_matching(built_p_graph,p_graph,new_matching)
plt.savefig(save_dir+'05_p_matching_filtered.png');plt.gcf().clear()

skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],new_matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'05_filtered_matching.png');plt.gcf().clear()


skgti.core.t_filtering_v2(image,new_residues,new_matching,built_t_graph,built_p_graph)
skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],new_matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'06_final_matching.png');plt.gcf().clear()

skgti.core.update_graphs([t_graph,p_graph],new_residues,new_matching)

skgti.io.plot_graphs_regions_new([t_graph,p_graph],nb_rows=1)
plt.savefig(save_dir+'06_result_graph_regions.png');plt.gcf().clear()

###########################################
# UPDATING MATCHINGS
###########################################
'''
skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'05_final_matching.png');plt.gcf().clear()

skgti.core.update_graphs([t_graph,p_graph],new_residues,matching)
'''
#Save details
skgti.io.save_graph("topo",t_graph,nodes=None,tree=True,directory=save_dir+"final_t_graph",save_regions=True)


skgti.io.plot_graphs_regions_new([t_graph,p_graph],nb_rows=1)
plt.savefig(save_dir+'06_result_graph_regions.png');plt.gcf().clear()


result=skgti.utils.combine(image.shape,t_graph,['A','B'],[1,2])
plt.imshow(result,cmap="gray",vmin=np.min(result),vmax=np.max(result),interpolation="nearest");plt.axis('off');
plt.savefig(save_dir+'06_result_single_image.png');plt.gcf().clear()

#print(nodes_with_ambiguity)
