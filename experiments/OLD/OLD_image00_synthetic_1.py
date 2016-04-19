import os
import numpy as np
import matplotlib.pyplot as plt
import skgtimage as skgti
import scipy as sp;from scipy import misc,ndimage

tmp_dir='synthetic_01/'

if not os.path.exists(tmp_dir): os.mkdir(tmp_dir)

#################
# IMAGE
#################
#Region A
image=50+np.zeros((100,100),dtype=np.float)
region_A=np.ones(image.shape)
#Region B
region_B=skgti.utils.draw_rectangle(image,(50,30),(80,30),100);#sp.misc.imsave(tmp_dir+'B.png',region_B)
#Region C
skgti.utils.draw_square(image,(30,30),20,150)
#Region D
skgti.utils.draw_square(image,(70,30),20,200)
#Region E
skgti.utils.draw_rectangle(image,(50,70),(60,30),200)
#Noise
image=np.round(skgti.utils.add_gaussian_noise(image,0,5.0)).astype(np.uint8)
print(np.min(image),np.max(image))
#Save image
sp.misc.imsave(tmp_dir+'image.png',image)
#################
# PLOT
#################
'''
plt.subplot(121)
plt.imshow(image,'gray');plt.axis('off');plt.title("Initial image")
plt.subplot(122)
histo,bins=skgti.utils.int_histogram(image)
#y,x=skgti.utils.float_histogram(image)
plt.plot(bins,histo);plt.title("Histogram")
plt.savefig(tmp_dir+'01_image_histo.png')
plt.show();plt.gcf().clear()
'''

#################
# KNOWLEDGE
#################
tp_model=skgti.core.TPModel()
tp_model.set_topology("C,D<B;B,E<A")
tp_model.add_photometry("A<B<C<D=E")
#tp_model.add_photometry("1<2<3<4=5")

#skgti.io.plot_graph_matching(tp_model.t_graph,tp_model.p_graphs[0],{'A':'1','B':'2','C':'3','D':'4','E':'5'});plt.show();quit()

'''
skgti.io.plot_model(tp_model)
plt.savefig(tmp_dir+'02_a_priori_knowledge.png')
plt.show();plt.gcf().clear()
'''

#################
# CONTEXT: IMAGE AND ALREADY SEGMENTED REGIONS
#################
tp_model.set_image(image)

#################
# AVAILABLE INFORMATIONS: ROI (TOPOLOGY), DISTINCT CLASSES (TOPOLOGY+PHOTOMETRY), INTERVAL PER CLASS
#################
#target=skgti.core.root_tree_node(tp_model.t_graph);print(target)
t_clusters=skgti.core.classes_for_target(tp_model.t_graph)
distinct_clusters=skgti.core.distinct_classes(list(t_clusters),tp_model.p_graphs)
constraints=skgti.core.interval_for_classes(image,tp_model.roi(),distinct_clusters,tp_model.t_graph,tp_model.p_graphs)


''''
plt.subplot(131)
plt.imshow(image,'gray');plt.axis('off');plt.title("Initial image")
plt.subplot(132)
plt.imshow(tp_model.roi(),'gray',vmin=0,vmax=1);plt.axis('off');plt.title("ROI")
plt.subplot(133)
histo,bins=skgti.utils.int_histogram(image)
plt.plot(bins,histo);plt.title("Histogram: "+str(len(distinct_clusters))+" classes\n"+str(constraints))
plt.savefig(tmp_dir+'03_parameters.png')
plt.show();plt.gcf().clear()
'''


#tp_model.set_region('A',region_A)

#################
# SEGMENTATION
#################
labelled_image=skgti.utils.kmeans(image,len(distinct_clusters))
'''
plt.imshow(labelled_image,'gray');plt.axis('off');plt.title("Segmentation");
plt.savefig(tmp_dir+'04_segmentation.png')
plt.show();plt.gcf().clear()
'''
#################
# RECOGNITION: MATCHING
#################

#Topological isomorphisms
built_t_graph,new_residues=skgti.core.topological_graph_from_labels(labelled_image)
t_isomorphisms=skgti.core.find_subgraph_isomorphims(built_t_graph,tp_model.t_graph)
'''
plt.savefig(tmp_dir+'05_iso_topo.png')
plt.show();plt.gcf().clear()
'''
#Photometric isomorphisms
n=skgti.core.number_of_brother_links(tp_model.p_graphs[0])
built_p_graph=skgti.core.photometric_graph_from_residues(image,new_residues,n)
p_isomorphisms=skgti.core.find_subgraph_isomorphims(skgti.core.transitive_closure(built_p_graph),skgti.core.transitive_closure(tp_model.p_graphs[0]))

#Plot
'''
plt.subplot(221);skgti.io.plot_graph_matching(tp_model.t_graph,built_t_graph,t_isomorphisms[0]);plt.title("Topological matching 0")
plt.subplot(222);skgti.io.plot_graph_matching(tp_model.t_graph,built_t_graph,t_isomorphisms[1]);plt.title("Topological matching 1")
plt.subplot(223);skgti.io.plot_graph_matching(tp_model.p_graphs[0],built_p_graph,p_isomorphisms[0]);plt.title("Photometric matching 0")
plt.subplot(224);skgti.io.plot_graph_matching(tp_model.p_graphs[0],built_p_graph,p_isomorphisms[1]);plt.title("Photometric matching 1")
plt.savefig(tmp_dir+'05_isomorphisms.png')
plt.show();plt.gcf().clear()
'''
isomorphisms=skgti.core.find_common_isomorphisms([t_isomorphisms,p_isomorphisms])
print(isomorphisms)

'''
plt.subplot(121);skgti.io.plot_graph_matching(tp_model.t_graph,built_t_graph,isomorphisms[0]);plt.title("Matching (topo)")
plt.subplot(122);skgti.io.plot_graph_matching(tp_model.p_graphs[0],built_p_graph,isomorphisms[0]);plt.title("Matching (photo)")
plt.savefig(tmp_dir+'06_matching.png')
plt.show();plt.gcf().clear()
'''

#################
# RECOGNITION: UPDATING GRAPHS
#################
matching=isomorphisms[0]
regions=skgti.core.regions_from_residues(built_t_graph,new_residues)
skgti.core.update_graphs_from_identified_regions([tp_model.t_graph]+tp_model.p_graphs,regions,matching)
skgti.io.plot_model(tp_model)
plt.savefig(tmp_dir+'07_result.png')
plt.show();plt.gcf().clear()


