import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

root_name="image03"
dir="Database/image03_seg/"


def simplify_graph_vs_submatching(ref_graph,cur_graph,residues):
    t_isomorphisms=skgti.core.find_subgraph_isomorphims(cur_graph,ref_graph)
    #Nodes to be removed
    nodes_to_be_removed=set()
    for n in cur_graph.nodes():
        is_n_found=False
        for iso in t_isomorphisms:
            if n in iso.keys(): is_n_found=True
        if is_n_found == False:
            nodes_to_be_removed|=set([n])

    print("To be removed: ", nodes_to_be_removed)
    ############################
    # ADDING UNMATCHED REGION RESIDUES TO FATHER ONES
    ############################
    for n in nodes_to_be_removed:
        father=cur_graph.successors(n)[0]
        new_residues[father]=np.logical_or(residues[father],residues[n]).astype(np.uint8)

    ############################
    # KEEPING MATCHED REGION RESIDUES ONLY
    ############################
    final_residues=[]
    for i in range(0,len(residues)):
        if i not in nodes_to_be_removed: final_residues+=[residues[i]]

    ############################
    # UPDATING THE GRAPH
    ############################
    tmp_t_graph,final_residues=skgti.core.topological_graph_from_residues(final_residues)
    return tmp_t_graph,final_residues

def manage_boundaries(image,roi):
    inner_boundary=roi-sp.ndimage.morphology.binary_erosion(roi,iterations=1).astype(np.uint8)
    inner_boundary_values=np.ma.MaskedArray(image,mask=np.logical_not(inner_boundary)).compressed()
    bins=np.arange(np.min(inner_boundary_values),np.max(inner_boundary_values)+2)
    h,b=np.histogram(inner_boundary_values,bins)
    dominant_value=b[np.argmax(h)]
    #print("dominant_value:",dominant_value)
    modified_image=np.ma.MaskedArray(image,mask=inner_boundary).filled(dominant_value)
    return modified_image





#########
# LABELLED IMAGE
#########
labelled_image=sp.misc.imread(dir+'meanshift_region1.png')
labelled_image=labelled_image[740:1360,150:1700] #Manual crop
#plt.imshow(labelled_image);plt.show();quit()

downsampling=1 #30 ; 20 ok for testing
labelled_image=labelled_image[::downsampling,::downsampling]
#labelled_image=np.where(labelled_image==0,1,labelled_image) #Ack for
roi=np.where(labelled_image==0,0,1)
labelled_image=manage_boundaries(labelled_image,roi)
labelled_image=np.ma.MaskedArray(labelled_image,mask=np.logical_not(roi))

#########
# A PRIORI KNOWLEDGE
#########
tp_model=skgti.core.TPModel()
#NEW
#tp_model.set_topology("8<7<3<2<1;5,6<3")
tp_model.set_topology("3,5,7,8<2<1<0;4<3;6<5")
tp_model.set_photometry(["1=2=3;5<8=6<7","1=2=3<5=6=7=8","2<1=3=5=6=7=8"])

tp_model.set_image(labelled_image)

##############################################
# PLOT OBTAINED LABELLED IMAGE
##############################################
#plt.imshow(labelled_image);plt.show()

##############################################
# PLOT OBTAINED TOPOLOGICAL GRAPH ONLY
##############################################
built_t_graph,new_residues=skgti.core.topological_graph_from_labels(labelled_image)

plt.subplot(121);skgti.io.plot_graph(tp_model.t_graph);plt.title("Expected")
plt.subplot(122);skgti.io.plot_graph(built_t_graph);plt.title("Built before filtering");plt.show()

##############################################
# FILTERING TOPOLOGICAL GRAPH
##############################################
built_t_graph,new_residues=simplify_graph_vs_submatching(tp_model.t_graph,built_t_graph,new_residues)

t_isomorphisms=skgti.core.find_subgraph_isomorphims(built_t_graph,tp_model.t_graph)
for iso in t_isomorphisms:
    print("ISOMORPHISM:" , iso)


plt.subplot(121);skgti.io.plot_graph(tp_model.t_graph);plt.title("Expected")
plt.subplot(122);skgti.io.plot_graph(built_t_graph);plt.title("Built filtered");plt.show() #;quit()



##############################################
# PHOTOMETRIC GRAPH [0]
##############################################
n=skgti.core.number_of_brother_links(tp_model.p_graphs[0])
print(n)
built_p_graph_0=skgti.core.photometric_graph_from_residues(tp_model.get_image(),new_residues,n)
plt.subplot(121);skgti.io.plot_graph(tp_model.p_graphs[0]);plt.title("Expected Photo 0")
plt.subplot(122);skgti.io.plot_graph(built_p_graph_0);plt.title("Built Photo 0");plt.show();quit()

quit()
#skgti.io.plot_graph(g);plt.show()

##############################################
# PLOT TOPOLOGICAL GRAPH + OBTAINED REGIONS
##############################################
regions=skgti.core.regions_from_residues(g,new_residues)
for i in range(0,len(regions)):
    g.set_region(i,regions[i])
tmp_model=skgti.core.TPModel()
tmp_model.t_graph=g
tmp_model.set_image(labelled_image)
skgti.io.plot_model(tmp_model);plt.show()
skgti.io.save_graph("image03_region1_",tmp_model.t_graph,directory="Tmp",save_regions=True)

