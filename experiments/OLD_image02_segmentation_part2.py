import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti


def manage_boundaries(image,roi):
    eroded_roi=sp.ndimage.morphology.binary_erosion(roi,iterations=2).astype(np.uint8)
    inner_boundary=roi/np.max(roi)-eroded_roi
    inner_boundary_values=np.ma.MaskedArray(image,mask=np.logical_not(inner_boundary)).compressed()
    bins=np.arange(np.min(inner_boundary_values),np.max(inner_boundary_values)+2)
    h,b=np.histogram(inner_boundary_values,bins)
    dominant_value=b[np.argmax(h)]
    #print("dominant_value:",dominant_value)
    modified_image=np.ma.MaskedArray(image,mask=inner_boundary).filled(dominant_value)
    return modified_image

def from_labels_to_residues(labelled_image):
    tmp_label=labelled_image+1 #to avoid confusion with 0s from masked area (roi)
    min_label=np.min(tmp_label)
    max_label=np.max(tmp_label)
    #residues=[np.where(tmp_label==i,1,0) for i in range(min_label,max_label+1)]
    bins=np.arange(np.min(tmp_label),np.max(tmp_label)+2)
    h,b=np.histogram(tmp_label,bins)

    residues=[]
    #for i in range(min_label,max_label+1):
    for i in range(0,len(h)):
        if h[i] != 0 : residues+=[np.where(tmp_label==b[i],1,0)]

    return residues


root_name="image02"
dir="Database/image02/"
save_dir="Database/image02/Evaluation/"


tp_model=skgti.core.TPModel()
tp_model.set_topology("tumor,vessel<liver")
tp_model.add_photometry("tumor<liver<vessel")

skgti.io.plot_graph(tp_model.p_graphs[0]);plt.title("A priori photometry")
plt.savefig(save_dir+'00_a_priori_photo.png')
#plt.show();quit()
plt.gcf().clear()
skgti.io.plot_graph(tp_model.t_graph);plt.title("A priori topology")
plt.savefig(save_dir+'00_a_priori_topo.png')
#plt.show();quit()
plt.gcf().clear()


slice_index=45 #AVEC CROP 45
'''
# IMAGE

image=np.load(os.path.join(dir,root_name+"_filtered.npy"))
tp_model.set_image(image)
#plt.imshow(np.rot90(image[:,:,slice_index]),cmap="gray");plt.show();quit()
region=np.load(os.path.join(dir,root_name+"_region2.npy"))
tp_model.set_region("liver",region)
#plt.imshow(np.rot90(region[:,:,slice_index]),cmap="gray");plt.show();quit()

tp_model.set_targets(['tumor','vessel'])
# PARAMETERS FROM KNOWLEDGE : ROI, NB CLUSTERS, INTERVALS
l_image=tp_model.roi_image()

#plt.imshow(np.rot90(l_image[:,:,slice_index].filled(0)),cmap="gray");plt.show();quit()
'''
##############################################
# LOAD DATA
##############################################
image=np.load(os.path.join(dir,root_name+"_filtered.npy"))
labelled_image=np.load(os.path.join(save_dir,root_name+"_meanshift.npy"))

roi=np.load(os.path.join(dir,root_name+"_region2.npy"))

downsampling=2
labelled_image=labelled_image[::downsampling,::downsampling,:]
roi=roi[::downsampling,::downsampling,:]
image=image[::downsampling,::downsampling,:]


##############################################
# MANAGE BOUNDARIES
##############################################
labelled_image=manage_boundaries(labelled_image,roi)

##############################################
# CROPPED VS ROI (DO ENLARGE BEFORE ?)
##############################################
labelled_image=skgti.utils.extract_subarray(labelled_image,roi)
image=skgti.utils.extract_subarray(image,roi)
roi=skgti.utils.extract_subarray(roi,roi)

roi_labelled_image=np.ma.array(labelled_image, mask=np.logical_not(roi))

##
# PLOT
##
roi_image=np.ma.array(image, mask=np.logical_not(roi))
plt.imshow(np.rot90((roi_image)[:,:,slice_index].filled(0)),cmap="gray");plt.axis('off');
plt.savefig(save_dir+'00_roied_image.png');#plt.show();quit()
plt.gcf().clear()
##
plt.imshow(np.rot90((roi_labelled_image+1)[:,:,slice_index].filled(0)),cmap="gray");plt.axis('off');
plt.savefig(save_dir+'00_labels.png');#plt.show();quit()
plt.gcf().clear()

# PLOT
##

skgti.io.plot_graph(skgti.core.graph_factory("0;1;2;3;4"));plt.title("Regions")
plt.savefig(save_dir+'00_observed_regions.png')
#plt.show();quit()
plt.gcf().clear()


##############################################
# CROPPED VS ROI (DO ENLARGE BEFORE ?)
##############################################

#plt.imshow(np.rot90(labelled_image[:,:,slice_index]),cmap="gray");plt.show();quit()


##############################################
# TOPOLOGICAL GRAPH
##############################################
residues=from_labels_to_residues(roi_labelled_image)

built_t_graph,new_residues=skgti.core.topological_graph_from_residues(residues)
print("Nodes: ",built_t_graph.nodes())

for i in range(0,len(new_residues)):
    slice=255*new_residues[i][:,:,slice_index].astype(np.uint8)
    sp.misc.imsave(os.path.join(save_dir,root_name+"_residue_"+str(i)+".png"),np.rot90(slice))

plt.subplot(121);skgti.io.plot_graph(built_t_graph);plt.title("Observed topology")
plt.subplot(122);skgti.io.plot_graph(tp_model.t_graph);plt.title("A priori topology")
plt.savefig(save_dir+'01_topological_graph.png')
#plt.show()
plt.gcf().clear()


##############################################
# PHOTOMETRIC GRAPH
##############################################
n=skgti.core.number_of_brother_links(tp_model.p_graphs[0])
built_p_graph=skgti.core.photometric_graph_from_residues(image,new_residues,n)

plt.subplot(121);skgti.io.plot_graph(built_p_graph);plt.title("Observed photometry")
plt.subplot(122);skgti.io.plot_graph(tp_model.p_graphs[0]);plt.title("A priori photometry")
plt.savefig(save_dir+'01_photo_graph.png')
#plt.show()
plt.gcf().clear()

##############################################
# MATCHINGS
##############################################
t_isomorphisms=skgti.core.find_subgraph_isomorphims(skgti.core.transitive_closure(built_t_graph),skgti.core.transitive_closure(tp_model.t_graph))
print("t iso:",t_isomorphisms)
print("t iso:",len(t_isomorphisms))
skgti.io.plot_graph_matchings(built_t_graph,tp_model.t_graph,t_isomorphisms);
plt.savefig(save_dir+'03_t_all_isos.png')
#plt.show();
plt.gcf().clear()

p_isomorphisms=skgti.core.find_subgraph_isomorphims(skgti.core.transitive_closure(built_p_graph),skgti.core.transitive_closure(tp_model.p_graphs[0]))
print("p iso:",p_isomorphisms)
print("p iso:",len(p_isomorphisms))
skgti.io.plot_graph_matchings(built_t_graph,tp_model.p_graphs[0],p_isomorphisms);
plt.savefig(save_dir+'03_p_all_isos.png')
#plt.show();
plt.gcf().clear()

matchings=skgti.core.find_common_isomorphisms([p_isomorphisms,t_isomorphisms])
print("common: ", matchings)

#quit()
skgti.io.plot_graph_matchings(built_t_graph,tp_model.t_graph,matchings);
plt.savefig(save_dir+'04_t_matchings.png')
#plt.show();
plt.gcf().clear()

skgti.io.plot_graph_matchings(built_p_graph,tp_model.p_graphs[0],matchings);
plt.savefig(save_dir+'04_p_matchings.png')
#plt.show();
plt.gcf().clear()

##############################################
# SURJECTION
##############################################
surj=skgti.core.find_sub_surjection(matchings)
skgti.io.plot_graph_surjection(built_t_graph,tp_model.t_graph,surj);
plt.savefig(save_dir+'05_t_surjection.png')
#plt.show();
plt.gcf().clear()

surj=skgti.core.find_sub_surjection(matchings)
skgti.io.plot_graph_surjection(built_p_graph,tp_model.p_graphs[0],surj);
plt.savefig(save_dir+'05_p_surjection.png')
#plt.show();
plt.gcf().clear()

