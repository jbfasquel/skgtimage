from skgtimage.core.topology import merge_nodes_topology,topological_merging_candidates
from skgtimage.core.photometry import merge_nodes_photometry,grey_levels,region_stat
from skgtimage.core.propagation import cost2merge

from skgtimage.core.factory import photometric_graph_from_residues_refactorying
import numpy as np


def compute_dm(p_g):
    ord_n = p_g.get_ordered_nodes()
    dm = {}
    for j in range(1, len(ord_n)):
        #diff = abs(p_g.get_mean_residue_intensity(j) - p_g.get_mean_residue_intensity(j - 1))
        diff = abs(p_g.get_mean_residue_intensity(ord_n[j]) - p_g.get_mean_residue_intensity(ord_n[j-1]))
        if diff in dm:
            #dm[diff] += [(j - 1, j)]
            dm[diff] += [(ord_n[j - 1], ord_n[j])]
        else:
            #dm[diff] = [(j - 1, j)]
            dm[diff] = [(ord_n[j - 1], ord_n[j])]
    return dm

def merge_photometry_color(image,label,roi,times,verbose=False):
    """
    Merge regions of similar photometry, even if there are not adjacent
    :param chsv_image:
    :param label:
    :param roi:
    :param times:
    :param verbose:
    :return:
    """
    ################
    #Precomputing properties
    #chsv_image = rgb2chsv(image)
    levels=grey_levels(label,roi)
    nb_levels=len(levels)
    mean_for_each_level=[]
    size_for_each_level=[]
    for i in range(0,nb_levels):
        if roi is not None:
            roi_i = np.logical_and(np.where(label == levels[i], 1, 0),roi)
        else:
            roi_i = np.where(label == levels[i], 1, 0)
        #plt.imshow(roi_i,"gray");plt.show()
        ri = region_stat(image, roi_i, np.mean, mc=True)
        mean_for_each_level+=[ri]
        size_for_each_level+=[np.sum(roi_i)]

    ################
    #Computing adjacency matrix
    adj_matrix=np.ones((nb_levels,nb_levels)) #1 assumes to be higher than any distance
    for i in range(0,nb_levels):
        for j in range(0, nb_levels):
            if i < j:
                ri=mean_for_each_level[i]
                rj=mean_for_each_level[j]
                dist=np.sqrt((rj[0]-ri[0])**2+(rj[1]-ri[1])**2+(rj[2]-ri[2])**2)
                adj_matrix[i,j]=dist
                adj_matrix[j,i]=dist

    new_label=np.copy(label)
    for it in range(0,times):
        if verbose:
            print("Merge_photometry_color, remaining iterations:", times-it)
        #plt.imshow(new_label);plt.show()
        ################
        #Search minimal distance
        mini=np.min(adj_matrix)
        min_is,min_js=np.where(adj_matrix==mini)
        min_i=min_is[0]
        min_j=min_js[0]
        ################
        #Merging j+i -> i ; on vire j
        #Modification of label, adjency matrix, etc
        ################
        merging = (levels[min_i], levels[min_j])
        #Label j takes the label i
        roi_to_change=np.where(new_label==merging[1],1,0)
        new_label=np.ma.array(new_label, mask=roi_to_change).filled(merging[0])
        #Update
        #levels.pop(min_j)
        levels=np.delete(levels, min_j, 0)
        #nb_levels=len(levels)
        tmp_mean_i=mean_for_each_level[min_i]
        tmp_mean_j=mean_for_each_level[min_j]
        mean_for_each_level[min_i]=(size_for_each_level[min_i]*mean_for_each_level[min_i]+size_for_each_level[min_j]*mean_for_each_level[min_j])/(size_for_each_level[min_i]+size_for_each_level[min_j])
        mean_for_each_level.pop(min_j)
        size_for_each_level[min_i]=size_for_each_level[min_i]+size_for_each_level[min_j]
        size_for_each_level.pop(min_j)
        adj_matrix=np.delete(adj_matrix, min_j, 0)
        adj_matrix=np.delete(adj_matrix, min_j, 1)
        if min_i < min_j:
            new_index_min_i=min_i
        else:
            new_index_min_i=min_i-1
        r_min_i=mean_for_each_level[new_index_min_i]
        for n in range(0,len(levels)):
            if n != new_index_min_i:
                rn = mean_for_each_level[n]
                dist = np.sqrt((rn[0] - r_min_i[0]) ** 2 + (rn[1] - r_min_i[1]) ** 2 + (rn[2] - r_min_i[2]) ** 2)
                adj_matrix[new_index_min_i, n] = dist
                adj_matrix[n, new_index_min_i] = dist

    return new_label


def merge_photometry_gray(image, label, nb_times=10):
    residues = [np.where(label == i, 255, 0) for i in range(0, np.max(label) + 1)]
    p_g = photometric_graph_from_residues_refactorying(image, residues)
    #dm=compute_dm(p_g)
    for i in range(0, nb_times):
        dm = compute_dm(p_g)
        min_diff = sorted(dm)[0]
        merge = dm[min_diff][0]
        #print("merge:",merge)
        if len(dm[min_diff]) == 1: dm.pop(min_diff)
        if (merge[0] in p_g.nodes()) and (merge[1] in p_g.nodes()):
            merge_nodes_photometry(p_g, merge[0], merge[1])
        #dm = compute_dm(p_g)

    new_labelled = np.zeros(image.shape, dtype=np.int16)
    nodes = p_g.nodes()
    print(nodes)
    for i in range(0, len(nodes)):
        roi = p_g.get_region(nodes[i])
        new_labelled = np.ma.masked_array(new_labelled, mask=roi).filled(i)

    return new_labelled



def merge_filtering(t_graph,p_graph,nb_times=1):
    for i in range(0,nb_times):
        remaining_nodes = t_graph.nodes()
        print("Merging - Remaining nodes", len(remaining_nodes))
        ordered_merging_candidates, d2m = cost2merge(t_graph, p_graph, remaining_nodes, remaining_nodes)
        merge = ordered_merging_candidates[0]
        # Apply merging
        merge_nodes_photometry(p_graph, merge[0], merge[1])
        merge_nodes_topology(t_graph, merge[0], merge[1])




def size_filtering(t_graph,p_graph,threshold=0,verbose=False):
    ###############################
    # Compute the size of each region
    ###############################
    node2size={}
    considered_nodes=t_graph.nodes() #search_leaf_nodes(graph)
    for n in considered_nodes:
        size=np.count_nonzero(t_graph.get_region(n))
        node2size[n]=size

    nodes_to_remove=[]
    for n in node2size:
        if node2size[n] < threshold:
            nodes_to_remove+=[n]

    if verbose:
        print("Sizes:", sorted(node2size.values()))
        print("--> ", len(nodes_to_remove)," nodes to remove:", nodes_to_remove)
    for n in nodes_to_remove:

        if len(t_graph.successors(n)) == 1:
            father=t_graph.successors(n)[0]
            merge_nodes_topology(t_graph,n,father)
            merge_nodes_photometry(p_graph,n,father)
        elif len(t_graph.successors(n)) == 0: #cas where node to remove is head
            c=topological_merging_candidates(t_graph,n)
            orders,d2m=cost2merge(t_graph, p_graph, set([n]), c)
            merge=orders[0]
            print(c,"merge: ", merge)
            merge_nodes_topology(t_graph,merge[0],merge[1])
            merge_nodes_photometry(p_graph,merge[0],merge[1])
        elif len(t_graph.successors(n)) > 1:
            raise Exception("error")




def remove_smallest_regions(t_graph,p_graph,number=1):
    ###############################
    # Compute the size of each region
    ###############################
    node2size={}
    considered_nodes=t_graph.nodes() #search_leaf_nodes(graph)
    for n in considered_nodes:
        size=np.count_nonzero(t_graph.get_region(n))
        node2size[n]=size
    ###############################
    # First the "number" "smallest" regions (i.e. nodes)
    ###############################
    increasing_sizes=sorted(node2size.values())
    nodes_to_remove=[]
    for i in range(0,number):
        current_size=increasing_sizes[i]
        for e in node2size:
            if (node2size[e]==current_size) and (e not in nodes_to_remove): nodes_to_remove+=[e]
    ###############################
    # Remove the "number" "smallest" regions (i.e. nodes), by merging them with their direct topological father
    ###############################
    for n in nodes_to_remove:
        if len(t_graph.successors(n)) == 1:
            father=t_graph.successors(n)[0]
            merge_nodes_topology(t_graph,n,father)
            merge_nodes_photometry(p_graph,n,father)
        elif len(t_graph.successors(n)) == 0: #cas where node to remove is head
            c=topological_merging_candidates(t_graph,n)
            orders,d2m=cost2merge(t_graph, p_graph, set([n]), c)
            merge=orders[0]
            print(c,"merge: ", merge)
            merge_nodes_topology(t_graph,merge[0],merge[1])
            merge_nodes_photometry(p_graph,merge[0],merge[1])
        elif len(t_graph.successors(n)) > 1:
            raise Exception("error")





