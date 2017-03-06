import numpy as np
from skgtimage.core.search_base import find_head,__recursive_ordering_search__

def decreasing_ordered_nodes(g):
    """
    Only meaning full for photometric graphs
    :param g:
    :return:
    """
    head=find_head(g)
    result=[head]
    __recursive_ordering_search__(g,head,result)
    return result


def best_common_subgraphisomorphism(common_isomorphisms,query_p_graph,ref_p_graph):
    #Computing energies

    eie_per_iso=[]
    for c_iso in common_isomorphisms:
        eie_per_iso+=[matching_criterion_value(query_p_graph,ref_p_graph,c_iso)]
    eie2iso={}
    for i in range(0,len(common_isomorphisms)):
        eie2iso[eie_per_iso[i]]=common_isomorphisms[i]

    #Taking the best common isomorphisms as result
    matching=None
    max_eie=max(eie_per_iso)
    index_of_max=eie_per_iso.index(max_eie)
    matching=common_isomorphisms[index_of_max]

    #Isos in decreasing order of energy
    best_isos=[]
    my_list=sorted(eie2iso)
    my_list.reverse()
    for e in my_list:
        best_isos+=[eie2iso[e]]

    return matching,eie_per_iso,best_isos


def oirelationships(io):
    oi={}
    for i in io:
        my_val=io[i]
        if type(my_val)==str:
            if my_val not in oi:
                oi[my_val]=set([i])
        else:
            for o in my_val:
                if o not in oi: oi[o]=set([i])
                else: oi[o] |= set([i])
    return oi


def matching_criterion_value(query_graph,ref_graph,iso):
    oi=oirelationships(iso)
    list_of_nodes=decreasing_ordered_nodes(ref_graph)
    mean_intensities=[]
    brothers_std_dev_for_each=[]
    #new_brothers_std_dev_for_each = []
    for i in range(0,len(list_of_nodes)):
        current_element=list_of_nodes[i]
        ################################################
        #When brothers: one computes the mean intensity and std of similar nodes
        ################################################
        if len(current_element) > 1:
            local_intensities=[]
            brother_nodes=[list(oi[b])[0] for b in current_element]
            for j in range(0,len(brother_nodes)):
                local_intensities+=[query_graph.get_mean_intensity(brother_nodes[j])]

            mean_intensity=np.mean(np.asarray(local_intensities))
            stddev_intensity=np.std(np.asarray(local_intensities))
            brothers_std_dev_for_each+=[stddev_intensity]
            #new_brothers_std_dev_for_each+=[stddev_intensity]
        ################################################
        #When simple node: one only retrieves the mean intensity
        ################################################
        else:
            tmp=list(current_element)[0]
            target=oi[tmp]
            corresponding_node=list(target)[0]
            mean_intensity=query_graph.get_mean_intensity(corresponding_node)
            #new_brothers_std_dev_for_each += [0.0]
        ################################################
        #Keep intensity
        ################################################
        mean_intensities+=[mean_intensity]

    intensity_diffs=[ np.abs(mean_intensities[i]-mean_intensities[i-1]) for i in range(1,len(mean_intensities))]
    mean_intensity_diff=np.mean(intensity_diffs)
    mean_intensity_dev=np.std(intensity_diffs)
    ##############
    #Variance management
    ##############
    if mean_intensity_dev == 0 : mean_intensity_dev=1
    if len(brothers_std_dev_for_each) == 0 : brothers_std_dev=1
    else:
        brothers_std_dev=np.mean(np.asarray(brothers_std_dev_for_each))
        #new_brothers_std_dev = np.mean(np.asarray(new_brothers_std_dev_for_each))


    ##############
    #Energy: case without similarities
    ##############
    if len(brothers_std_dev_for_each) == 0 :
        eie = mean_intensity_diff / mean_intensity_dev #celui de l'article
        '''
        #new eie: ne marche pas pour liver et k-means
        max_diff=np.abs(mean_intensities[0]-mean_intensities[-1])
        #eie = max_diff / mean_intensity_diff
        eie = max_diff * mean_intensity_diff / mean_intensity_dev
        '''
    ##############
    # Energy: case with similarities
    ##############
    else:
        eie = mean_intensity_diff / brothers_std_dev

    return eie


