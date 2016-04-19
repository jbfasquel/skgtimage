#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import numpy as np
from skgtimage.core.parameters import classes_for_targets,classes_for_target,roi_for_targets,regions_from_residues,root_tree_node,distinct_classes,interval_for_classes
from skgtimage.core.search_base import number_of_brother_links,recursive_successors,recursive_brothers
from skgtimage.core.search_filtered import  recursive_predecessors,\
                                            recursive_predecessors_until_first_segmented,\
                                            recursive_segmented_brothers,\
                                            first_segmented_predecessors,first_segmented_successors

from skgtimage.core.topology import topological_graph_from_residues
from skgtimage.core.factory import graph_factory
from skgtimage.core.graph import transitive_closure,transitive_reduction
from skgtimage.core.photometry import region_stat,photometric_graph_from_residues,build_similarities
from skgtimage.core.matching import find_subgraph_isomorphims,find_subgraph_isomorphims,find_common_isomorphisms
#from skgtimage.core.recognition import update_graphs_from_identified_regions


import networkx as nx


class TPModel:
    def __init__(self,t_graph=None,p_graphs=[],image=None):
        self.image=image
        self.t_graph=t_graph
        self.p_graphs=p_graphs #list of photometric graphs (one for gray images, three for color images)
        self.__current_targets__=None #can be either a value or a list of values


    def __manage_target_request__(self,requested_targets=None):
        t=requested_targets
        if t is None : t=self.__current_targets__
        #If requested target is also none: one assumes
        #a "single-step" interpreation where root node is the root tree node of the topological graph
        #this root tree node being assoicated to region == numpy.ones(self.image.shape)
        if t is None :
            self.set_targets(list(classes_for_target(self.t_graph)))
            root_topo_node=root_tree_node(self.t_graph)
            '''
            root_topo_node=None
            for n in self.t_graph.nodes():
                succ=recursive_successors(self.t_graph,n)
                if len(succ) == 0: root_topo_node=n
            '''
            self.set_region(root_topo_node,np.ones(self.image.shape))

            #self.set_targets(list(recursive_predecessors_until_first_segmented(self.t_graph,root_topo_node)))
            t=self.__current_targets__
        return t


    def set_topology(self,desc):
        self.t_graph=graph_factory(desc)

    def add_photometry(self,desc):
        self.p_graphs+=[graph_factory(desc)]

    def set_photometry(self,descs):
        self.p_graphs=[graph_factory(d) for d in descs]

    def set_region(self,n,r):
        """Add node

        :returns: None
        """
        self.t_graph.set_region(n,r)
        for g in self.p_graphs: g.set_region(n,r)

    def get_region(self,n):
        """Add node

        :returns: None
        """
        region=self.t_graph.get_region(n)
        return region

    def get_image(self):
        return self.image

    def set_image(self,image):
        self.image=image

    def set_targets(self,t):
        self.__current_targets__=t

    def __spatial_shape__(self):
        return self.image.shape[0:len(self.p_graphs)-1]

    def roi(self,targets=None):
        """

        :param targets:
        :return:
        """
        #Targets
        t=self.__manage_target_request__(targets)
        #roi=roi_for_targets(self.t_graph,t,self.image.shape)
        roi=roi_for_targets(self.t_graph,t,self.__spatial_shape__())
        return roi

    def roi_image(self,targets=None):
        #Targets
        t=self.__manage_target_request__(targets)
        roi=self.roi(t)
        #Depth stack of roi for multicomponent images (gray being mono component)
        roi_mask=np.dstack(tuple([roi for i in range(0,len(self.p_graphs))]))
        roi_image=np.ma.array(self.image, mask=np.logical_not(roi_mask))
        return roi_image

    def __distinct_nodes_for_targets__(self,t):
        #Nodes: classes for targets
        nodes=list(classes_for_targets(self.t_graph,t))
        nodes_of_interest=distinct_classes(nodes,self.p_graphs)
        '''
        #Nodes of interest are none similar nodes (i.e. simultaneously similar for each photometric graph)
        nodes_of_interest=set([nodes[0]])
        for i in range(1,len(nodes)):
            c_node=nodes[i]
            #Find common 'brothers' along each photometric graph
            c_brothers=recursive_brothers(self.p_graphs[0],c_node)
            for component in range(1,len(self.p_graphs)):
                c_brothers &= recursive_brothers(self.p_graphs[component],c_node) #common brother -> set intersection
            #If nodes of interest (in construction during this loop) do already contain a 'brother'
            #then we consider that the current node belongs to 'nodes of interest'
            if len(nodes_of_interest & c_brothers) == 0: nodes_of_interest|=set([c_node])
        '''
        return nodes_of_interest

    def intervals_for_clusters(self,targets=None):
        #Targets
        t=self.__manage_target_request__(targets)
        nodes_of_interest=self.__distinct_nodes_for_targets__(t)
        ####################
        # Case "one component image" (e.g. grayscale)
        # if class2constraint['A']==[[2,2]] -> Mean intensity of unknown region A is close to 2
        # if class2constraint['A']==[[2,4]] -> Mean intensity of unknown region A should lie within [2,4] interval
        ####################
        # Case "multi component image" (e.g. color)
        # if class2constraint['A']==[[2,2],[2,4]]
        #       -> The first component (e.g. red) of the mean intensity of unknown region A is close to 2
        #       -> The second component (e.g. green) of the mean intensity of unknown region A should lie within [2,4] interval
        ####################
        class2constraint=interval_for_classes(self.image,self.roi(),nodes_of_interest,self.t_graph,self.p_graphs)
        return class2constraint

    def number_of_clusters(self,targets=None):
        t=self.__manage_target_request__(targets)
        nodes_of_interest=self.__distinct_nodes_for_targets__(t)
        return len(nodes_of_interest)

    def is_identification_feasible(self,targets=None):
        t=self.__manage_target_request__(targets)

        subgraph_nodes=classes_for_targets(self.t_graph,t)
        ############################
        #Subgraphs
        ############################
        #Topological subgraph
        t_subgraph=self.t_graph.subgraph(subgraph_nodes)
        #Adding photometric graphs
        p_subgraphs=[]
        for g in self.p_graphs: p_subgraphs+=[ g.subgraph(subgraph_nodes) ]

        ############################
        #Number of automorphisms per subgraph
        ############################
        auto=[]
        t_automorphisms=find_subgraph_isomorphims(t_subgraph,t_subgraph)
        auto+=[len(t_automorphisms)]
        for sg in p_subgraphs:
            closed_sg=transitive_closure(sg)
            p_automorphisms=find_subgraph_isomorphims(closed_sg,closed_sg)
            auto+=[len(p_automorphisms)]

        ############################
        #Identification if at least one of the subgraph depicts only one automorphism
        ############################
        answer=(min(auto)==1)
        return answer

    def identify_from_labels(self,labelled_image):
        tmp_label=labelled_image+1 #to avoid confusion with 0s from masked area (roi)
        min_label=np.min(tmp_label)
        max_label=np.max(tmp_label)
        residues=[np.where(tmp_label==i,1,0) for i in range(min_label,max_label+1)]
        self.identify_from_residues(residues)


####################################
# NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW
# PB ON "MATCHING INDEX VS TMP_T_GRAPH"
####################################

    def identify_from_residues(self,residues):
        t=self.__manage_target_request__(None)
        ######################
        #NODES CONSIDERED FROM SUBGRAPHS
        ######################
        subgraph_nodes=classes_for_targets(self.t_graph,t)
        ######################
        #TOPOLOGICAL SUBGRAPH (A PRIORI KNOWLEDGE)
        ######################
        t_subgraph=self.t_graph.subgraph(subgraph_nodes)
        ######################
        #PHOTOMETRIC SUBGRAPHS (A PRIORI KNOWLEDGE - USE OF TRANSITIVE CLOSURE/REDUCTION
        ######################
        p_subgraphs=[]
        for g in self.p_graphs:
            #Subgraph is taken from transitive closure, to avoid loosing connections !!!
            g_closure=transitive_closure(g)
            sub_g_closure=g_closure.subgraph(subgraph_nodes)
            #sub_g=transitive_reduction(sub_g_closure) #WARNING: CYCLES ARE ERASED !!! (NEW REDUCTION ALGORITHMS)
            p_subgraphs+=[ sub_g_closure ]

        ######################
        #Build topological graph from residues
        ######################
        regions,matching=self.__graph_matching_identification__(t_subgraph,p_subgraphs,residues)
        ######################
        #Update graphs (matching.values()) with appropriate regions (matching.keys())
        ######################
        update_graphs_from_identified_regions([self.t_graph]+self.p_graphs,regions,matching)

    def __graph_matching_identification__(self,t_subgraph,p_subgraphs,residues):
        ########################################################
        # Topologic matchings for each photometric subgraph
        ########################################################
        tmp_t_graph,new_residues=topological_graph_from_residues(residues)
        t_isomorphisms=find_subgraph_isomorphims(tmp_t_graph,t_subgraph)
        #Build discovered photometric graph and isomorphisms
        tmp_p_graphs=[]
        for i in range(0, len(p_subgraphs)):
            n=number_of_brother_links(p_subgraphs[i])
            if len(p_subgraphs) == 1: #single component image
                tmp_p_graph=photometric_graph_from_residues(self.get_image(),new_residues)
                build_similarities(self.get_image(),new_residues,tmp_p_graph,n)
                tmp_p_graphs+=[tmp_p_graph]
            else: #Warning: only 2D images are considered
                sub_image=self.get_image()[:,:,i]
                tmp_p_graph=photometric_graph_from_residues(sub_image,new_residues)
                build_similarities(sub_image,new_residues,tmp_p_graph,n)
                tmp_p_graphs+=[tmp_p_graph]
        ########################################################
        # Photometric matchings for each photometric subgraph
        # Matching is performed on transitive closure
        #We compute isomorphism for p_subgraphs iff nodes (from topology) belong
        #to a same connected components (otherwise they are assumed unrelated: no need to consider this matching)
        ########################################################
        p_isomorphisms=[]
        for i in range(0, len(p_subgraphs)):
            ccs=[i for i in nx.connected_components(self.p_graphs[i].to_undirected())]
            is_related=False
            for cc in ccs:
                if set(t_subgraph.nodes()).issubset(cc): is_related=True
            if is_related:
                tmp_p_ref=transitive_closure(p_subgraphs[i])
                tmp_p_graph=transitive_closure(tmp_p_graphs[i])
                #Should check that they are in the same connected component
                p_isomorphisms+=[find_subgraph_isomorphims(tmp_p_graph,tmp_p_ref)]
            else: #to keep a trace of that
                p_isomorphisms+=[None]

        ########################################################
        #Retrieve the list of common isomorphisms between topological and photometric ones
        ########################################################
        all_isomorphisms=[t_isomorphisms]+p_isomorphisms
        matchings=find_common_isomorphisms(all_isomorphisms)
        ########################################################
        #The matching (Checking: the list should contain one element only!!!)
        ########################################################
        if len(matchings) == 0: raise Exception('No matching found: undecidable situation')
        elif len(matchings) > 1: raise Exception('More than one matching found: ambiguity')
        matching=matchings[0]

        ########################################################
        #NOW FILTER IF REQUIRED UNMATCHED NODES ARE DISCOVERED -> ADDING UNMATCHED NODES TO TOPOLOGICAL FATHER IS NOT RELEVANT -> TAKE PHOTOMETRY INTO ACCOUNT
        ########################################################
        nodes_to_be_removed=set()
        for n in tmp_t_graph.nodes():
            if n not in matching.keys(): nodes_to_be_removed|=set([n])

        if len(nodes_to_be_removed) != 0:
            # ADDING UNMATCHED REGION RESIDUES TO FATHER ONES
            for n in nodes_to_be_removed:
                father=tmp_t_graph.successors(n)[0]
                new_residues[father]=np.logical_or(new_residues[father],new_residues[n]).astype(np.uint8)

            # KEEPING MATCHED REGION RESIDUES ONLY
            final_residues=[]
            for i in range(0,len(new_residues)):
                if i not in nodes_to_be_removed: final_residues+=[new_residues[i].astype(np.uint8)]

            #tmp_t_graph,new_residues=topological_graph_from_residues(final_residues)
            return self.__graph_matching_identification__(t_subgraph,p_subgraphs,final_residues)

        else:
            regions=regions_from_residues(tmp_t_graph,new_residues)
            return regions,matching
        ########################################################
        #Build regions from tmp_t_graph and residues
        ########################################################


####################################
# OLD OLD OLD OLD OLD
####################################
    '''
    def identify_from_residues(self,residues):
        t=self.__manage_target_request__(None)
        ######################
        #NODES CONSIDERED FROM SUBGRAPHS
        ######################
        subgraph_nodes=classes_for_targets(self.t_graph,t)
        ######################
        #TOPOLOGICAL SUBGRAPH
        ######################
        t_subgraph=self.t_graph.subgraph(subgraph_nodes)
        ######################
        #PHOTOMETRIC SUBGRAPHS (USE OF TRANSITIVE CLOSURE/REDUCTION)
        ######################
        p_subgraphs=[]
        for g in self.p_graphs:
            #Subgraph is taken from transitive closure, to avoid loosing connections !!!
            g_closure=transitive_closure(g)
            sub_g_closure=g_closure.subgraph(subgraph_nodes)
            #Transitive reduction to recover "simplified" photometric graph
            #sub_g=transitive_reduction(sub_g_closure) #WARNING: CYCLES ARE ERASED !!! (NEW REDUCTION ALGORITHMS)
            #sub_g=sub_g_closure
            p_subgraphs+=[ sub_g_closure ]

        ######################
        #Build topological graph from residues
        ######################
        tmp_t_graph,new_residues=topological_graph_from_residues(residues)
        #self.__tmp_graph_matching_identification__(t_subgraph,p_subgraphs,tmp_t_graph,new_residues)
        if len(tmp_t_graph.nodes()) == len(t_subgraph.nodes()):
            self.__graph_matching_identification__(t_subgraph,p_subgraphs,tmp_t_graph,new_residues)
        else:
            self.__subgraph_matching_identification__(t_subgraph,p_subgraphs,tmp_t_graph,new_residues)
    '''
    '''
    def tmp_identify_from_labels(self,labelled_image):
        tmp_label=labelled_image+1 #to avoid confusion with 0s from masked area (roi)
        min_label=np.min(tmp_label)
        max_label=np.max(tmp_label)
        residues=[np.where(tmp_label==i,1,0) for i in range(min_label,max_label+1)]
        self.tmp_identify_from_residues(residues)
    '''

    '''
    def __graph_matching_identification__(self,t_subgraph,p_subgraphs,tmp_t_graph,new_residues):
        ########################################################
        # Topologic matchings for each photometric subgraph
        ########################################################
        t_isomorphisms=find_isomorphims(tmp_t_graph,t_subgraph)
        #Build discovered photometric graph and isomorphisms
        tmp_p_graphs=[]
        for i in range(0, len(p_subgraphs)):
            n=number_of_brother_links(p_subgraphs[i])
            if len(p_subgraphs) == 1: #single component image
                tmp_p_graphs+=[photometric_graph_from_residues(self.get_image(),new_residues,n)]
            else: #Warning: only 2D images are considered
                sub_image=self.get_image()[:,:,i]
                tmp_p_graphs+=[photometric_graph_from_residues(sub_image,new_residues,n)]
        ########################################################
        # Photometric matchings for each photometric subgraph
        # Matching is performed on transitive closure
        #We compute isomorphism for p_subgraphs iff nodes (from topology) belong
        #to a same connected components (otherwise they are assumed unrelated: no need to consider this matching)
        ########################################################
        p_isomorphisms=[]
        t_subnodes=set(t_subgraph.nodes())
        for i in range(0, len(p_subgraphs)):
            ccs=[i for i in nx.connected_components(self.p_graphs[i].to_undirected())]
            is_related=False
            for cc in ccs:
                #if len(t_subnodes & cc)==len(t_subnodes):
                if t_subnodes.issubset(cc): is_related=True
            if is_related:
                tmp_p_ref=transitive_closure(p_subgraphs[i])
                tmp_p_graph=transitive_closure(tmp_p_graphs[i])
                #Should check that they are in the same connected component
                p_isomorphisms+=[find_isomorphims(tmp_p_graph,tmp_p_ref)]
            else: #to keep a trace of that
                p_isomorphisms+=[None]

        ########################################################
        #Retrieve the list of common isomorphisms between topological and photometric ones
        ########################################################
        all_isomorphisms=[t_isomorphisms]+p_isomorphisms
        matchings=find_common_isomorphisms(all_isomorphisms)
        ########################################################
        #The matching (Checking: the list should contain one element only!!!)
        ########################################################
        if len(matchings) == 0: raise Exception('No matching found: undecidable situation')
        elif len(matchings) > 1: raise Exception('More than one matching found: ambiguity')
        matching=matchings[0]

        ########################################################
        #Build regions from tmp_t_graph and residues
        ########################################################
        regions=regions_from_residues(tmp_t_graph,new_residues)

        ########################################################
        #Update graphs
        ########################################################
        #update_graphs_from_identified_regions(self.t_graph,self.p_graphs,regions,matching)
        update_graphs_from_identified_regions([self.t_graph]+self.p_graphs,regions,matching)

    def __subgraph_matching_identification__(self,t_subgraph,p_subgraphs,tmp_t_graph,new_residues):
        ############################
        # TOPOLOGICAL SUBISOMORPHISMS
        ############################
        t_sub_isomorphisms=find_sub_isomorphims(tmp_t_graph,t_subgraph)
        if len(t_sub_isomorphisms) != 1: raise Exception('No subgraph matching found: undecidable situation')
        #Subisomorphism
        sub_isomorphism=t_sub_isomorphisms[0]
        #Nodes to be removed
        nodes_to_be_removed=set()
        for n in tmp_t_graph.nodes():
            if n not in sub_isomorphism.keys(): nodes_to_be_removed|=set([n])

        ############################
        # ADDING UNMATCHED REGION RESIDUES TO FATHER ONES
        ############################
        for n in nodes_to_be_removed:
            father=tmp_t_graph.successors(n)[0]
            new_residues[father]=np.logical_or(new_residues[father],new_residues[n]).astype(np.uint8)

        ############################
        # KEEPING MATCHED REGION RESIDUES ONLY
        ############################
        final_residues=[]
        for i in range(0,len(new_residues)):
            if i not in nodes_to_be_removed: final_residues+=[new_residues[i]]

        ############################
        # UPDATING THE GRAPH
        ############################
        tmp_t_graph,final_residues=topological_graph_from_residues(final_residues)

        ############################
        # EXACT GRAPH MATCHING
        ############################
        self.__graph_matching_identification__(t_subgraph,p_subgraphs,tmp_t_graph,final_residues)
    '''