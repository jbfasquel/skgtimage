from skgtimage.core.recognition import recognize_version2,greedy_refinement_v3,greedy_refinement_v4,remove_smallest_leaf_regions,rename_nodes
from skgtimage.core.factory import from_string,from_labelled_image_refactorying
import copy

def recognize_regions(image,labelled_image,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False,verbose=False):
    #A priori graphs
    t_graph=from_string(t_desc)
    p_graph=from_string(p_desc)
    #Built graphs
    built_t_graph,built_p_graph=from_labelled_image_refactorying(image,labelled_image,roi,manage_bounds)
    #Perform recognition by inexact-graph matching
    matcher=IPMatcher(built_t_graph,built_p_graph,t_graph,p_graph,filtering)
    matcher.compute_maching(verbose)
    matcher.compute_merge()
    matcher.update_final_graph()
    id2r=matcher.get_id2regions()
    #Return
    return id2r,matcher
'''
def matcher_factory_refactorying(image,labelled_image,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False):
    #A priori graphs
    t_graph=from_string(t_desc)
    p_graph=from_string(p_desc)
    #Built graphs
    built_t_graph,built_p_graph=from_labelled_image_refactorying(image,labelled_image,roi,manage_bounds)
    #Prepare matcher
    matcher=IPMatcher(built_t_graph,built_p_graph,t_graph,p_graph,filtering)
    return matcher
'''

def matcher_factory(image,labelled_image,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False):
    #A priori graphs
    t_graph=from_string(t_desc)
    p_graph=from_string(p_desc)
    #Built graphs
    built_t_graph,built_p_graph=from_labelled_image_refactorying(image,labelled_image,roi,manage_bounds)
    #Prepare matcher
    matcher=IPMatcher(built_t_graph,built_p_graph,t_graph,p_graph,filtering)
    return matcher

class IPMatcher:
    def __init__(self,built_t_graph,built_p_graph,ref_t_graph,ref_p_graph,filtering=False):
        self.built_t_graph=built_t_graph
        self.built_p_graph=built_p_graph
        self.query_t_graph=copy.deepcopy(built_t_graph)
        self.query_p_graph=copy.deepcopy(built_p_graph)

        self.filtering=filtering
        if self.filtering:
            remove_smallest_leaf_regions(self.query_t_graph,self.query_p_graph)

        self.ref_t_graph=ref_t_graph
        self.ref_p_graph=ref_p_graph
        #Initial matching and related isomorphisms
        self.t_isomorphisms=None
        self.p_isomorphisms=None
        self.common_isomorphisms=None
        self.eie_sim=None
        self.eie_dist=None
        self.matching=None
        #Merging
        self.ordered_merges=None
        self.t_graph_merges=None
        self.p_graph_merges=None
        #Graphs after merges
        self.final_t_graph=None
        self.final_p_graph=None

        #Graphs after renaming final graphs
        self.relabelled_final_t_graph=None
        self.relabelled_final_p_graph=None

    def compute_maching(self,verbose=False):
        self.matching,self.common_isomorphisms,self.t_isomorphisms,self.p_isomorphisms,self.eie_sim,self.eie_dist=recognize_version2(self.query_t_graph,
                                                                                                       self.ref_t_graph,
                                                                                                       self.query_p_graph,
                                                                                                       self.ref_p_graph,verbose)
    def compute_merge(self):

        self.final_t_graph,self.final_p_graph,histo=greedy_refinement_v4(self.query_t_graph,
                                                                          self.query_p_graph,
                                                                          self.ref_t_graph,
                                                                          self.ref_p_graph,self.matching)

        self.t_graph_merges=[i[0] for i in histo]
        self.p_graph_merges=[i[1] for i in histo]
        self.ordered_merges=[i[2] for i in histo]

        #return ordered_list_of_merges


    def update_final_graph(self):
        (self.relabelled_final_t_graph,self.relabelled_final_p_graph)=rename_nodes([self.final_t_graph,self.final_p_graph],self.matching)
        self.relabelled_final_t_graph.set_image(self.query_t_graph.get_image())
        self.relabelled_final_p_graph.set_image(self.query_t_graph.get_image())

    def get_id2regions(self):
        get_id2regions={}
        for n in self.relabelled_final_t_graph.nodes():
            get_id2regions[n]=self.relabelled_final_t_graph.get_region(n)
        return get_id2regions