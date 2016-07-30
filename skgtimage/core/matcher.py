from skgtimage.core.graph import rename_nodes
from skgtimage.core.filtering import remove_smallest_regions
from skgtimage.core.subisomorphism import best_common_subgraphisomorphism
from skgtimage.core.propagation import propagate,merge_until_commonisomorphism
from skgtimage.core.factory import from_string,from_labelled_image
import time
import copy

def recognize_regions(image,labelled_image,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False,verbose=False):
    #A priori graphs
    t_graph=from_string(t_desc)
    p_graph=from_string(p_desc)
    #Built graphs
    t0=time.clock()
    built_t_graph,built_p_graph=from_labelled_image(image,labelled_image,roi,manage_bounds)
    t1=time.clock()
    #Perform recognition by inexact-graph matching
    matcher=IPMatcher(built_t_graph,built_p_graph,t_graph,p_graph,filtering,t1-t0)
    matcher.compute_maching(verbose)
    matcher.compute_merge()
    matcher.update_final_graph()
    id2r=matcher.get_id2regions()
    #Return
    return id2r,matcher

def matcher_factory(image,labelled_image,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False):
    #A priori graphs
    t_graph=from_string(t_desc)
    p_graph=from_string(p_desc)
    #Built graphs
    t0=time.clock()
    built_t_graph,built_p_graph=from_labelled_image(image,labelled_image,roi,manage_bounds)
    t1=time.clock()
    #Prepare matcher
    matcher=IPMatcher(built_t_graph,built_p_graph,t_graph,p_graph,filtering,t1-t0)
    return matcher

class IPMatcher:
    def __init__(self,built_t_graph,built_p_graph,ref_t_graph,ref_p_graph,filtering=0,build_runtime=0):
        self.built_t_graph=built_t_graph
        self.built_p_graph=built_p_graph
        self.query_t_graph=copy.deepcopy(built_t_graph)
        self.query_p_graph=copy.deepcopy(built_p_graph)

        #INITIAL REFINEMENT
        self.refined_t_graph_intermediates=[]

        # INITIAL FILTERING
        self.filtering=filtering
        if self.filtering != 0 :
            remove_smallest_regions(self.query_t_graph,self.query_p_graph,self.filtering)

        self.ref_t_graph=ref_t_graph
        self.ref_p_graph=ref_p_graph
        #Initial matching and related isomorphisms
        self.t_isomorphisms=None
        self.p_isomorphisms=None
        self.common_isomorphisms=None
        self.eie=None
        self.matching=None
        #Merging
        self.ordered_merges=None

        #Graphs after merges
        self.final_t_graph=None
        self.final_p_graph=None

        #Graphs after renaming final graphs
        self.relabelled_final_t_graph=None
        self.relabelled_final_p_graph=None

        #Runtimes
        self.build_runtime=build_runtime
        self.matching_runtime=0
        self.merging_runtime=0

    def compute_maching(self,verbose=False):
        t0=time.clock()
        self.matching,self.common_isomorphisms,self.t_isomorphisms,self.p_isomorphisms,self.eie=best_common_subgraphisomorphism(self.query_t_graph,
                                                                                                       self.ref_t_graph,
                                                                                                       self.query_p_graph,
                                                                                                       self.ref_p_graph,verbose)
        t1=time.clock()
        if self.common_isomorphisms is None:
            print("Need to initially merge")
            modification, self.refined_t_graph_intermediates= merge_until_commonisomorphism(self.query_t_graph, self.query_p_graph, self.ref_t_graph,self.ref_p_graph, True)
            t0 = time.clock()
            self.matching, self.common_isomorphisms, self.t_isomorphisms, self.p_isomorphisms, self.eie = best_common_subgraphisomorphism(
                self.query_t_graph,
                self.ref_t_graph,
                self.query_p_graph,
                self.ref_p_graph, verbose)
            t1 = time.clock()

        self.matching_runtime=t1-t0
    def compute_merge(self):
        t0=time.clock()
        self.final_t_graph,self.final_p_graph,self.ordered_merges=propagate(self.query_t_graph,
                                                                          self.query_p_graph,
                                                                          self.ref_t_graph,self.ref_p_graph,self.matching)
        t1=time.clock()
        self.merging_runtime=t1-t0


    def update_final_graph(self):
        (self.relabelled_final_t_graph,self.relabelled_final_p_graph)=rename_nodes([self.final_t_graph,self.final_p_graph],self.matching)
        self.relabelled_final_t_graph.set_image(self.query_t_graph.get_image())
        self.relabelled_final_p_graph.set_image(self.query_t_graph.get_image())

    def get_id2regions(self):
        get_id2regions={}
        for n in self.relabelled_final_t_graph.nodes():
            get_id2regions[n]=self.relabelled_final_t_graph.get_region(n)
        return get_id2regions