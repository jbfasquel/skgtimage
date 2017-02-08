from skgtimage.core.graph import rename_nodes
from skgtimage.core.filtering import remove_smallest_regions,size_filtering,merge_filtering
from skgtimage.core.subisomorphism import best_common_subgraphisomorphism,common_subgraphisomorphisms,common_subgraphisomorphisms_optimized,common_subgraphisomorphisms_optimized_v2
from skgtimage.core.propagation import propagate,merge_until_commonisomorphism
from skgtimage.core.factory import from_string,from_labelled_image
from skgtimage.core.background import background_removal_by_iso
import time
import copy
import numpy as np

class matcher_exception(Exception):
    def __init__(self,matcher):
        self.matcher=matcher


#def recognize_regions(image,labelled_image,t_desc,p_desc,roi=None,manage_bounds=False,thickness=1,filtering=False,verbose=False,bf=False,filter_size=0,filter_merge=0,background=False,mc=False):
def recognize_regions(image,labelled_image,t_desc,p_desc,roi=None,manage_bounds=False,thickness=1,verbose=False,bf=False,filter_size=0,filter_merge=0,background=False,mc=False):
    #Color to gray if required
    if mc:
        tmp=0.2125*image[:,:,0]+ 0.7154*image[:,:,1]+0.0721*image[:,:,2]
        #return recognize_regions(tmp, labelled_image, t_desc, p_desc, roi, manage_bounds, thickness,filtering, verbose, bf, filter_size, filter_merge, background,mc=False)
        return recognize_regions(tmp, labelled_image, t_desc, p_desc, roi, manage_bounds, thickness, verbose, bf,
                             filter_size, filter_merge, background, mc=False)

    #A priori graphs
    t_graph=from_string(t_desc)
    p_graph=from_string(p_desc)
    #Built graphs
    t0=time.clock()
    #built_t_graph,built_p_graph=from_labelled_image(image,labelled_image,roi,manage_bounds)
    built_t_graph, built_p_graph =from_labelled_image(image, labelled_image, roi, manage_bounds,thickness)
    if background:
        #roi, built_t_graph, built_p_graph = background_removal_by_iso(image, labelled_image, t_graph, p_graph)
        roi, built_t_graph, built_p_graph = background_removal_by_iso(image, built_t_graph,built_p_graph, t_graph, p_graph)

    t1=time.clock()
    #Perform recognition by inexact-graph matching
    #matcher=IPMatcher(built_t_graph,built_p_graph,t_graph,p_graph,filtering,t1-t0,bf=bf,filter_size=filter_size,filter_merge=filter_merge)
    matcher = IPMatcher(built_t_graph, built_p_graph, t_graph, p_graph, t1 - t0, bf=bf,filter_size=filter_size, filter_merge=filter_merge)
    matcher.compute_maching(verbose)
    matcher.compute_merge()
    matcher.update_final_graph()
    if manage_bounds:
        for n in matcher.relabelled_final_t_graph.nodes():
            print("Logical and with ROI and ",n)
            region=matcher.relabelled_final_t_graph.get_region(n)
            new_region=np.logical_and(region,roi)
            matcher.relabelled_final_t_graph.set_region(n,new_region)
            matcher.relabelled_final_p_graph.set_region(n,new_region)
    id2r=matcher.get_id2regions()
    #Return
    return id2r,matcher

def matcher_factory(image,labelled_image,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False):
    #A priori graphs
    t_graph=from_string(t_desc)
    p_graph=from_string(p_desc)
    #Built graphs
    t0=time.clock()
    #built_t_graph,built_p_graph=from_labelled_image(image,labelled_image,roi,manage_bounds)
    built_t_graph, built_p_graph = from_labelled_image(image, labelled_image, roi, manage_bounds)
    t1=time.clock()
    #Prepare matcher
    matcher=IPMatcher(built_t_graph,built_p_graph,t_graph,p_graph,filtering,t1-t0)
    return matcher

class IPMatcher:
#    def __init__(self,built_t_graph,built_p_graph,ref_t_graph,ref_p_graph,filtering=0,build_runtime=0,bf=False,filter_size=0,filter_merge=0):
    def __init__(self, built_t_graph, built_p_graph, ref_t_graph, ref_p_graph, build_runtime=0, bf=False,filter_size=0, filter_merge=0):
        self.built_t_graph=built_t_graph
        self.built_p_graph=built_p_graph
        self.query_t_graph=copy.deepcopy(built_t_graph)
        self.query_p_graph=copy.deepcopy(built_p_graph)

        #INITIAL REFINEMENT
        self.refined_t_graph_intermediates=[]

        #COMMON ISOMORPHISMS SEARCH: BRUTE FORCE (bf==True) or optimized
        self.bf=bf

        #ISO in decreasing order of energy
        self.isos=None
        self.current_iso_index=0
        # INITIAL FILTERING
        '''
        self.filtering=filtering
        if self.filtering != 0 :
            remove_smallest_regions(self.query_t_graph,self.query_p_graph,self.filtering)
        '''
        if filter_size !=0 :
            size_filtering(self.query_t_graph,self.query_p_graph,filter_size)
        if filter_merge !=0 :
            merge_filtering(self.query_t_graph,self.query_p_graph,filter_merge)

        self.ref_t_graph=ref_t_graph
        self.ref_p_graph=ref_p_graph
        #Initial matching and related isomorphisms
        self.t_isomorphisms=[]
        self.p_isomorphisms=[]
        self.common_isomorphisms=[]
        self.eie=[]
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

        if self.bf:
            self.common_isomorphisms, isomorphisms_per_graph = common_subgraphisomorphisms([self.query_t_graph, self.query_p_graph],[self.ref_t_graph, self.ref_p_graph])
            self.t_isomorphisms,self.p_isomorphisms=isomorphisms_per_graph[0],isomorphisms_per_graph[1]
        else:
            #self.common_isomorphisms = common_subgraphisomorphisms_optimized([self.query_t_graph, self.query_p_graph], [self.ref_t_graph, self.ref_p_graph])
            self.common_isomorphisms = common_subgraphisomorphisms_optimized_v2([self.query_t_graph, self.query_p_graph],
                                                                             [self.ref_t_graph, self.ref_p_graph])
        t1=time.clock()


        if len(self.common_isomorphisms) == 0 :
            print("Start merge until at least one common iso is found")
            try :
                modification, self.refined_t_graph_intermediates= merge_until_commonisomorphism(self.query_t_graph, self.query_p_graph, self.ref_t_graph,self.ref_p_graph, True)
                print("Stop merge until at least one common iso is found: at least one has been found")
                print(len(self.query_t_graph.nodes()))
            except :
                raise matcher_exception(self)
            self.compute_maching(verbose)

        self.matching_runtime=t1-t0

        self.matching, self.eie,self.isos = best_common_subgraphisomorphism(self.common_isomorphisms, self.query_p_graph,self.ref_p_graph, verbose)

    def compute_merge(self):
        t0=time.clock()
        self.final_t_graph, self.final_p_graph, self.ordered_merges = propagate(self.query_t_graph,
                                                                                self.query_p_graph, self.ref_t_graph,
                                                                                self.ref_p_graph, self.matching)

        '''
        current_iso=self.isos[self.current_iso_index]
        try:
            self.final_t_graph,self.final_p_graph,self.ordered_merges=propagate(self.query_t_graph,
                                                                          self.query_p_graph,self.ref_t_graph,self.ref_p_graph,current_iso)
        except Exception:
            self.current_iso_index+=1
            print("Failed current iso: try next -> ", self.current_iso_index," / ",len(self.isos))
            self.compute_merge()
        '''

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