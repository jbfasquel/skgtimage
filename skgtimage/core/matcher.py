from skgtimage.core.recognition import recognize_version2,greedy_refinement_v3,rename_nodes
from skgtimage.core.factory import from_string,from_labelled_image

def recognize_regions(image,labelled_image,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,verbose=False):
    #A priori graphs
    t_graph=from_string(t_desc)
    p_graph=from_string(p_desc)
    #Built graphs
    built_t_graph,built_p_graph=from_labelled_image(image,labelled_image,roi,manage_bounds)
    #Perform recognition by inexact-graph matching
    matcher=IPMatcher(built_t_graph,built_p_graph,t_graph,p_graph)
    matcher.compute_maching(verbose)
    matcher.compute_merge()
    matcher.update_final_graph()
    id2r=matcher.get_id2regions()
    #Return
    return id2r,matcher



class IPMatcher:
    def __init__(self,built_t_graph,built_p_graph,ref_t_graph,ref_p_graph):
        self.built_t_graph=built_t_graph
        self.built_p_graph=built_p_graph
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
        self.matching,self.common_isomorphisms,self.t_isomorphisms,self.p_isomorphisms,self.eie_sim,self.eie_dist=recognize_version2(self.built_t_graph,
                                                                                                       self.ref_t_graph,
                                                                                                       self.built_p_graph,
                                                                                                       self.ref_p_graph,verbose)
    def compute_merge(self):
        self.final_t_graph,self.final_p_graph,histo=greedy_refinement_v3(self.built_t_graph,
                                                                          self.built_p_graph,
                                                                          self.ref_t_graph,
                                                                          self.ref_p_graph,self.matching)
        self.t_graph_merges=[i[0] for i in histo]
        self.p_graph_merges=[i[1] for i in histo]
        self.ordered_merges=[i[2] for i in histo]

        #return ordered_list_of_merges


    def update_final_graph(self):
        (self.relabelled_final_t_graph,self.relabelled_final_p_graph)=rename_nodes([self.final_t_graph,self.final_p_graph],self.matching)

    def get_id2regions(self):
        get_id2regions={}
        for n in self.relabelled_final_t_graph.nodes():
            get_id2regions[n]=self.relabelled_final_t_graph.get_region(n)
        return get_id2regions