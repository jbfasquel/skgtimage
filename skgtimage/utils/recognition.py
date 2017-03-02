from skgtimage.core.graph import rename_nodes
from skgtimage.core.photometry import grey_levels
from skgtimage.core.filtering import size_filtering,merge_photometry_gray,merge_photometry_color
from skgtimage.core.isomorphism import common_subgraphisomorphisms
from skgtimage.core.criterion import best_common_subgraphisomorphism
from skgtimage.core.propagation import propagate
from skgtimage.core.factory import from_string,from_labelled_image
from skgtimage.core.background import background_removal_by_iso
from skgtimage.utils.rag import rag_merge,rag_merge_until_commonisomorphism
from skgtimage.utils.color import rgb2chsv


import time
import copy
import numpy as np

class RecognitionException(Exception):
    def __init__(self,recognizer,message):
        self.recognizer=recognizer
        self.message=message

def recognize(image, label, t_desc, p_desc, mc=False, roi=None, min_size=None, bg=False, bound_thickness=0, rag=None, merge=None, verbose=False):
    """
        Compute and return identified regions, specified in qualitative descriptions (t_desc, p_desc), from the provided over-segmentation (label) of the image (image)

        :param image: input image (numpy array), can be 2D, 3D, grayscale, color
        :param label: input oversegmentation (numpy array)
        :param t_desc: description of inclusion relationships (string)
        :param p_desc: description of photometric relationships (string)
        :param mc: specifies if image is multi-component (True - color in our case) or not (False - grayscale).
        :param roi: region of interest (numpy array), corresponding to non zeros.
        :param min_size: minimum size (in pixels) of considered regions. Regions smaller than min_size are removed.
        :param bg: specifies whether background must be removed
        :param bound_thickness: thickness of the enveloppe surrounding the roi (if roi is not none)
        :param rag: if not None, a preliminary merging of photometrically similar neighboring regions is performed. The parameter specifies the similarity threshold (threshold the in merge_hierarchical function of scikit-image)
        :param merge: if not None, a preliminary merging of photometrically similar regions is performed (not necessarily neighboring regions). The parameter specifies the number of finally expected regions.
        :param verbose: if True, details of the procedure are printed
        :return: a mapping "id - regions" (python mapping type - dictionnary) and the object in charge of managing the entire procedure. "id" are names specified in the description (t_desc, p_desc), regions are "binary images" (numpy array). The object embedded many intermediate informations (e.g. graphs, isomorphisms,...)

    """
    #Create recognizer instance and trigger recognition
    recognizer = Recognizer(image, label, t_desc, p_desc, mc, roi, min_size, bg, bound_thickness, rag, merge, verbose)
    recognizer.process()
    #Retrieve regions and return them for external use not requiring 'recognizer'
    id2regions = {}
    if recognizer.relabelled_final_t_graph is not None:
        for n in recognizer.relabelled_final_t_graph.nodes():
            id2regions[n] = recognizer.relabelled_final_t_graph.get_region(n)
    return id2regions,recognizer


class Recognizer:
    def __init__(self, image,label, t_desc, p_desc,mc=False,roi=None,size_min=None,bg=False,bound_thickness=0,prerag=None,premnoragmerging=None,verbose=False):
        """

        :param image: input image (color or gray, nd)
        :param label: labelled image
        :param t_desc: string
        :param p_desc: string
        :param mc: True is color, False otherwise
        :param roi: region of interest
        :param bound_thickness: thickness of the boundary to be added (none if thickness is 0)
        """
        #Model
        self.t_desc,self.p_desc=t_desc,p_desc
        self.ref_t_graph,self.ref_p_graph=from_string(self.t_desc),from_string(self.p_desc)
        #Context
        self.mc=mc
        self.raw_image=image
        self.image=image
        if mc : #input color image
            self.image=0.2125*image[:,:,0]+ 0.7154*image[:,:,1]+0.0721*image[:,:,2]
        self.label=label
        self.spatial_dim=len(self.label.shape)
        #Optional elements
        self.roi=roi
        self.bound_thickness=bound_thickness
        self.size_min=size_min
        self.remove_background=bg
        self.verbose=verbose
        self.pre_ragmerging=prerag
        self.pre_photmerging=premnoragmerging
        # Produced intermediate labelling (simplification of the initial labelled image)
        self.label_pre_rag= None
        self.label_pre_photomerge = None
        # Produced intermediate operations (strings), labelled image and graphs
        self.intermediate_operations=[]
        self.intermediate_labels=[]
        self.intermediate_graphs=[]
        self.operation_step=0

        self.t_graph, self.p_graph = None, None  # Graphs for computing isomorphisms and merging, can be after rag, filt, background removal
        #Common isomorphisms and best isomorphism
        self.common_isomorphisms=None
        self.matching, self.eies=None,None
        #Merging
        self.ordered_merges=None
        #Final result
        self.final_t_graph,self.final_p_graph=None,None
        self.relabelled_final_t_graph, self.relabelled_final_p_graph = None, None
        #Runtimes
        self.action2runtime={}

    def preliminary_processing(self):

        if self.pre_ragmerging is not None:
            if self.verbose: print("Preprocessing: RAG merging")
            #self.label_pre_rag=np.copy(self.label)
            self.label=rag_merge(self.raw_image, self.label, self.pre_ragmerging, self.mc, self.roi)
            self.intermediate_operations+=[str(self.operation_step)+"_After_preliminary_RAG_merging"]
            self.intermediate_labels+= [np.copy(self.label)]
            self.intermediate_graphs+=[None]
            self.operation_step+=1


        if self.pre_photmerging is not None:
            if self.verbose: print("Preprocessing: Photometry merging")
            #self.label_pre_photomerge = np.copy(self.label)
            nb = len(grey_levels(self.label))
            times = nb - self.pre_photmerging
            if self.mc is True:
                tmp_chsv=rgb2chsv(self.raw_image)
                self.label = merge_photometry_color(tmp_chsv, self.label, self.roi, times, self.verbose)
            else:
                self.label = merge_photometry_gray(self.raw_image, self.label, times)

            self.intermediate_operations += [str(self.operation_step) + "_After_preliminary_Photometry_merging"]
            self.intermediate_labels += [np.copy(self.label)]
            self.intermediate_graphs += [None]
            self.operation_step += 1


    def process(self):
        #Initial status
        self.intermediate_operations+=[str(self.operation_step)+"_Initial"]
        self.intermediate_labels+= [np.copy(self.label)]
        self.intermediate_graphs+=[None]
        self.operation_step+=1

        #Step -1: preliminary rag merging (cut threshold)
        self.preliminary_processing()

        #Step 0: built graphs from label
        self.t_graph, self.p_graph=from_labelled_image(self.image,self.label,self.roi,self.bound_thickness,self.bound_thickness)
        self.intermediate_operations+=[str(self.operation_step)+"_Initial_graph"]
        self.intermediate_labels+= [np.copy(self.label)]
        self.intermediate_graphs+=[(copy.deepcopy(self.t_graph),copy.deepcopy(self.p_graph))]
        self.operation_step+=1

        #Step 1: merge adjacent region until at least one common isomorphism is found
        self.rag_merging()
        #Step 2: size filtering by removing region smaller than self.size_min (in pixels)
        if self.size_min is not None:
            self.filtering()
        #Step 3: remove background
        if self.remove_background:
            self.extract_from_background()
        #Step 4: compute common iso and best iso
        self.compute_common_iso()
        #Step 5: merge region and relabelled graph nodes to identify each expected region
        self.compute_merge()


    def build_graphs(self):
        self.t_graph, self.p_graph=from_labelled_image(self.image,self.label,self.roi,self.bound_thickness,self.bound_thickness)

    def rag_merging(self):
        common_isomorphisms = common_subgraphisomorphisms([self.t_graph, self.p_graph], [self.ref_t_graph, self.ref_p_graph])
        if len(common_isomorphisms) == 0:
            if self.verbose: print("Starting RAG merge until common iso is found...")
            #self.t_graph_before_rag, self.p_graph_before_rag=copy.deepcopy(self.t_graph),copy.deepcopy(self.p_graph)
            try:
                self.t_graph, self.p_graph = rag_merge_until_commonisomorphism(self.t_graph, self.p_graph, self.ref_t_graph,self.ref_p_graph,self.raw_image,self.roi,self.mc,self.verbose)
                self.intermediate_operations += [str(self.operation_step) + "_Initial_graph_after_RAG_for_common_isos"]
                self.label=self.t_graph.get_labelled()
                self.intermediate_labels += [np.copy(self.label)]
                self.intermediate_graphs += [(copy.deepcopy(self.t_graph), copy.deepcopy(self.p_graph))]
                self.operation_step += 1

                if self.verbose: print("Ending RAG merge: common iso is found...")
            except:
                self.t_graph,self.p_graph=None,None
                raise RecognitionException(self,"Unefficient rag merging")

    def filtering(self):
        if self.verbose: print("Filtering by removing regions smaller than ", self.size_min, " pixels")
        #self.t_graph_before_filtering, self.p_graph_before_filtering = copy.deepcopy(self.t_graph), copy.deepcopy(self.p_graph)
        size_filtering(self.t_graph, self.p_graph, self.size_min,self.verbose)

        self.intermediate_operations += [str(self.operation_step) + "_Initial_graph_size_filtered"]
        self.label = self.t_graph.get_labelled()
        self.intermediate_labels += [np.copy(self.label)]
        self.intermediate_graphs += [(copy.deepcopy(self.t_graph), copy.deepcopy(self.p_graph))]
        self.operation_step += 1


    def extract_from_background(self):
        if self.verbose: print("Removing background")
        self.t_graph_before_background, self.p_graph_before_background = copy.deepcopy(self.t_graph), copy.deepcopy(self.p_graph)
        roi, self.t_graph,self.p_graph = background_removal_by_iso(self.image, self.t_graph,self.p_graph, self.ref_t_graph, self.ref_p_graph)

        self.intermediate_operations += [str(self.operation_step) + "_Initial_background_removed"]
        self.label = self.t_graph.get_labelled()
        self.intermediate_labels += [np.copy(self.label)]
        self.intermediate_graphs += [(copy.deepcopy(self.t_graph), copy.deepcopy(self.p_graph))]
        self.operation_step += 1


    def compute_common_iso(self):
        if self.verbose: print("Searching for common isomorphisms")
        t0 = time.clock()
        self.common_isomorphisms = common_subgraphisomorphisms([self.t_graph, self.p_graph], [self.ref_t_graph, self.ref_p_graph])
        t1 = time.clock()
        self.action2runtime["Iso."]=t1-t0

        if self.verbose: print("Searching for the best common isomorphism")
        self.matching, self.eies,_ = best_common_subgraphisomorphism(self.common_isomorphisms, self.p_graph,self.ref_p_graph)


    def compute_merge(self):
        if self.verbose: print("Merging regions")
        t0 = time.clock()
        self.final_t_graph, self.final_p_graph, self.ordered_merges = propagate(self.t_graph,self.p_graph, self.ref_t_graph,self.ref_p_graph, self.matching,verbose=self.verbose)
        t1 = time.clock()
        self.action2runtime["Merge"] = t1 - t0
        (self.relabelled_final_t_graph, self.relabelled_final_p_graph) = rename_nodes([self.final_t_graph, self.final_p_graph], self.matching)
        self.relabelled_final_t_graph.set_image(self.t_graph.get_image())
        self.relabelled_final_p_graph.set_image(self.t_graph.get_image())



