from skgtimage.core.graph import rename_nodes,transitive_closure
from skgtimage.core.filtering import remove_smallest_regions,size_filtering,merge_filtering,rag_merge_until_commonisomorphism,merge_photometry_gray
from skgtimage.core.subisomorphism import find_subgraph_isomorphims,best_common_subgraphisomorphism,common_subgraphisomorphisms,common_subgraphisomorphisms_optimized,common_subgraphisomorphisms_optimized_v2
from skgtimage.core.propagation import propagate,merge_until_commonisomorphism
from skgtimage.core.factory import from_string,from_labelled_image
from skgtimage.core.background import background_removal_by_iso
from skgtimage.utils import extract_subarray,grey_levels,extract_subarray_rgb
from skgtimage.utils.rag_merging import rag_merge
from skgtimage.utils.color import merge_photometry_color
from skgtimage.io.with_graphviz import __save_image2d__, __save_image3d__, save_image2d_boundaries, \
    save_image3d_boundaries,save_graph,save_intensities,save_graphregions,save_graph_links,matching2links,save_graph_links_v2,save_graph_v2


import time
import copy
import os
import csv
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
                self.label = merge_photometry_color(self.raw_image, self.label, self.roi, times, self.verbose)
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
        common_isomorphisms = common_subgraphisomorphisms_optimized_v2([self.t_graph, self.p_graph], [self.ref_t_graph,self.ref_p_graph])
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
        self.common_isomorphisms = common_subgraphisomorphisms_optimized_v2([self.t_graph, self.p_graph],[self.ref_t_graph, self.ref_p_graph])
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


def clear_dir_content(save_dir):
    if os.path.exists(save_dir):
        for e in os.listdir(save_dir):
            if not(os.path.isdir(save_dir+e)):
                os.remove(save_dir+e)
            else:
                clear_dir_content(save_dir+e+"/")

def save_image_context(image,label,context_dir,roi=None,slices=[],mc=False):
    """
    save image+superimposed labels, roied_label,roied_image+superimposed labels
    :param image:
    :param label:
    :param directory:
    :param roi:
    :param slices:
    :param mc:
    :return:
    """
    if not os.path.exists(context_dir): os.mkdir(context_dir)
    nb = len(grey_levels(label, roi))
    #Case 2D grayscale
    if (len(image.shape) == 2) and (mc is False):
        tmp_image=image
        #if roi is not None: tmp_image = np.ma.array(image.astype(np.float), mask=np.logical_not(roi)).filled(np.min(image) - 1)
        if roi is not None: tmp_image = np.ma.array(image, mask=np.logical_not(roi)).filled(0)
        save_image2d_boundaries(tmp_image, label, directory=context_dir, filename="image_and_"+str(nb)+"_labels")
        __save_image2d__(tmp_image,os.path.join(context_dir,"image.png"))
        __save_image2d__(label,os.path.join(context_dir,"label.png"))
        #Crop
        if roi is not None:
            tmp_image_crop = extract_subarray(tmp_image, roi=roi)
            label_crop=extract_subarray(label, roi=roi)
            save_image2d_boundaries(tmp_image_crop, label_crop, directory=context_dir, filename="image_and_"+str(nb)+"_label_crop")
            __save_image2d__(tmp_image_crop, os.path.join(context_dir, "image_crop.png"))
            __save_image2d__(label_crop, os.path.join(context_dir, "label_crop.png"))

    #Case 2D color
    elif mc is True:
        tmp_image = image
        if roi is not None:
            tmp_roi=np.dstack(tuple([roi for i in range(0,3)]))
            tmp_image = np.ma.array(tmp_image, mask=np.logical_not(tmp_roi)).filled(0)
            __save_image2d__(tmp_image, os.path.join(context_dir, "image_roied.png"))
        save_image2d_boundaries(tmp_image, label, directory=context_dir, filename="image_and_"+str(nb)+"_label")
        if roi is not None:
            tmp_image_crop = extract_subarray_rgb(tmp_image, roi=roi)
            label_crop = extract_subarray(label, roi=roi)
            save_image2d_boundaries(tmp_image_crop, label_crop, directory=context_dir, filename="image_and_"+str(nb)+"_label_crop")
            __save_image2d__(tmp_image_crop, os.path.join(context_dir, "image_crop.png"))
            __save_image2d__(label_crop, os.path.join(context_dir, "label_crop.png"))


    #Case 3D grayscale (with slices)
    elif (len(image.shape) == 3) and (mc is False):
        __save_image3d__(image,context_dir+"image/",slices,True)
        if roi is not None:
            l_image=np.ma.array(image.astype(np.float), mask=np.logical_not(roi))
        __save_image3d__(l_image,context_dir+"image_roi/",slices,True)
        save_image3d_boundaries(l_image, label, directory=context_dir+"image_"+str(nb)+"_label/", slices=slices)
'''
def save_recognizer_report(recognizer,save_dir,algo_info="",seg_runtime=None):
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    #Nb initial labels
    nb_init_labels=len(grey_levels(recognizer.label,roi=recognizer.roi))
    #Nb nodes
    nb_nodes=len(recognizer.t_graph.nodes())
    #Nb iso
    t_isomorphisms_candidates = find_subgraph_isomorphims(transitive_closure(recognizer.t_graph),transitive_closure(recognizer.ref_t_graph))
    nb_t_isos = len(t_isomorphisms_candidates)
    nb_c_isos=len(recognizer.common_isomorphisms)
    #Preliminary
    rag_merging=(recognizer.t_graph_before_rag is not None)
    filtered_size=recognizer.size_min
    bg_removal=recognizer.remove_background
    csv_file = open(save_dir+"stats_procedure.csv", "w")
    c_writer = csv.writer(csv_file, dialect='excel')
    c_writer.writerow(["Context","#Label","#Nodes","#I-isos","#C-isos","RAG ?","Min size ?","Background removal ?"])
    c_writer.writerow([algo_info,str(nb_init_labels),str(nb_nodes),str(nb_t_isos),str(nb_c_isos),str(rag_merging),str(filtered_size),str(bg_removal)])
    csv_file.close()
    ###################
    #Runtimes
    ###################
    #Hack: rerunning graph building after preprocessing steps
    image,labelled,roi=recognizer.image,recognizer.t_graph.get_labelled(),recognizer.roi
    t0=time.clock()
    t,p=from_labelled_image(image,labelled,roi)
    t1 = time.clock()
    #CVS file
    csv_file = open(save_dir + "stats_runtime_old.csv", "w")
    c_writer = csv.writer(csv_file, dialect='excel')
    title_row=[]
    value_row=[]
    if seg_runtime is not None:
        title_row += ["Context","Seg"]
        value_row += [algo_info,np.round(seg_runtime,2)]
    title_row+=["Build"]
    value_row+=[np.round(t1-t0,2)]
    for r in sorted(recognizer.action2runtime):
        title_row+=[r]
        runtime=recognizer.action2runtime[r]
        value_row+=[np.round(runtime,2)]
    c_writer.writerow(title_row)
    c_writer.writerow(value_row)
    csv_file.close()
'''

def save_recognizer_details(recognizer,save_dir,full=False,slices=[]):
    """
    Save details of the recognition procedure: regions, graphs and statistics (mean intensities) related to each step

    :param recognizer: object embedding all details, and returned by recognize
    :param save_dir: directory within which all details are saved
    :param full: if True, all regions, and mean intensities, related to initial graphs are saved (time consuming if many nodes)
    :param slices: list of slice indices to be exported in .png image files, in case of 3D images
    :return: None
    """
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    #####################################
    # Save final result
    #####################################
    if recognizer.relabelled_final_t_graph is not None:
        save_graph(recognizer.relabelled_final_t_graph, name="topological", directory=save_dir, tree=True)
        save_graph(recognizer.relabelled_final_p_graph, name="photometric", directory=save_dir, tree=True)
        save_graphregions(recognizer.relabelled_final_t_graph, directory=save_dir, slices=slices)
        save_intensities(recognizer.relabelled_final_p_graph, directory=save_dir)
        label_from_final = recognizer.relabelled_final_t_graph.get_labelled()
        save_image_context(recognizer.raw_image, label_from_final, save_dir, recognizer.roi, slices=slices, mc=recognizer.mc)

    #############
    #Details
    #############
    details_save_dir=save_dir+"Details/"
    if not os.path.exists(details_save_dir): os.mkdir(details_save_dir)

    for step in range(0,len(recognizer.intermediate_operations)):
        operation=recognizer.intermediate_operations[step]
        tmp_dir=details_save_dir+operation+"/"
        if not os.path.exists(tmp_dir) : os.mkdir(tmp_dir)
        clear_dir_content(tmp_dir)
        label=recognizer.intermediate_labels[step]
        save_image_context(recognizer.raw_image, label, tmp_dir, recognizer.roi, slices=slices,mc=recognizer.mc)
        if recognizer.intermediate_graphs[step] is not None:
            (t_graph,p_graph)=recognizer.intermediate_graphs[step]
            nb_nodes = len(t_graph.nodes())
            save_graph(t_graph, name="g_topological_" + str(nb_nodes), directory=tmp_dir,tree=True)
            save_graph(p_graph, name="g_photometric_" + str(nb_nodes), directory=tmp_dir,tree=True)
            # Simplified
            save_graph_v2(t_graph, name="g_topological_simple_" + str(nb_nodes),directory=tmp_dir, tree=True)
            save_graph_v2(p_graph, name="g_photometric_simple_" + str(nb_nodes), directory=tmp_dir, tree=True)
            if step == (len(recognizer.intermediate_operations)-1):
                save_intensities(p_graph, directory=tmp_dir)
                save_graphregions(t_graph, directory=tmp_dir, slices=slices)

        else:
            if len(grey_levels(label)) < 50:
                t_graph,p_graph=from_labelled_image(recognizer.image,label,recognizer.roi,recognizer.bound_thickness,recognizer.bound_thickness)
                nb_nodes = len(t_graph.nodes())
                save_graph(t_graph, name="g_topological_" + str(nb_nodes), directory=tmp_dir, tree=True)
                save_graph(p_graph, name="g_photometric_" + str(nb_nodes), directory=tmp_dir, tree=True)
                # Simplified
                save_graph_v2(t_graph, name="g_topological_simple_" + str(nb_nodes), directory=tmp_dir, tree=True)
                save_graph_v2(p_graph, name="g_photometric_simple_" + str(nb_nodes), directory=tmp_dir, tree=True)

    #####################################
    # Save common iso and matching
    #####################################
    step_index=len(recognizer.intermediate_operations)
    tmp_dir = details_save_dir + str(step_index)+"_graphs_and_common_isos/";clear_dir_content(tmp_dir)
    clear_dir_content(tmp_dir)
    if recognizer.common_isomorphisms is not None:
        if not os.path.exists(tmp_dir): os.mkdir(tmp_dir)
        t_isomorphisms_candidates = find_subgraph_isomorphims(transitive_closure(recognizer.t_graph), transitive_closure(recognizer.ref_t_graph))
        nb_t_isos=len(t_isomorphisms_candidates)
        f=open(tmp_dir+str(nb_t_isos)+"_t_isos",'w');f.close()
        f=open(tmp_dir+str(len(recognizer.common_isomorphisms))+"_c_isos",'w');f.close()
        if recognizer.matching is not None:
            matching_links = matching2links(recognizer.matching)
            save_graph_links(recognizer.t_graph, recognizer.ref_t_graph, [matching_links], ['red'], name="1_matching_t",directory=tmp_dir, tree=True)
            save_graph_links(recognizer.p_graph, recognizer.ref_p_graph, [matching_links], ['red'], name="1_matching_p",directory=tmp_dir, tree=True)
            #Simplified
            save_graph_links_v2(recognizer.t_graph, recognizer.ref_t_graph, [matching_links], ['red'], name="1_matching_t_simple",directory=tmp_dir, tree=True)
            save_graph_links_v2(recognizer.p_graph, recognizer.ref_p_graph, [matching_links], ['red'],
                                name="1_matching_p_simple", directory=tmp_dir, tree=True)
        if len(recognizer.common_isomorphisms) < 20:
            for i in range(0,len(recognizer.common_isomorphisms)):
                matching_links = matching2links(recognizer.common_isomorphisms[i])
                save_graph_links(recognizer.t_graph, recognizer.ref_t_graph, [matching_links], ['red'],
                         name="common_iso_t_" + str(i), directory=tmp_dir, tree=True)
                save_graph_links(recognizer.p_graph, recognizer.ref_p_graph, [matching_links], ['red'],
                         name="common_iso_p_" + str(i), directory=tmp_dir, tree=True)
        if full:
            save_intensities(recognizer.p_graph, directory=tmp_dir)
            save_graphregions(recognizer.t_graph, directory=tmp_dir, slices=slices)

        #Energies
        csv_file=open(os.path.join(tmp_dir,"2_all_energies.csv"), "w")
        c_writer = csv.writer(csv_file,dialect='excel')
        c_writer.writerow(["Common iso"]+[i for i in range(0,len(recognizer.common_isomorphisms))])
        c_writer.writerow(['Eie']+[i for i in recognizer.eies])
        csv_file.close()


    ##############################
    # Saving merging
    ##############################
    step_index+=1
    tmp_dir = details_save_dir + str(step_index)+"_merging/";clear_dir_content(tmp_dir)
    if (recognizer.matching is not None) and (recognizer.ordered_merges is not None):
        # All merging
        matching_links = matching2links(recognizer.matching)
        save_graph_links(recognizer.t_graph, recognizer.ref_t_graph, [matching_links, recognizer.ordered_merges],
                         ['red', 'green'], label_lists=[[], range(0, len(recognizer.ordered_merges) + 1)], name="matching_t",directory=tmp_dir, tree=True)
        save_graph_links(recognizer.p_graph, recognizer.ref_p_graph, [matching_links, recognizer.ordered_merges],
                         ['red', 'green'], label_lists=[[], range(0, len(recognizer.ordered_merges) + 1)], name="matching_p",
                         directory=tmp_dir, tree=True)
        #Simplified
        save_graph_links_v2(recognizer.t_graph, recognizer.ref_t_graph, [matching_links, recognizer.ordered_merges],
                     ['red', 'green'], label_lists=[[], range(0, len(recognizer.ordered_merges) + 1)],
                     name="matching_t_simplified",
                     directory=tmp_dir, tree=True)
        save_graph_links_v2(recognizer.p_graph, recognizer.ref_p_graph, [matching_links, recognizer.ordered_merges],
                    ['red', 'green'], label_lists=[[], range(0, len(recognizer.ordered_merges) + 1)],
                    name="matching_p_simplified",
                    directory=tmp_dir, tree=True)

