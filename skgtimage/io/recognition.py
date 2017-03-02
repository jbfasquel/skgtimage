import os
import csv
from skgtimage.io.with_graphviz import save_graph,save_intensities,save_graphregions,save_graph_links,matching2links,save_graph_links_v2,save_graph_v2
from skgtimage.io.image import save_image_context
from skgtimage.core.photometry import grey_levels
from skgtimage.core.factory import from_labelled_image
from skgtimage.core.subisomorphism import find_subgraph_isomorphims
from skgtimage.core.graph import transitive_closure

def clear_dir_content(save_dir):
    if os.path.exists(save_dir):
        for e in os.listdir(save_dir):
            if not(os.path.isdir(save_dir+e)):
                os.remove(save_dir+e)
            else:
                clear_dir_content(save_dir+e+"/")


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






