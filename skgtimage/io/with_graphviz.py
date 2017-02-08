#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import networkx as nx
import os,csv
import numpy as np
import scipy as sp;from scipy import misc
import skgtimage as skgti
import matplotlib.pyplot as plt

def matching2links(matching):
    return [ (i,matching[i]) for i in matching]

##############################
# TOP FUNCTION FOR SAVING ALL "MATCHER" CONTENT
##############################
def save_matcher_result(matcher,image=None,labelled_image=None,roi=None,directory=None,slices=[],mc=False):
    if (mc==True) and (image is not None):
        tmp=0.2125*image[:,:,0]+ 0.7154*image[:,:,1]+0.0721*image[:,:,2]
        return save_matcher_result(matcher, tmp, labelled_image, roi, directory, slices, mc=False)
    ##############################
    #Image and labelled_image
    ##############################
    context_dir=directory+"00_context/"
    if not os.path.exists(context_dir) : os.mkdir(context_dir)
    if image is not None:
        l_image=image
        if roi is not None:
            l_image=np.ma.array(image.astype(np.float), mask=np.logical_not(roi)).filled(np.min(image)-1)
        if len(image.shape) == 2:
            __save_image2d__(image,os.path.join(context_dir,"image.png"))
            __save_image2d__(l_image,os.path.join(context_dir,"image_roi.png"))
            if roi is not None:
                tmp_image_cropped=skgti.utils.extract_subarray(l_image,roi=roi)
                __save_image2d__(tmp_image_cropped, os.path.join(context_dir, "image_crop.png"))

        elif len(image.shape) == 3:
            __save_image3d__(image,context_dir+"image/",slices,True)
            __save_image3d__(l_image,context_dir+"image_roi/",slices,True)
    if labelled_image is not None:
        tmp_labelled_image=labelled_image
        if roi is not None:
            tmp_labelled_image = np.ma.array(labelled_image.astype(np.float), mask=np.logical_not(roi)).filled(np.min(labelled_image) - 1)
        if len(labelled_image.shape) == 2:
            __save_image2d__(tmp_labelled_image, os.path.join(context_dir, "labelled_image.png"))
            if roi is not None:
                tmp_labelled_image_cropped=skgti.utils.extract_subarray(tmp_labelled_image,roi=roi)
                __save_image2d__(tmp_labelled_image_cropped, os.path.join(context_dir, "labelled_image_crop.png"))
        elif len(labelled_image.shape) == 3:
            __save_image3d__(tmp_labelled_image,context_dir+"labelled_image/",slices,True)

    ##############################
    #Image and labelled_image
    ##############################
    if matcher.relabelled_final_t_graph is not None:
        save_graph(matcher.relabelled_final_t_graph, name="topological", directory=directory + "06_final/", tree=True)
        save_graph(matcher.relabelled_final_p_graph, name="photometric", directory=directory + "06_final/", tree=True)
        skgti.io.plot_graph_histogram(matcher.relabelled_final_t_graph, matcher.relabelled_final_p_graph,
                                      True)  # ;plt.show()
        plt.savefig(directory + "06_final/" + "histograms.svg", format="svg", bbox_inches='tight')
        plt.savefig(directory + "06_final/" + "histograms.png", format="png", bbox_inches='tight')
        plt.gcf().clear()
        save_graphregions(matcher.relabelled_final_t_graph, directory=directory + "06_final/", slices=slices)
        save_intensities(matcher.relabelled_final_p_graph, directory=directory + "06_final/")


def save_matcher_details(matcher,image=None,labelled_image=None,roi=None,directory=None,save_all_iso=False,slices=[],mc=False):
    if (mc==True) and (image is not None):
        tmp=0.2125*image[:,:,0]+ 0.7154*image[:,:,1]+0.0721*image[:,:,2]
        return save_matcher_details(matcher, tmp, labelled_image, roi, directory, save_all_iso,slices, mc=False)



    if not os.path.exists(directory) : os.mkdir(directory)

    ##############################
    #Image and labelled_image
    ##############################
    context_dir=directory+"00_context/"
    if not os.path.exists(context_dir) : os.mkdir(context_dir)
    if image is not None:
        l_image=image
        if roi is not None:
            l_image=np.ma.array(image.astype(np.float), mask=np.logical_not(roi)).filled(np.min(image)-1)
        if len(image.shape) == 2:
            __save_image2d__(image,os.path.join(context_dir,"image.png"))
            __save_image2d__(l_image,os.path.join(context_dir,"image_roi.png"))
            if roi is not None:
                tmp_image_cropped=skgti.utils.extract_subarray(l_image,roi=roi)
                __save_image2d__(tmp_image_cropped, os.path.join(context_dir, "image_crop.png"))

        elif len(image.shape) == 3:
            __save_image3d__(image,context_dir+"image/",slices,True)
            __save_image3d__(l_image,context_dir+"image_roi/",slices,True)
    if labelled_image is not None:
        tmp_labelled_image=labelled_image
        if roi is not None:
            tmp_labelled_image = np.ma.array(labelled_image.astype(np.float), mask=np.logical_not(roi)).filled(np.min(labelled_image) - 1)
        if len(labelled_image.shape) == 2:
            __save_image2d__(tmp_labelled_image, os.path.join(context_dir, "labelled_image.png"))
            if roi is not None:
                tmp_labelled_image_cropped=skgti.utils.extract_subarray(tmp_labelled_image,roi=roi)
                __save_image2d__(tmp_labelled_image_cropped, os.path.join(context_dir, "labelled_image_crop.png"))
        elif len(labelled_image.shape) == 3:
            __save_image3d__(tmp_labelled_image,context_dir+"labelled_image/",slices,True)

    ##############################
    #Saving a priori knowledge
    ##############################
    save_graph(matcher.ref_t_graph,name="ref_topological",directory=directory+"01_apiori/",tree=True)
    save_graph(matcher.ref_p_graph,name="ref_photometric",directory=directory+"01_apiori/",tree=False)
    nb_brothers=skgti.core.find_groups_of_brothers(matcher.ref_p_graph)
    if (len(nb_brothers) > 0) and (save_all_iso):
        all_ref_graphs=skgti.core.compute_possible_graphs(matcher.ref_p_graph) #we add a list of n elements (all possible graphs)
        for i in range(0,len(all_ref_graphs)):
            save_graph(all_ref_graphs[i],name="ref_photometric_unwrapped_"+str(i),directory=directory+"01_apiori/",tree=False)

    ##############################
    #Saving built graphs and regions
    ##############################
    save_graph(matcher.built_t_graph,name="topological",directory=directory+"02_built_topology/",tree=True)
    save_graphregions(matcher.built_t_graph,directory=directory+"02_built_topology/",slices=slices)
    save_graph(matcher.built_p_graph,name="photometric",directory=directory+"02_built_photometry/",tree=False)
    skgti.io.plot_graph_histogram(matcher.built_t_graph,matcher.built_p_graph,True)#;plt.show()
    plt.savefig(directory+"02_built_photometry/"+"histograms.svg",format="svg",bbox_inches='tight')
    plt.savefig(directory+"02_built_photometry/"+"histograms.png",format="png",bbox_inches='tight')
    plt.gcf().clear()
    save_graphregions(matcher.built_p_graph,directory=directory+"02_built_photometry/",slices=slices)
    save_intensities(matcher.built_p_graph,directory=directory+"02_built_photometry/")
    ##############################
    #Saving filtered built graphs and regions
    ##############################
    save_graph(matcher.query_t_graph,name="topological",directory=directory+"03_filtered_built_topology/",tree=True)
    save_graphregions(matcher.query_t_graph,directory=directory+"03_filtered_built_topology/",slices=slices)
    save_graph(matcher.query_p_graph,name="photometric",directory=directory+"03_filtered_built_photometry/",tree=False)
    skgti.io.plot_graph_histogram(matcher.query_t_graph,matcher.query_p_graph,True)#;plt.show()
    plt.savefig(directory+"03_filtered_built_photometry/"+"histograms.svg",format="svg",bbox_inches='tight')
    plt.savefig(directory+"03_filtered_built_photometry/"+"histograms.png",format="png",bbox_inches='tight')
    plt.gcf().clear()
    save_graphregions(matcher.query_p_graph,directory=directory+"03_filtered_built_photometry/",slices=slices)
    save_intensities(matcher.query_p_graph,directory=directory+"03_filtered_built_photometry/")
    ##############################
    #Saving intermediate graphs when preliminary merging is performed
    ##############################
    if len(matcher.refined_t_graph_intermediates) != 0:
        merged_dir=directory+"03_merged_built_topology/"
        if not os.path.exists(merged_dir) : os.mkdir(merged_dir)
        for i in range(0,len(matcher.refined_t_graph_intermediates)):
            save_graph(matcher.refined_t_graph_intermediates[i], name="topological", directory=merged_dir + "step_"+str(i)+"/",tree=True)
            save_graphregions(matcher.refined_t_graph_intermediates[i], directory=merged_dir + "step_"+str(i)+"/", slices=slices)

    ##############################
    #Saving process information: nb initial regions, nb region after graph building, nb region after filtering,
    #Nb t_iso, nb p_iso, nb common_iso, cputime (step1, step2, total)...
    ##############################
    context_dir=directory+"06_procedure/"
    if not os.path.exists(context_dir) : os.mkdir(context_dir)

    names=[]
    values=[]
    #Nb initial regions
    nb_initial_regions=len(skgti.core.labelled_image2regions(labelled_image,roi))
    names+=["Nb initial regions"];values+=[nb_initial_regions]
    #A priori nb nodes
    nb_ref_nodes=len(matcher.ref_t_graph.nodes())
    names+=["Nb a priori nodes"];values+=[nb_ref_nodes]
    #A priori brothers
    brother_grps=skgti.core.find_groups_of_brothers(matcher.ref_p_graph)
    nb_sim=0
    for g in brother_grps: nb_sim+=len(g)-1
    names+=["nb_sim"];values+=[nb_sim]
    nb_filtered_regions=len(matcher.query_t_graph.nodes())
    nb_built_regions=len(matcher.built_t_graph.nodes())
    names+=["nb built regions (bef filt"];values+=[str(nb_filtered_regions)+"("+str(nb_built_regions)+")"]
    #Isomorphisms
    nb_c_iso=0
    if matcher.common_isomorphisms is not None: nb_c_iso=len(matcher.common_isomorphisms)
    names+=["Nb common iso"];values+=[nb_c_iso]
    nb_t_iso=0
    if matcher.t_isomorphisms is not None: nb_t_iso=len(matcher.t_isomorphisms)
    names+=["Nb topo iso"];values+=[nb_t_iso]
    nb_p_iso=0
    if matcher.t_isomorphisms is not None: nb_p_iso=len(matcher.p_isomorphisms)
    names+=["Nb photo iso"];values+=[nb_p_iso]

    fullfilename=os.path.join(context_dir,"procedure.csv")
    csv_file=open(fullfilename, "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    c_writer.writerow(names)
    c_writer.writerow(values)
    csv_file.close()


    #Runtimes
    names=[];values=[]
    names+=["Recognition run (sec)"];values+=[np.round(matcher.matching_runtime,2)+np.round(matcher.merging_runtime,2)+np.round(matcher.build_runtime,2)]
    names+=["build run (sec)"];values+=[np.round(matcher.build_runtime,2)]
    names+=["Initial match run (sec)"];values+=[np.round(matcher.matching_runtime,2)]
    names+=["Merging run (sec)"];values+=[np.round(matcher.merging_runtime,2)]

    fullfilename=os.path.join(context_dir,"runtimes.csv")
    csv_file=open(fullfilename, "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    c_writer.writerow(names)
    c_writer.writerow(values)
    csv_file.close()


    ##############################
    #Saving all isomorphisms (if save_all_iso == True)
    ##############################
    if save_all_iso:
        for i in range(0,len(matcher.t_isomorphisms)):
            matching_links=matching2links(matcher.t_isomorphisms[i])
            save_graph_links(matcher.query_t_graph,matcher.ref_t_graph,[matching_links],['red'],name="3_iso_t_"+str(i),directory=directory+"04_matching/",tree=True)
            save_graph_links(matcher.query_p_graph,matcher.ref_p_graph,[matching_links],['red'],name="3_iso_p_"+str(i),directory=directory+"04_matching/",tree=True)
        for i in range(0,len(matcher.p_isomorphisms)):
            matching_links=matching2links(matcher.p_isomorphisms[i])
            save_graph_links(matcher.query_t_graph,matcher.ref_t_graph,[matching_links],['red'],name="3_iso_t_"+str(i),directory=directory+"04_matching/",tree=True)
            save_graph_links(matcher.query_p_graph,matcher.ref_p_graph,[matching_links],['red'],name="3_iso_p_"+str(i),directory=directory+"04_matching/",tree=True)

    ##############################
    #Saving common isomorphisms and related energies
    ##############################
    if matcher.common_isomorphisms is not None:
        tmp_dir=directory+"04_matching/"
        if not os.path.exists(tmp_dir): os.mkdir(tmp_dir)

        #Common isomorphisms
        if len(matcher.common_isomorphisms) < 50:
            for i in range(0,len(matcher.common_isomorphisms)):
                matching_links=matching2links(matcher.common_isomorphisms[i])
                save_graph_links(matcher.query_t_graph,matcher.ref_t_graph,[matching_links],['red'],name="2_common_iso_t_"+str(i),directory=tmp_dir,tree=True)
                save_graph_links(matcher.query_p_graph,matcher.ref_p_graph,[matching_links],['red'],name="2_common_iso_p_"+str(i),directory=tmp_dir,tree=True)
        #Energies
        fullfilename=os.path.join(tmp_dir,"2_all_energies.csv")
        csv_file=open(fullfilename, "w")
        c_writer = csv.writer(csv_file,dialect='excel')
        c_writer.writerow(["Common iso"]+[i for i in range(0,len(matcher.common_isomorphisms))])
        c_writer.writerow(['Eie']+[i for i in matcher.eie])
        csv_file.close()

    ##############################
    #Saving matching
    ##############################
    if matcher.matching is not None:
        matching_links=matching2links(matcher.matching)
        save_graph_links(matcher.query_t_graph,matcher.ref_t_graph,[matching_links],['red'],name="1_matching_t",directory=directory+"04_matching/",tree=True)
        save_graph_links(matcher.query_p_graph,matcher.ref_p_graph,[matching_links],['red'],name="1_matching_p",directory=directory+"04_matching/",tree=True)

    ##############################
    #Saving merging
    ##############################
    if (matcher.matching is not None) and (matcher.ordered_merges is not None):
        #All merging
        matching_links=matching2links(matcher.matching)
        save_graph_links(matcher.query_t_graph,matcher.ref_t_graph,[matching_links,matcher.ordered_merges],['red','green'],label_lists=[[],range(0,len(matcher.ordered_merges)+1)],name="matching_t",directory=directory+"05_merges/",tree=True)
        save_graph_links(matcher.query_p_graph,matcher.ref_p_graph,[matching_links,matcher.ordered_merges],['red','green'],label_lists=[[],range(0,len(matcher.ordered_merges)+1)],name="matching_p",directory=directory+"05_merges/",tree=True)
        #All intermediate graphs

    ##############################
    #Final result
    ##############################
    if matcher.relabelled_final_t_graph is not None:
        save_graph(matcher.relabelled_final_t_graph,name="topological",directory=directory+"06_final/",tree=True)
        save_graph(matcher.relabelled_final_p_graph,name="photometric",directory=directory+"06_final/",tree=True)
        skgti.io.plot_graph_histogram(matcher.relabelled_final_t_graph,matcher.relabelled_final_p_graph,True)#;plt.show()
        plt.savefig(directory+"06_final/"+"histograms.svg",format="svg",bbox_inches='tight')
        plt.savefig(directory+"06_final/"+"histograms.png",format="png",bbox_inches='tight')
        plt.gcf().clear()
        save_graphregions(matcher.relabelled_final_t_graph,directory=directory+"06_final/",slices=slices)
        save_intensities(matcher.relabelled_final_p_graph,directory=directory+"06_final/")
##############################
# FUNCTION FOR DISPLAY
##############################

def plot_graph_links(source_graph,target_graph,link_lists=[],colors=[]):
    """
    Plot graph using graphviz and matplotlib
    :param graph: graph to be plotted
    :return: None
    """
    import matplotlib.pyplot as plt
    save_graph_links(source_graph,target_graph,link_lists,colors,name="tmp")
    tmp_image=sp.misc.imread("tmp.png")
    os.remove("tmp.png");os.remove("tmp.svg")
    plt.imshow(tmp_image);plt.axis('off')


def plot_graph(graph):
    """
    Plot graph using graphviz and matplotlib
    :param graph: graph to be plotted
    :return: None
    """
    import matplotlib.pyplot as plt
    save_graph(graph,name="tmp")
    tmp_image=sp.misc.imread("tmp.png")
    os.remove("tmp.png");os.remove("tmp.svg")
    plt.imshow(tmp_image);plt.axis('off')

def plot_graph_with_regions(graph,nb_rows=1,slice=None):
    """

    :param graph: graph to be plotted, including image and regions
    :param nb_rows: for matplotlib layout
    :param slice: in case of 3D images
    :return: None
    """
    import matplotlib.pyplot as plt
    nb_elts=1+1+len(graph.nodes())
    nb_cols=int(np.ceil(float(nb_elts)/nb_rows))
    #Graph
    n_plot=1
    plt.subplot(nb_rows,nb_cols,n_plot)
    skgti.io.plot_graph(graph);plt.title("Graph")
    #Image
    n_plot+=1
    image=graph.get_image()
    plt.subplot(nb_rows,nb_cols,n_plot)
    plt.imshow(image,"gray");plt.axis('off');plt.title("Image")
    #Regions
    n_plot+=1
    all_nodes=list(sorted(graph.nodes()))
    for i in range(0,len(all_nodes)):
        n=all_nodes[i]
        region=graph.get_region(n)
        plt.subplot(nb_rows,nb_cols,n_plot+i)
        plt.imshow(region,"gray");plt.axis('off');plt.title(str(n))

##############################
# FUNCTION FOR SAVING
##############################
import skimage; from skimage import segmentation;from skimage.future import graph
from skgtimage.utils.evaluation import grey_levels
def save_image2d_boundaries(image,labelled,directory=None,filename="img_bound"):
    if not os.path.exists(directory): os.mkdir(directory)

    nb = len(grey_levels(labelled))
    if np.max(labelled) != np.min(labelled):
        tmp_labelled=(labelled.astype(np.float)-np.min(labelled))*(255.0)/(np.max(labelled)-np.min(labelled)).astype(np.uint8)
    else:
        tmp_labelled=labelled.astype(np.uint8)
    sp.misc.imsave(directory + filename + "_" + str(nb) + "_labels.png", tmp_labelled)
    if len(image.shape) == 2:
        tmp = np.dstack(tuple([image for i in range(0,3)]))
        tmp = skimage.segmentation.mark_boundaries(tmp, labelled)
    else:
        tmp = skimage.segmentation.mark_boundaries(image, labelled)
    sp.misc.imsave(directory + filename + "_" + str(nb) + "_labels_bounds.png", tmp)

def save_image3d_boundaries(image,labelled,directory=None,slices=[]):
    #Directory
    if not os.path.exists(directory) : os.mkdir(directory)
    #Rescale
    '''
    mini,maxi=np.min(image),np.max(image)
    if (maxi-mini != 0) and do_rescale:
        tmp_image=(image.astype(np.float)-mini)*(255.0)/(maxi-mini)
    else:
        tmp_image=image
    '''
    tmp_image=image
    #Save
    for s in slices:
        current_labelled=labelled[:,:,s]
        current_slice=tmp_image[:,:,s]
        mini, maxi = np.min(current_slice), np.max(current_slice)
        if mini != maxi:
            current_slice = (current_slice.astype(np.float) - mini) * (255.0) / (maxi - mini)
        current_slice = current_slice.astype(np.uint8)
        if type(current_slice) == np.ma.MaskedArray:
            current_slice=current_slice.filled(0)
        if type(current_labelled) == np.ma.MaskedArray:
            current_labelled=current_labelled.filled(0)

        current_slice=np.rot90(current_slice)
        current_labelled = np.rot90(current_labelled)
        #filename=os.path.join(directory,"slice_"+str(s)+".png");
        save_image2d_boundaries(current_slice,current_labelled,directory,"slice_"+str(s))


def save_intensities(graph,directory=None,filename="intensities"):
    if not os.path.exists(directory) : os.mkdir(directory)
    csv_file=open(os.path.join(directory,filename+".csv"), "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    for n in graph.nodes():
        c_writer.writerow([n]+[graph.get_mean_residue_intensity(n)])
    csv_file.close()


def __save_image2d__(image,filename,do_rescale=True):
    mini,maxi=np.min(image),np.max(image)
    if (maxi-mini != 0) and do_rescale:
        tmp_image=(image.astype(np.float)-mini)*(255.0)/(maxi-mini)
        sp.misc.imsave(filename, tmp_image.astype(np.uint8))
    else:
        sp.misc.imsave(filename, image.astype(np.uint8))

def __save_image3d__(image,directory,slices=[],do_rescale=True):
    #Directory
    if not os.path.exists(directory) : os.mkdir(directory)
    #Rescale
    '''
    mini,maxi=np.min(image),np.max(image)
    if (maxi-mini != 0) and do_rescale:
        tmp_image=(image.astype(np.float)-mini)*(255.0)/(maxi-mini)
    else:
        tmp_image=image
    '''
    tmp_image=image
    #Save
    for s in slices:
        current_slice=tmp_image[:,:,s]
        mini, maxi = np.min(current_slice), np.max(current_slice)
        if mini != maxi:
            current_slice = (current_slice.astype(np.float) - mini) * (255.0) / (maxi - mini)
        else:
            current_slice = current_slice.astype(np.uint8)
        if type(current_slice) == np.ma.MaskedArray:
            current_slice=current_slice.filled(0)
        current_slice=np.rot90(current_slice)
        filename=os.path.join(directory,"slice_"+str(s)+".png");
        __save_image2d__(current_slice,filename,False)

def save_graphregions(graph,directory=None,slices=[]):
    if directory is not None:
        if not os.path.exists(directory) : os.mkdir(directory)

    for n in graph.nodes():
        current_region=graph.get_region(n)
        if current_region is not None:
            #Save in generic npy format
            filename="region_"+str(n)+".npy"
            if directory is not None: filename=os.path.join(directory,filename);
            np.save(filename,current_region)
            #Save in human viewable formats: png for 2D images, png slices for 3D images
            if len(current_region.shape) == 2:
                filename="region_"+str(n)+".png"
                if directory is not None: filename=os.path.join(directory,filename);
                __save_image2d__(current_region,filename)

            elif len(current_region.shape) == 3:
                slice_dir=directory+"region_"+str(n)+"/"
                if not os.path.exists(slice_dir) : os.mkdir(slice_dir)
                __save_image3d__(current_region,slice_dir,slices,True)

            else: raise Exception("Not a 2D nor a 3D image")

def save_graph_v2(graph,name="graph",directory=None,tree=True,colored_nodes=[]):
    """
    Without labels on graph nodes
    :param graph:
    :param name:
    :param directory:
    :param tree:
    :param colored_nodes:
    :return:
    """
    #To pygraphviz AGraph object
    #a=nx.to_agraph(graph)
    a = nx.nx_agraph.to_agraph(graph)
    #Global layout
    if tree:
        a.graph_attr.update(rankdir='BT') #Bottom to top (default is top to bottom)
        a.graph_attr.update(ranksep=1) #edge length
        a.layout(prog='dot')
    else:
        #a.layout(prog='neato')
        #a.layout(prog='circo')
        a.layout(prog='twopi')
    for n in graph.nodes():
        #a.get_node(n).attr['label']=""
        a.get_node(n).attr['shape'] ='point'
        a.get_node(n).attr['width'] = 0.5
        a.get_node(n).attr['height'] = 0.5
        #a.get_node(n).attr['area'] = 0.1
        #a.get_node(n).attr['margin'] = 0
        #a.get_node(n).attr['cellborder'] = 0.1
        a.get_node(n).attr['fixedsize']=True

    for e in graph.edges():
        tmp=a.get_edge(e[0], e[1])
        tmp.attr['penwidth']=6
        #a.get_edge(e[0],e[1]).attr['length']=1


    #Marking nodes corresponding to, e.g., already segmented regions
    for n in colored_nodes:
        #a.get_node(n).attr['shape']='box' #Shape
        #a.get_node(n).attr['color']='red' #Border
        a.get_node(n).attr['style']='filled';a.get_node(n).attr['fillcolor']='red';

    #Hack for plottin with matplotlib -> png -> numpy array -> imshow
    if directory is None:
        a.draw(name+".png") #;a.draw("tmp.svg")
        a.draw(name+".svg") #;a.draw("tmp.svg")
    else:
        if not os.path.exists(directory) : os.mkdir(directory)
        filename=os.path.join(directory,name)
        a.draw(filename+".png") #;a.draw("tmp.svg")
        a.draw(filename+".svg") #;a.draw("tmp.svg")


def save_graph(graph,name="graph",directory=None,tree=True,colored_nodes=[]):
    #To pygraphviz AGraph object
    #a=nx.to_agraph(graph)
    a = nx.nx_agraph.to_agraph(graph)
    #Global layout
    if tree:
        a.graph_attr.update(rankdir='BT') #Bottom to top (default is top to bottom)
        a.layout(prog='dot')
    else:
        #a.layout(prog='neato')
        #a.layout(prog='circo')
        a.layout(prog='twopi')
    #Marking nodes corresponding to, e.g., already segmented regions
    for n in colored_nodes:
        #a.get_node(n).attr['shape']='box' #Shape
        #a.get_node(n).attr['color']='red' #Border
        a.get_node(n).attr['style']='filled';a.get_node(n).attr['fillcolor']='red';

    #Hack for plottin with matplotlib -> png -> numpy array -> imshow
    if directory is None:
        a.draw(name+".png") #;a.draw("tmp.svg")
        a.draw(name+".svg") #;a.draw("tmp.svg")
    else:
        if not os.path.exists(directory) : os.mkdir(directory)
        filename=os.path.join(directory,name)
        a.draw(filename+".png") #;a.draw("tmp.svg")
        a.draw(filename+".svg") #;a.draw("tmp.svg")

def save_graph_links_v2(source_graph,target_graph,link_lists=[],colors=[],label_lists=[],name="matching",directory=None,tree=True):
    """
    typically invokation (g1,g2,link_lists=[ [(a,b),(c,d)] , [(k,l),(u,v)] ],colors=["red","green"], name=...)
    :param source_graph:
    :param target_graph:
    :param links: new edges (e.g. modeling matching between graph nodes)
    :param colors:
    :param name:
    :param directory:
    :param tree:
    :return:
    """
    #bi_graph=nx.DiGraph()
    bi_graph=nx.MultiDiGraph() #To support multiedges: e.g. link (==edge) corresponding to an existing edge
    bi_graph.add_nodes_from(source_graph)
    bi_graph.add_edges_from(source_graph.edges())
    bi_graph.add_nodes_from(target_graph)
    bi_graph.add_edges_from(target_graph.edges())
    #a=nx.to_agraph(bi_graph)

    a = nx.nx_agraph.to_agraph(bi_graph)

    #Global layout
    if tree:
        a.graph_attr.update(rankdir='BT') #Bottom to top (default is top to bottom)
        a.graph_attr.update(ranksep=1)  # edge length
        a.graph_attr['splines']='spline'
        a.layout(prog='dot')
    else:
        #a.layout(prog='neato')
        #a.layout(prog='circo')
        a.layout(prog='twopi')

    for n in bi_graph.nodes():
        #a.get_node(n).attr['label']=""
        a.get_node(n).attr['shape'] ='point'
        a.get_node(n).attr['width'] = 0.5
        a.get_node(n).attr['height'] = 0.5
        #a.get_node(n).attr['area'] = 0.1
        #a.get_node(n).attr['margin'] = 0
        #a.get_node(n).attr['cellborder'] = 0.1
        a.get_node(n).attr['fixedsize']=True


    for e in bi_graph.edges():
        tmp=a.get_edge(e[0], e[1])
        tmp.attr['penwidth']=6
    for n in target_graph.nodes():
        a.get_node(n).attr['color']='blue'

    for e in target_graph.edges():
        tmp=a.get_edge(e[0], e[1])
        tmp.attr['color']='blue'

    #Marking nodes corresponding to, e.g., already segmented regions
    for i in range(0,len(link_lists)):
        color=colors[i]
        links=link_lists[i]
        labels=None
        if i < len(label_lists): labels=label_lists[i]
        for j in range(0,len(links)):
            a.add_edge(links[j][0],links[j][1]) #after the layout has been set
            a.get_edge(links[j][0],links[j][1]).attr['color']=color
            a.get_edge(links[j][0],links[j][1]).attr['splines']='curved'
            a.get_edge(links[j][0], links[j][1]).attr['penwidth'] = 3
            if (labels is not None) and (j < len(labels)):
                e=a.get_edge(links[j][0],links[j][1])
                #e.attr['label']=str(labels[j])
                e.attr['penwidth'] = 3

    #Hack for plottin with matplotlib -> png -> numpy array -> imshow
    if directory is None:
        a.draw(name+".png") #;a.draw("tmp.svg")
        a.draw(name+".svg") #;a.draw("tmp.svg")
    else:
        if not os.path.exists(directory) : os.mkdir(directory)
        filename=os.path.join(directory,name)
        a.draw(filename+".png") #;a.draw("tmp.svg")
        a.draw(filename+".svg") #;a.draw("tmp.svg")


def save_graph_links(source_graph,target_graph,link_lists=[],colors=[],label_lists=[],name="matching",directory=None,tree=True):
    """
    typically invokation (g1,g2,link_lists=[ [(a,b),(c,d)] , [(k,l),(u,v)] ],colors=["red","green"], name=...)
    :param source_graph:
    :param target_graph:
    :param links: new edges (e.g. modeling matching between graph nodes)
    :param colors:
    :param name:
    :param directory:
    :param tree:
    :return:
    """
    #bi_graph=nx.DiGraph()
    bi_graph=nx.MultiDiGraph() #To support multiedges: e.g. link (==edge) corresponding to an existing edge
    bi_graph.add_nodes_from(source_graph)
    bi_graph.add_edges_from(source_graph.edges())
    bi_graph.add_nodes_from(target_graph)
    bi_graph.add_edges_from(target_graph.edges())
    #a=nx.to_agraph(bi_graph)

    a = nx.nx_agraph.to_agraph(bi_graph)

    #Global layout
    if tree:
        a.graph_attr.update(rankdir='BT') #Bottom to top (default is top to bottom)
        a.graph_attr['splines']='spline'
        a.layout(prog='dot')
    else:
        #a.layout(prog='neato')
        #a.layout(prog='circo')
        a.layout(prog='twopi')
    #Marking nodes corresponding to, e.g., already segmented regions
    for i in range(0,len(link_lists)):
        color=colors[i]
        links=link_lists[i]
        labels=None
        if i < len(label_lists): labels=label_lists[i]
        for j in range(0,len(links)):
            a.add_edge(links[j][0],links[j][1]) #after the layout has been set
            a.get_edge(links[j][0],links[j][1]).attr['color']=color
            a.get_edge(links[j][0],links[j][1]).attr['splines']='curved'
            if (labels is not None) and (j < len(labels)):
                #a.get_edge(links[j][0],links[j][1]).attr['label']=str(labels[j])
                pass

    #Hack for plottin with matplotlib -> png -> numpy array -> imshow
    if directory is None:
        a.draw(name+".png") #;a.draw("tmp.svg")
        a.draw(name+".svg") #;a.draw("tmp.svg")
    else:
        if not os.path.exists(directory) : os.mkdir(directory)
        filename=os.path.join(directory,name)
        a.draw(filename+".png") #;a.draw("tmp.svg")
        a.draw(filename+".svg") #;a.draw("tmp.svg")
