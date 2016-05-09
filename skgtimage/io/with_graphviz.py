#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import networkx as nx
import os,csv
import numpy as np
import scipy as sp;from scipy import misc
import skgtimage as skgti


def matching2links(matching):
    return [ (i,matching[i]) for i in matching]

##############################
# TOP FUNCTION FOR SAVING ALL "MATCHER" CONTENT
##############################
def save_matcher_details(matcher,image=None,labelled_image=None,roi=None,directory=None,save_all_iso=False,slices=[]):
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
        elif len(image.shape) == 3:
            __save_image3d__(image,context_dir+"image/",slices,True)
            __save_image3d__(l_image,context_dir+"image_roi/",slices,True)
    if labelled_image is not None:
        if len(labelled_image.shape) == 2:
            __save_image2d__(labelled_image,os.path.join(context_dir,"labelled_image.png"))
        elif len(labelled_image.shape) == 3:
            __save_image3d__(labelled_image,context_dir+"labelled_image/",slices,True)

    ##############################
    #Saving a priori knowledge
    ##############################
    save_graph(matcher.ref_t_graph,name="ref_topological",directory=directory+"01_apiori/",tree=True)
    save_graph(matcher.ref_p_graph,name="ref_photometric",directory=directory+"01_apiori/",tree=False)
    ##############################
    #Saving built graphs and regions
    ##############################
    save_graph(matcher.built_t_graph,name="topological",directory=directory+"02_built_topology/",tree=True)
    save_graphregions(matcher.built_t_graph,directory=directory+"02_built_topology/",slices=slices)
    save_graph(matcher.built_p_graph,name="photometric",directory=directory+"02_built_photometry/",tree=False)
    save_graphregions(matcher.built_p_graph,directory=directory+"02_built_photometry/",slices=slices)
    save_intensities(matcher.built_p_graph,directory=directory+"02_built_photometry/")
    ##############################
    #Saving filtered built graphs and regions
    ##############################
    save_graph(matcher.query_t_graph,name="topological",directory=directory+"03_filtered_built_topology/",tree=True)
    save_graphregions(matcher.query_t_graph,directory=directory+"03_filtered_built_topology/",slices=slices)
    save_graph(matcher.query_p_graph,name="photometric",directory=directory+"03_filtered_built_photometry/",tree=False)
    save_graphregions(matcher.query_p_graph,directory=directory+"03_filtered_built_photometry/",slices=slices)
    save_intensities(matcher.query_p_graph,directory=directory+"03_filtered_built_photometry/")
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
        #Common isomorphisms
        for i in range(0,len(matcher.common_isomorphisms)):
            matching_links=matching2links(matcher.common_isomorphisms[i])
            save_graph_links(matcher.query_t_graph,matcher.ref_t_graph,[matching_links],['red'],name="2_common_iso_t_"+str(i),directory=directory+"04_matching/",tree=True)
            save_graph_links(matcher.query_p_graph,matcher.ref_p_graph,[matching_links],['red'],name="2_common_iso_p_"+str(i),directory=directory+"04_matching/",tree=True)
        #Energies
        fullfilename=os.path.join(directory+"04_matching/","2_all_energies.csv")
        csv_file=open(fullfilename, "w")
        c_writer = csv.writer(csv_file,dialect='excel')
        c_writer.writerow(["Common iso"]+[i for i in range(0,len(matcher.common_isomorphisms))])
        c_writer.writerow(['Eie dist']+[i for i in matcher.eie_dist])
        c_writer.writerow(['Eie sim']+[i for i in matcher.eie_sim])
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
    mini,maxi=np.min(image),np.max(image)
    if (maxi-mini != 0) and do_rescale:
        tmp_image=(image.astype(np.float)-mini)*(255.0)/(maxi-mini)
    else:
        tmp_image=image
    #Save
    for s in slices:
        current_slice=tmp_image[:,:,s]
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

def save_graph(graph,name,directory=None,tree=True,colored_nodes=[]):
    #To pygraphviz AGraph object
    a=nx.to_agraph(graph)
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
    a=nx.to_agraph(bi_graph)

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
                a.get_edge(links[j][0],links[j][1]).attr['label']=str(labels[j])

    #Hack for plottin with matplotlib -> png -> numpy array -> imshow
    if directory is None:
        a.draw(name+".png") #;a.draw("tmp.svg")
        a.draw(name+".svg") #;a.draw("tmp.svg")
    else:
        if not os.path.exists(directory) : os.mkdir(directory)
        filename=os.path.join(directory,name)
        a.draw(filename+".png") #;a.draw("tmp.svg")
        a.draw(filename+".svg") #;a.draw("tmp.svg")
