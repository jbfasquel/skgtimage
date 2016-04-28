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
    save_graph_refactorying(matcher.ref_t_graph,name="ref_topological",directory=directory+"01_apiori/",tree=True)
    save_graph_refactorying(matcher.ref_p_graph,name="ref_photometric",directory=directory+"01_apiori/",tree=False)
    ##############################
    #Saving built graphs and regions
    ##############################
    save_graph_refactorying(matcher.built_t_graph,name="topological",directory=directory+"02_built_topology/",tree=True)
    save_graphregions_refactorying(matcher.built_t_graph,directory=directory+"02_built_topology/",slices=slices)
    save_graph_refactorying(matcher.built_p_graph,name="photometric",directory=directory+"02_built_photometry/",tree=False)
    save_graphregions_refactorying(matcher.built_p_graph,directory=directory+"02_built_photometry/",slices=slices)
    save_intensities(matcher.built_p_graph,directory=directory+"02_built_photometry/")
    ##############################
    #Saving filtered built graphs and regions
    ##############################
    save_graph_refactorying(matcher.query_t_graph,name="topological",directory=directory+"03_filtered_built_topology/",tree=True)
    save_graphregions_refactorying(matcher.query_t_graph,directory=directory+"03_filtered_built_topology/",slices=slices)
    save_graph_refactorying(matcher.query_p_graph,name="photometric",directory=directory+"03_filtered_built_photometry/",tree=False)
    save_graphregions_refactorying(matcher.query_p_graph,directory=directory+"03_filtered_built_photometry/",slices=slices)
    save_intensities(matcher.query_p_graph,directory=directory+"03_filtered_built_photometry/")
    ##############################
    #Saving all isomorphisms (if save_all_iso == True)
    ##############################
    if save_all_iso:
        for i in range(0,len(matcher.t_isomorphisms)):
            matching_links=matching2links(matcher.t_isomorphisms[i])
            save_graph_links_refactorying(matcher.query_t_graph,matcher.ref_t_graph,[matching_links],['red'],name="3_iso_t_"+str(i),directory=directory+"04_matching/",tree=True)
            save_graph_links_refactorying(matcher.query_p_graph,matcher.ref_p_graph,[matching_links],['red'],name="3_iso_p_"+str(i),directory=directory+"04_matching/",tree=True)
        for i in range(0,len(matcher.p_isomorphisms)):
            matching_links=matching2links(matcher.p_isomorphisms[i])
            save_graph_links_refactorying(matcher.query_t_graph,matcher.ref_t_graph,[matching_links],['red'],name="3_iso_t_"+str(i),directory=directory+"04_matching/",tree=True)
            save_graph_links_refactorying(matcher.query_p_graph,matcher.ref_p_graph,[matching_links],['red'],name="3_iso_p_"+str(i),directory=directory+"04_matching/",tree=True)

    ##############################
    #Saving common isomorphisms and related energies
    ##############################
    #Common isomorphisms
    for i in range(0,len(matcher.common_isomorphisms)):
        matching_links=matching2links(matcher.common_isomorphisms[i])
        save_graph_links_refactorying(matcher.query_t_graph,matcher.ref_t_graph,[matching_links],['red'],name="2_common_iso_t_"+str(i),directory=directory+"04_matching/",tree=True)
        save_graph_links_refactorying(matcher.query_p_graph,matcher.ref_p_graph,[matching_links],['red'],name="2_common_iso_p_"+str(i),directory=directory+"04_matching/",tree=True)
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
    matching_links=matching2links(matcher.matching)
    save_graph_links_refactorying(matcher.query_t_graph,matcher.ref_t_graph,[matching_links],['red'],name="1_matching_t",directory=directory+"04_matching/",tree=True)
    save_graph_links_refactorying(matcher.query_p_graph,matcher.ref_p_graph,[matching_links],['red'],name="1_matching_p",directory=directory+"04_matching/",tree=True)

    ##############################
    #Saving merging
    ##############################
    #All merging
    matching_links=matching2links(matcher.matching)
    save_graph_links_refactorying(matcher.query_t_graph,matcher.ref_t_graph,[matching_links,matcher.ordered_merges],['red','green'],name="matching_t",directory=directory+"05_merges/",tree=True)
    save_graph_links_refactorying(matcher.query_p_graph,matcher.ref_p_graph,[matching_links,matcher.ordered_merges],['red','green'],name="matching_t",directory=directory+"05_merges/",tree=True)
    #All intermediate graphs
    for i in range(0,len(matcher.ordered_merges)):
        save_graph_links_refactorying(matcher.t_graph_merges[i],matcher.ref_t_graph,[matching_links],['red'],name="merging_t_step_"+str(i),directory=directory+"05_merges/",tree=True)
        save_graph_links_refactorying(matcher.p_graph_merges[i],matcher.ref_p_graph,[matching_links],['red'],name="merging_p_step_"+str(i),directory=directory+"05_merges/",tree=True)


    ##############################
    #Final result
    ##############################
    save_graph_refactorying(matcher.relabelled_final_t_graph,name="topological",directory=directory+"06_final/",tree=True)
    save_graph_refactorying(matcher.relabelled_final_p_graph,name="photometric",directory=directory+"06_final/",tree=True)
    save_graphregions_refactorying(matcher.relabelled_final_t_graph,directory=directory+"06_final/",slices=slices)
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
    save_graph_links_refactorying(source_graph,target_graph,link_lists,colors,name="tmp")
    tmp_image=sp.misc.imread("tmp.png")
    os.remove("tmp.png");os.remove("tmp.svg")
    plt.imshow(tmp_image);plt.axis('off')


def plot_graph_refactorying(graph):
    """
    Plot graph using graphviz and matplotlib
    :param graph: graph to be plotted
    :return: None
    """
    import matplotlib.pyplot as plt
    save_graph_refactorying(graph,name="tmp")
    tmp_image=sp.misc.imread("tmp.png")
    os.remove("tmp.png");os.remove("tmp.svg")
    plt.imshow(tmp_image);plt.axis('off')

def plot_graph_with_regions_refactorying(graph,nb_rows=1,slice=None):
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
    skgti.io.plot_graph_refactorying(graph);plt.title("Graph")
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

def save_graphregions_refactorying(graph,directory=None,slices=[]):
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

def save_graph_refactorying(graph,name,directory=None,tree=True,colored_nodes=[]):
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


def save_graph_links_refactorying(source_graph,target_graph,link_lists=[],colors=[],name="matching",directory=None,tree=True):
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
        for link in links:
            a.add_edge(link[0],link[1]) #after the layout has been set
            a.get_edge(link[0],link[1]).attr['color']=color
            a.get_edge(link[0],link[1]).attr['splines']='curved'

    #Hack for plottin with matplotlib -> png -> numpy array -> imshow
    if directory is None:
        a.draw(name+".png") #;a.draw("tmp.svg")
        a.draw(name+".svg") #;a.draw("tmp.svg")
    else:
        if not os.path.exists(directory) : os.mkdir(directory)
        filename=os.path.join(directory,name)
        a.draw(filename+".png") #;a.draw("tmp.svg")
        a.draw(filename+".svg") #;a.draw("tmp.svg")


#############################
# OLD OLD OLD
#############################

def save_to_csv(graph,dir=None,name="intensities"):
    if dir is None:
        fullfilename=name+".csv"
    else:
        fullfilename=os.path.join(dir,name+".csv")
    csv_file=open(fullfilename, "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    for n in graph.nodes():
        c_writer.writerow([n]+[graph.get_mean_residue_intensity(n)])
    csv_file.close()


def plot_graphs_regions_new(graphs,nb_rows=None):
    import matplotlib.pyplot as plt
    #First row is for the image and the two graphs -> 3 cols
    nb_elts=len(graphs)+len(graphs[0].segmented_nodes())
    if nb_rows is None:
        nb_cols=len(graphs)
        #if model.get_image() is not None: nb_cols+=1
        nb_regions=len(graphs[0].segmented_nodes()) #+len(segmentations)
        nb_rows=int(1+np.ceil(float(nb_regions)/nb_cols))
    else:
        nb_cols=int(np.ceil(float(nb_elts)/float(nb_rows)))
    #print(nb_elts,nb_rows,nb_cols)
    #Plot the first row

    '''
    if model.get_image() is not None:
        plt.subplot(nb_rows,nb_cols,n_plot);n_plot+=1
        plt.imshow(model.get_image(),cmap="gray",interpolation="nearest");plt.axis('off')
        plt.title("Image")
    '''
    n_plot=1
    for i in range(0,len(graphs)):
        plt.subplot(nb_rows,nb_cols,n_plot)
        #skgti.io.plot_graph(model.p_graphs[i],tree=False)
        skgti.io.plot_graph(graphs[i])
        plt.title("Graph")
        n_plot+=1

    '''
    plt.subplot(nb_rows,nb_cols,n_plot);n_plot+=1
    skgti.io.plot_graph(model.t_graph)
    plt.title("Topological graph")
    for i in range(0,len(model.p_graphs)):
        plt.subplot(nb_rows,nb_cols,n_plot+i)
        #skgti.io.plot_graph(model.p_graphs[i],tree=False)
        skgti.io.plot_graph(model.p_graphs[i])
        plt.title("Photometric graph")
    '''
    #Start next
    #subplot_id=nb_cols
    '''
    #Plot segmented regions
    for i in range(0,len(segmentations)):
        subplot_id+=1
        plt.subplot(nb_rows,nb_cols,subplot_id)
        plt.title("Segmentation "+str(i+1))
        plt.imshow(segmentations[i],cmap="gray",interpolation="nearest");plt.axis('off')
    '''
    #Plot segmented regions
    #subplot_id=nb_cols+1
    #names=sorted(list(model.t_graph.segmented_nodes()))
    names=sorted(list(graphs[0].segmented_nodes()))
    for i in range(0,len(names)):
        #subplot_id+=1
        plt.subplot(nb_rows,nb_cols,n_plot)
        name=names[i]
        region=graphs[0].get_region(name)
        if type(name)!=str:
            plt.title("Region "+str(name))
        else: plt.title("Region "+name)
        plt.imshow(region,cmap="gray",vmin=0,vmax=1,interpolation="nearest");plt.axis('off')
        n_plot+=1


def plot_graph(graph,nodes=None,tree=True):
    import matplotlib.pyplot as plt
    save_graph('tmp',graph,nodes,tree,directory=None)
    tmp_image=sp.misc.imread("tmp.png")
    os.remove("tmp.png")
    os.remove("tmp.svg")
    plt.imshow(tmp_image);plt.axis('off')

def plot_graphs_regions(graphs,regions,nodes=None,tree=True):
    import matplotlib.pyplot as plt

    nb_cols=3
    nb_regions=len(regions)
    nb_rows=int(len(graphs)+np.ceil(float(nb_regions)/nb_cols))

    current_index=0
    for i in range(0,len(graphs)):
        current_index+=1
        plt.subplot(nb_rows,nb_cols,current_index)
        save_graph('tmp',graphs[i],nodes,tree,directory=None)
        tmp_image=sp.misc.imread("tmp.png")
        os.remove("tmp.png")
        os.remove("tmp.svg")
        plt.imshow(tmp_image);plt.axis('off')

    for i in range(0,len(regions)):
        current_index+=1
        plt.subplot(nb_rows,nb_cols,current_index)
        plt.imshow(regions[i],cmap="gray",vmin=0,vmax=1,interpolation="nearest");plt.axis('off');plt.title("Region "+str(i))




def save_graph(name,graph,nodes=None,tree=True,directory=None,save_regions=False,save_residues=False):
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
    specific_nodes=nodes
    #if specific_nodes is None: specific_nodes=graph.segmented_nodes()
    if specific_nodes is None: specific_nodes=graph.nodes()
    for n in specific_nodes:
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
    #Save mean intensities
    #save_to_csv(graph,directory,"mean_intensities")
    #Save regions
    if save_regions:
        for n in graph.nodes():
            current_region=graph.get_region(n)
            if current_region is not None:
                if len(current_region.shape) == 2:
                    image_uint8=current_region.astype(np.uint8)
                    max=np.max(image_uint8)
                    if max != 255 : image_uint8=255*(image_uint8.astype(np.float)/max).astype(np.uint8)
                    #print(np.max(current_region),current_region)
                    filename="region_"+str(n)+".png"
                    if directory is not None: filename=os.path.join(directory,filename);
                    sp.misc.imsave(filename, image_uint8)
                    #residue
                    if save_residues:
                        residue_image_uint8=(graph.get_residue(n)).astype(np.uint8)
                        max=np.max(residue_image_uint8)
                        if max != 255 : residue_image_uint8=255*(residue_image_uint8.astype(np.float)/max).astype(np.uint8)
                        #print(np.max(current_region),current_region)
                        filename="region_residue_"+str(n)+".png"
                        if directory is not None: filename=os.path.join(directory,filename);
                        sp.misc.imsave(filename, residue_image_uint8)
                '''
                else: raise Exception("Not a 2D image")
                '''

def plot_graph_matching(graph1,graph2,matching,tree=True):
    import matplotlib.pyplot as plt
    save_graph_matching('tmp',graph1,graph2,matching,tree,directory=None)
    tmp_image=sp.misc.imread("tmp.png")
    os.remove("tmp.png")
    os.remove("tmp.svg")
    plt.imshow(tmp_image);plt.axis('off')

def plot_graph_matching_and_merge(graph1,graph2,matching,merges,tree=True):
    import matplotlib.pyplot as plt
    save_graph_matching_and_merge('tmp',graph1,graph2,matching,merges,tree,directory=None)
    tmp_image=sp.misc.imread("tmp.png")
    os.remove("tmp.png")
    os.remove("tmp.svg")
    plt.imshow(tmp_image);plt.axis('off')


def plot_graphs_matching(graphs1,graphs2,matching,tree=True,nb_rows=1,titles=None):
    #nb_rows=1
    nb_cols=np.ceil(float(len(graphs1))/float(nb_rows))
    import matplotlib.pyplot as plt

    for i in range(0,len(graphs1)):
        plt.subplot(nb_rows,nb_cols,i+1)
        g1=graphs1[i]
        g2=graphs2[i]
        if titles is not None: plt.title(titles[i])
        plot_graph_matching(g1,g2,matching,tree)


def plot_graph_matchings(graph1,graph2,matchings,tree=True,nb_rows=1):
    #nb_rows=1
    nb_cols=np.ceil(float(len(matchings))/float(nb_rows))
    import matplotlib.pyplot as plt

    for i in range(0,len(matchings)):
        plt.subplot(nb_rows,nb_cols,i+1)
        plot_graph_matching(graph1,graph2,matchings[i],tree)


def save_graph_matching_and_merge(name,graph1,graph2,matching,merges,tree=True,directory=None,save_regions=False):
    bi_graph=nx.DiGraph()
    bi_graph.add_nodes_from(graph1)
    bi_graph.add_edges_from(graph1.edges())
    bi_graph.add_nodes_from(graph2)
    bi_graph.add_edges_from(graph2.edges())

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
    for k in matching.keys():
        if type(matching[k])!=set:
            a.add_edge(k,matching[k]) #after the layout has been set
            a.get_edge(k,matching[k]).attr['color']='red'
            a.get_edge(k,matching[k]).attr['splines']='curved'
        else:
            for l in matching[k]:
                a.add_edge(k,l) #after the layout has been set
                a.get_edge(k,l).attr['color']='red'
                a.get_edge(k,l).attr['splines']='curved'

    #Merges
    for m in merges:
        a.add_edge(m[0],m[1]) #after the layout has been set
        a.get_edge(m[0],m[1]).attr['color']='green'
        a.get_edge(m[0],m[1]).attr['splines']='curved'


    #Hack for plottin with matplotlib -> png -> numpy array -> imshow
    if directory is None:
        a.draw(name+".png") #;a.draw("tmp.svg")
        a.draw(name+".svg") #;a.draw("tmp.svg")
    else:
        if not os.path.exists(directory) : os.mkdir(directory)
        filename=os.path.join(directory,name)
        a.draw(name+".png") #;a.draw("tmp.svg")
        a.draw(name+".svg") #;a.draw("tmp.svg")

def save_graph_matching(name,graph1,graph2,matching,tree=True,directory=None,save_regions=False):
    bi_graph=nx.DiGraph()
    bi_graph.add_nodes_from(graph1)
    bi_graph.add_edges_from(graph1.edges())
    bi_graph.add_nodes_from(graph2)
    bi_graph.add_edges_from(graph2.edges())

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
    for k in matching.keys():
        if type(matching[k])!=set:
            a.add_edge(k,matching[k]) #after the layout has been set
            a.get_edge(k,matching[k]).attr['color']='red'
            a.get_edge(k,matching[k]).attr['splines']='curved'
        else:
            for l in matching[k]:
                a.add_edge(k,l) #after the layout has been set
                a.get_edge(k,l).attr['color']='red'
                a.get_edge(k,l).attr['splines']='curved'


    #Hack for plottin with matplotlib -> png -> numpy array -> imshow
    if directory is None:
        a.draw(name+".png") #;a.draw("tmp.svg")
        a.draw(name+".svg") #;a.draw("tmp.svg")
    else:
        if not os.path.exists(directory) : os.mkdir(directory)
        filename=os.path.join(directory,name)
        a.draw(name+".png") #;a.draw("tmp.svg")
        a.draw(name+".svg") #;a.draw("tmp.svg")

