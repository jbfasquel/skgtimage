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
        c_writer.writerow([n] + [graph.get_mean_intensity(n)])
    csv_file.close()

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
                skgti.io.image.__save_image2d__(current_region,filename)

            elif len(current_region.shape) == 3:
                slice_dir=directory+"region_"+str(n)+"/"
                if not os.path.exists(slice_dir) : os.mkdir(slice_dir)
                skgti.io.image.__save_image3d__(current_region,slice_dir,slices,True)

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
    try:
        a = nx.nx_agraph.to_agraph(graph)
    except Exception as m:
        print("No pygraphviz installed")
        print(m)
        return

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
        a.get_node(n).attr['shape'] ='point'
        a.get_node(n).attr['width'] = 0.5
        a.get_node(n).attr['height'] = 0.5
        a.get_node(n).attr['fixedsize']=True

    for e in graph.edges():
        tmp=a.get_edge(e[0], e[1])
        tmp.attr['penwidth']=6
        #a.get_edge(e[0],e[1]).attr['length']=1


    #Marking nodes corresponding to, e.g., already segmented regions
    for n in colored_nodes:
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
    try:
        a = nx.nx_agraph.to_agraph(graph)
    except Exception as m:
        print("No pygraphviz installed")
        print(m)
        return
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
    try:
        a = nx.nx_agraph.to_agraph(bi_graph)
    except Exception as m:
        print("No pygraphviz installed")
        print(m)
        return

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
    try:
        a = nx.nx_agraph.to_agraph(bi_graph)
    except Exception as m:
        print("No pygraphviz installed")
        print(m)
        return


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
