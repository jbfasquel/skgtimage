#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import networkx as nx
import os,csv
import numpy as np
import scipy as sp;from scipy import misc
import skgtimage as skgti


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


def plot_model(model,segmentations=[]):
    import matplotlib.pyplot as plt
    #First row is for the image and the two graphs -> 3 cols
    nb_cols=1+len(model.p_graphs)
    if model.get_image() is not None: nb_cols+=1
    nb_regions=len(model.t_graph.segmented_nodes())+len(segmentations)
    nb_rows=int(1+np.ceil(float(nb_regions)/nb_cols))
    #Plot the first row
    n_plot=1
    if model.get_image() is not None:
        plt.subplot(nb_rows,nb_cols,n_plot);n_plot+=1
        plt.imshow(model.get_image(),cmap="gray",interpolation="nearest");plt.axis('off')
        plt.title("Image")
    plt.subplot(nb_rows,nb_cols,n_plot);n_plot+=1
    skgti.io.plot_graph(model.t_graph)
    plt.title("Topological graph")
    for i in range(0,len(model.p_graphs)):
        plt.subplot(nb_rows,nb_cols,n_plot+i)
        #skgti.io.plot_graph(model.p_graphs[i],tree=False)
        skgti.io.plot_graph(model.p_graphs[i])
        plt.title("Photometric graph")
    #Start next
    subplot_id=nb_cols
    #Plot segmented regions
    for i in range(0,len(segmentations)):
        subplot_id+=1
        plt.subplot(nb_rows,nb_cols,subplot_id)
        plt.title("Segmentation "+str(i+1))
        plt.imshow(segmentations[i],cmap="gray",interpolation="nearest");plt.axis('off')

    #Plot segmented regions
    #subplot_id=nb_cols+1
    names=sorted(list(model.t_graph.segmented_nodes()))
    for i in range(0,len(names)):
        subplot_id+=1
        plt.subplot(nb_rows,nb_cols,subplot_id)
        name=names[i]
        region=model.get_region(name)
        if type(name)!=str:
            plt.title("Region "+str(name))
        else: plt.title("Region "+name)
        plt.imshow(region,cmap="gray",vmin=0,vmax=1,interpolation="nearest");plt.axis('off')

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
    if specific_nodes is None: specific_nodes=graph.segmented_nodes()
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
    save_to_csv(graph,directory,"mean_intensities")
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
def save_model(directory,tp_model):
    #create dir
    if not os.path.exists(directory) : os.mkdir(directory)
    #image
    image=tp_model.get_image()
    if (image is not None) and (len(image.shape) == 2):
        filename=os.path.join(directory,"image.png");
        sp.misc.imsave(filename, image)
    #save topological graph
    save_graph('topology_apriori',tp_model.t_graph,nodes=[],tree=True,directory=directory,save_regions=True)
    save_graph('topology_context',tp_model.t_graph,nodes=None,tree=True,directory=directory,save_regions=False)
    #save photometric graphs
    for i in range(0,len(tp_model.p_graphs)):
        save_graph('photometry_apriori_'+str(i),tp_model.p_graphs[i],nodes=[],tree=False,directory=directory,save_regions=False)
        save_graph('photometry_context_'+str(i),tp_model.p_graphs[i],nodes=None,tree=False,directory=directory,save_regions=False)

def plot_graph_matching(graph1,graph2,matching,tree=True):
    import matplotlib.pyplot as plt
    save_graph_matching('tmp',graph1,graph2,matching,tree,directory=None)
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

'''
def plot_graph_surjection(graph1,graph2,surjection,tree=True):
    import matplotlib.pyplot as plt
    save_graph_surjection('tmp',graph1,graph2,surjection,tree,directory=None)
    tmp_image=sp.misc.imread("tmp.png")
    os.remove("tmp.png")
    os.remove("tmp.svg")
    plt.imshow(tmp_image);plt.axis('off')


def save_graph_surjection(name,graph1,graph2,surjection,tree=True,directory=None,save_regions=False):
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
    for k in surjection.keys():
        for l in surjection[k]:
            a.add_edge(l,k) #after the layout has been set
            a.get_edge(l,k).attr['color']='red'
            a.get_edge(l,k).attr['splines']='curved'

    #Hack for plottin with matplotlib -> png -> numpy array -> imshow
    if directory is None:
        a.draw(name+".png") #;a.draw("tmp.svg")
        a.draw(name+".svg") #;a.draw("tmp.svg")
    else:
        if not os.path.exists(directory) : os.mkdir(directory)
        filename=os.path.join(directory,name)
        a.draw(filename+".png") #;a.draw("tmp.svg")
        a.draw(filename+".svg") #;a.draw("tmp.svg")
'''