#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause
"""
Truc
"""
import networkx as nx
import numpy as np

def extract_subgraph(built_p,subnodes):
    """
    Return
    :return:
    """
    built_p=built_p.copy()
    nodes_to_remove=set(built_p.nodes())-set(subnodes)
    for n in nodes_to_remove:
        if len(built_p.successors(n)) == 1:
            father=built_p.successors(n)[0]
            top_edge=(n,father)
            built_p.remove_edge(top_edge[0],top_edge[1])
            #Bottom edge
            bottom_edges=[(i,n) for i in built_p.predecessors(n)]
            for e in bottom_edges:
                built_p.remove_edge(e[0],e[1])
                built_p.add_edge(e[0],father)
            built_p.remove_node(n)
        elif len(built_p.successors(n)) == 0:
            if len(built_p.predecessors(n)) !=0:
                pred=built_p.predecessors(n)[0]
                built_p.remove_edge(pred,n)
            built_p.remove_node(n)
    return built_p


def rename_nodes(graphs,matching):
    resulting_graphs=[]
    for g in graphs:
        resulting_graphs+=[nx.relabel_nodes(g,matching)]
    #Assign image
    image=graphs[0].get_image()
    for g in resulting_graphs: g.set_image(image)
    return tuple(resulting_graphs)

def labelled_image2regions(labelled_image,roi=None):
    """
        Generate regions from labelled image: each region correspond to a specific label
    """
    #If explicit ROI (i.e. explicit as not integrated within an image of type np.ma.masked_array
    if roi is not None:
        tmp_masked_array=np.ma.masked_array(labelled_image, mask=np.logical_not(roi))
        return labelled_image2regions(tmp_masked_array)
    #Use histogram to find labels
    regions=[]
    if type(labelled_image) == np.ma.masked_array :
        mask_roi=np.logical_not(labelled_image.mask)
        min_image,max_image=labelled_image.compressed().min(),labelled_image.compressed().max()
        hist,bins = np.histogram(labelled_image.compressed(), bins=max_image-min_image+1,range=(min_image,max_image+1))
        bins=bins[0:bins.size-1]
        for i in range(0,len(hist)):
            if hist[i] != 0:
                new_region=np.where(labelled_image==bins[i],1,0)
                new_region=np.logical_and(mask_roi,new_region)
                regions+=[new_region]
    else:
        min_image,max_image=labelled_image.min(),labelled_image.max()
        hist,bins = np.histogram(labelled_image, bins=max_image-min_image+1,range=(min_image,max_image+1))
        bins=bins[0:bins.size-1]
        for i in range(0,len(hist)):
            if hist[i] != 0: regions+=[np.where(labelled_image==bins[i],1,0)]
    return regions


def get_sub_graphs(graphs,nodes):
    sub_graphs=[]
    for g in graphs:
        sub_graphs+=[transitive_reduction(transitive_closure(g).subgraph(nodes))]
    return sub_graphs


def transitive_reduction_matrix(m):
    g=mat2graph(m)
    g=transitive_reduction(g)
    return graph2mat(g)


def mat2graph(m):
    #Build graph
    g=IrDiGraph()
    #Nodes
    for i in range(0,m.shape[0]): g.add_node(i)
    #Edges
    for i in range(0,m.shape[0]):
        for j in range(0,m.shape[1]):
            if (i != j) and (m[i,j]!=0):
                g.add_edge(i,j)
    return g

def graph2mat(g):
    return nx.adjacency_matrix(g).toarray()

def transitive_closure(g):
    """
    Compute the transitive closure of a graph on IrDiGraph

    TODO: unittest.

    :param g: graph to be reduced (no copy)
    :return: g without transitive closure (being a IrDiGraph)
    """
    import networkx as nx
    nx_g=nx.transitive_closure(g)
    closed_g=IrDiGraph()
    for n in nx_g.nodes():
        closed_g.add_node(n)
    for e in nx_g.edges():
        closed_g.add_edge(e[0],e[1])
    return closed_g


def transitive_reduction(g):
    """
    Compute the transitive reduction of a graph

    TODO: unittest.

    :param g: graph to be reduced (no copy)
    :return: g without transitive closure
    """
    for n1 in g.nodes_iter():
        if g.has_edge(n1, n1):
            g.remove_edge(n1, n1)
        for n2 in g.successors(n1):
            for n3 in g.successors(n2):
                nodes=list(nx.dfs_preorder_nodes(g, n3))
                for n4 in nodes:
                    if g.has_edge(n1, n4):
                        g.remove_edge(n1, n4)
    return g

class IrDiGraph(nx.DiGraph):
    """With automatically assigned attribute 'region' to each node (numpy array, 'None' by default)

    """
    def __init__(self,data=None,image=None,**attr):
        """
        documentation

        :param data: as in networkx.DiGraph.__init__
        :param numpy.array image: image to be analyzed
        :param attr: as in networkx.DiGraph.__init__
        :return: nothing

        :example:
            >>> import skgtimage as sgt
            >>> sgt.core.IrDiGraph() #doctest: +ELLIPSIS
            <skgtimage.core.graph.IrDiGraph object at ...>
        """
        super(IrDiGraph,self).__init__(data,**attr)
        self.__image=image

    def set_image(self,image):
        self.__image=image
    def get_image(self):
        return self.__image

    def add_node(self, n, attr_dict=None, **attr):
        """Add node

        :parameters:
            * n: as as in networkx.DiGraph.add_node
            * attr_dict: as as in networkx.DiGraph.add_node

        """
        nx.DiGraph.add_node(self, n, attr_dict=attr_dict, **attr)
        self.node[n]['region']=None
        self.node[n]['mean_residue']=None

    def update_intensities(self,image):
        from skgtimage.core.photometry import region_stat
        for n in self.nodes():
            #region=self.get_residue(n)
            region=self.get_region(n)
            intensity=region_stat(image,region)
            self.set_mean_residue_intensity(n,intensity)

    def set_mean_residue_intensity(self,n,value):
        self.node[n]['mean_residue']=value

    def get_mean_residue_intensity(self,n):
        return self.node[n]['mean_residue']

    def get_ordered_nodes(self):
        """
        :return: nodes in increasing order of related region mean intensity
        """
        i2n={}
        for n in self.nodes():
            i=self.get_mean_residue_intensity(n)
            if i in i2n: i2n[i]+=[n]
            else: i2n[i]=[n]
            #i2n[self.get_mean_residue_intensity(n)]=n
        ordered=[]
        for i in sorted(i2n):
            ordered+=i2n[i]

        #ordered=[ i2n[i] for i in sorted(i2n)]
        return ordered

    def add_nodes_from(self, n, **attr):
        nx.DiGraph.add_nodes_from(self, n, **attr)
        for i in n: self.node[i]['region']=None
    def set_region(self,n,r):
        """Add node

        :returns: None
        """
        if n not in self.nodes(): raise Exception(str(n)+" not a existing node (declared nodes are: "+str(self.nodes())+")")
        import numpy as np
        self.node[n]['region']=np.copy(r)
        #self.node[n]['region']=r
    def get_region(self,n):
        """
        Return the image region associated with the node name n
        :param n: node name
        :return: image region (numpy array)
        """
        return self.node[n]['region']

    def segmented_nodes(self):
        s=set()
        for n in self.nodes():
            if self.get_region(n) is not None: s = s | set([n])
        return s

    def get_labelled(self):
        labelled=np.zeros(self.get_image().shape,dtype=np.int)
        fill_value=1
        for n in self.nodes():
            region=self.get_region(n)
            labelled=np.ma.masked_array(labelled,mask=region).filled(fill_value)
            fill_value+=1
        return labelled

    def __str__(self):
        chaine=""
        for n in self.nodes():
            chaine+="Node:" + str(n) + " ("+str(self.get_mean_residue_intensity(n))+")"
        return chaine