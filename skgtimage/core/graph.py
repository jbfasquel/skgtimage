#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause
"""
Truc
"""
import networkx as nx
import numpy as np

def downsample(g, d):
    """
    Only for graphs involving 2D grayscale images
    :param g:
    :param d:
    :return:
    """
    g.set_image(g.get_image()[::d, ::d])
    for n in g.nodes():
        g.set_region(n,g.get_region(n)[::d, ::d])
        #self.set_region(n, region)
    g.update_intensities(g.get_image())

def get_node2mean(g, round=False):
    mapping = {}
    for n in g.nodes():
        intensity = g.get_mean_intensity(n)
        if round: intensity = np.round(intensity, 0)
        mapping[n] = intensity
    return mapping

def get_ordered_nodes(g):
    """
    :return: nodes in increasing order of related region mean intensity
    """
    i2n={}
    for n in g.nodes():
        i=g.get_mean_intensity(n)
        if i in i2n: i2n[i]+=[n]
        else: i2n[i]=[n]
    ordered=[]
    for i in sorted(i2n):
        ordered+=i2n[i]

    return ordered

def relabel_nodes(graphs, matching):
    resulting_graphs=[]
    for g in graphs:
        resulting_graphs+=[nx.relabel_nodes(g,matching)]
    #Assign image
    image=graphs[0].get_image()
    for g in resulting_graphs: g.set_image(image)
    return tuple(resulting_graphs)

def transitive_reduction_adjacency_matrix(m):
    g=IrDiGraph()
    #Nodes
    for i in range(0,m.shape[0]): g.add_node(i)
    #Edges
    for i in range(0,m.shape[0]):
        for j in range(0,m.shape[1]):
            if (i != j) and (m[i,j]!=0):
                g.add_edge(i,j)
    #Transitive reduction
    #g=transitive_reduction(g)
    for n1 in g.nodes_iter():
        if g.has_edge(n1, n1):
            g.remove_edge(n1, n1)
        for n2 in g.successors(n1):
            for n3 in g.successors(n2):
                nodes=list(nx.dfs_preorder_nodes(g, n3))
                for n4 in nodes:
                    if g.has_edge(n1, n4):
                        g.remove_edge(n1, n4)
    #Back to adjacency_matrix and return
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
        nx.DiGraph.add_node(self, n, attr_dict=attr_dict, **attr)
        self.node[n]['region']=None
        self.node[n]['mean_residue']=None

    def add_nodes_from(self, n, **attr):
        nx.DiGraph.add_nodes_from(self, n, **attr)
        for i in n:
            self.node[i]['region']=None
            self.node[i]['mean_residue'] = None

    def set_mean_intensity(self, n, value):
        self.node[n]['mean_residue']=value

    def get_mean_intensity(self, n):
        return self.node[n]['mean_residue']

    def set_region(self,n,r):
        if n not in self.nodes(): raise Exception(str(n)+" not a existing node (declared nodes are: "+str(self.nodes())+")")
        self.node[n]['region']=np.copy(r)

    def get_region(self,n):
        return self.node[n]['region']

    def update_intensities(self,image):
        from skgtimage.core.photometry import region_stat
        for n in self.nodes():
            region=self.get_region(n)
            intensity=region_stat(image,region)
            self.set_mean_intensity(n, intensity)


    def get_labelled(self,mapping=None,bg=0):
        labelled=np.zeros(self.get_image().shape,dtype=np.int)-bg
        fill_value=1
        for n in self.nodes():
            region=self.get_region(n)
            if mapping is not None:
                intensity = mapping[n]
                labelled=np.ma.masked_array(labelled, mask=region).filled(intensity)
            else:
                labelled=np.ma.masked_array(labelled,mask=region).filled(fill_value)
                fill_value+=1
        return labelled

