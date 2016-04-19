#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause
"""
Truc
"""
import networkx as nx




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
    '''
    #Use this for acyclic graph ??? -> Problem for photometric graph
    closed_g=transitive_closure(g) #reduction is done on the transitive closure of the graph
    nodes = closed_g.nodes()
    #edges = g.edges()
    for x in nodes:
       for y in nodes:
          for z in nodes:
              if (x, y) != (y, z) and (x, y) != (x, z):
                  if ((x, y) in closed_g.edges()) and ((y, z) in closed_g.edges()) and ((x, z) in closed_g.edges()): closed_g.remove_edge(x,z)
    return closed_g
    '''

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

    def get_residue(self,n):
        import numpy as np
        residue=np.copy(self.get_region(n))
        for e in self.predecessors(n):
            residue=np.logical_and(residue,np.logical_not(self.get_region(e)))
        return residue

    def update_intensities(self,image):
        from skgtimage.core.photometry import region_stat
        for n in self.nodes():
            intensity=region_stat(image,self.get_residue(n))
            self.set_mean_residue_intensity(n,intensity)

    def set_mean_residue_intensity(self,n,value):
        self.node[n]['mean_residue']=value

    def get_mean_residue_intensity(self,n):
        return self.node[n]['mean_residue']

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

    def __str__(self):
        chaine=""
        for n in self.nodes():
            chaine+="Node:" + str(n) + " ("+str(self.get_mean_residue_intensity(n))+")"
        return chaine