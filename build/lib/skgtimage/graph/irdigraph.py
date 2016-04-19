#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause
"""
Truc
"""
import networkx as nx

class IrDiGraph(nx.DiGraph):
    """With automatically assigned attribute for core and nodes

    core: 'image' (default value is 'None')
    node: 'region' (default value is 'None')    
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
            >>> sgt.core.IrDiGraph()
            <skgtimage.core.irdigraph.IrDiGraph object at ...>
        """
        super(IrDiGraph,self).__init__(data,**attr)
        self.graph['image']=image
    def add_node(self, n, attr_dict=None, **attr):
        """Add node

        :parameters:
            * n: as as in networkx.DiGraph.add_node
            * attr_dict: as as in networkx.DiGraph.add_node

        """
        nx.DiGraph.add_node(self, n, attr_dict=attr_dict, **attr)
        self.node[n]['region']=None
    def add_nodes_from(self, n, **attr):
        nx.DiGraph.add_nodes_from(self, n, **attr)
        for i in n: self.node[i]['region']=None
    def add_region(self,n,r):
        """Add node

        :returns: None
        """
        self.node[n]['region']=r
    def region(self,n):
        """ mes commentaires """
        return self.node[n]['region']
    def image(self):
        return self.graph['image']
    
    
