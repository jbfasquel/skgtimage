#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import networkx as nx


class IGraph(nx.DiGraph):
    """With automatically assigned attribute 'region' to each node (numpy array, 'None' by default)

    """
    def __init__(self,data=None,image=None,**attr):
        """
        documentation

        :param data: as in networkx.DiGraph.__init__
        :param numpy.array image: image to be analyzed
        :param attr: as in networkx.DiGraph.__init__
        :return: nothing

        """
        super(IGraph,self).__init__(data,**attr)
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

    def get_mean_residue_intensity(self,n):
        return 0.0

    def segmented_nodes(self):
        s=set()
        for n in self.nodes():
            if self.get_region(n) is not None: s = s | set([n])
        return s


    def __str__(self):
        chaine=""
        for n in self.nodes():
            chaine+="Node:" + str(n) + "\n( "+str(self.get_region(n))+" )\n"
        return chaine

class IPGraph:
    """Facade to both inclusion and photometric graphs

    """
    def __init__(self,image=None,igraph=None,pgraph=None):
        """
        documentation

        :param data: as in networkx.DiGraph.__init__
        :param numpy.array image: image to be analyzed
        :param attr: as in networkx.DiGraph.__init__
        :return: nothing

        """
        self.__image=image
        self.i_graph=None
        self.p_graph=None

'''
#test
import numpy as np
image=np.ones((5,5))
r0=np.copy(image)
r1=np.copy(image)
r2=np.copy(image)
ig=IGraph(image)
ig.add_nodes_from([0,1,2])
ig.add_node(0)
#print(ig.nodes())
print(ig)
print(ig.get_region(0))
'''