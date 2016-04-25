#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

# IMAGE
image=np.array([ [1, 1, 1, 1, 1, 1],
                [1, 2, 2, 1, 1, 1],
                [1, 2, 2, 1, 3, 1],
                [1, 2, 2, 1, 3, 1],
                [1, 2, 2, 1, 1, 1],
                [1, 1, 1, 1, 1, 1]])

# KNOWLEDGE
t_graph=sgi.core.graph_factory("B,C<A")
p_graph=sgi.core.graph_factory("A<B<C")
#tp_model=sgi.core.TPModel(t_graph,[p_graph])

#TRUTH
regionA=np.where(image>0,1,0)
regionB=np.where(image==2,1,0)
regionC=np.where(image==3,1,0)

###############################
#AMBIGUITY ON TOPOLOGY BUT NOT IN PHOTOMETRY
#TOPOLOGY
#       B -> A
#       C -> A
#PHOTOMETRY
#       A -> B -> C
###############################
'''
class TestRecognitionUseCase1(unittest.TestCase):
    def setUp(self):
        self.residues=[ np.where(image==i,1,0) for i in [1,2,3] ]

    ####################################
    #   GRAPHS FROM RESIDUES
    ####################################
    def test01(self):
        #Topology
        g,_=sgi.core.topological_graph_from_residues(self.residues)
        self.assertEqual(set(g.nodes()),set([0, 1, 2]))
        self.assertEqual(set(g.edges()),set([(1, 0),(2, 0)]))
        #Photometry
        g=sgi.core.photometric_graph_from_residues(image,self.residues)
        self.assertEqual(set(g.nodes()),set([0, 1, 2]))
        self.assertEqual(set(g.edges()),set([(0, 1), (1, 2)]))

    ####################################
    #   ISOMORPHISMS
    ####################################
    def test02(self):
        tmp_t_graph,_=sgi.core.topological_graph_from_residues(self.residues)
        tmp_p_graph=sgi.core.photometric_graph_from_residues(image,self.residues)
        self.assertEqual(sgi.core.find_subgraph_isomorphims(tmp_t_graph,t_graph),[{0: 'A', 1: 'B', 2: 'C'}, {0: 'A', 1: 'C', 2: 'B'}])
        self.assertEqual(sgi.core.find_subgraph_isomorphims(tmp_p_graph,p_graph),[{0: 'A', 1: 'B', 2: 'C'}])

    ####################################
    #   IDENTIFICATION
    ####################################
    def test03(self):
        tp_model.set_image(image)
        #tp_model.set_region('A',regionA)
        #tp_model.set_targets(['B','C'])
        tp_model.identify_from_residues(self.residues)
        self.assertTrue(np.array_equal(tp_model.get_region('A'),regionA))
        self.assertTrue(np.array_equal(tp_model.get_region('B'),regionB))
        self.assertTrue(np.array_equal(tp_model.get_region('C'),regionC))

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRecognitionUseCase1)
    unittest.TextTestRunner(verbosity=2).run(suite)
'''