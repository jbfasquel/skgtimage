#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

# IMAGE
image=np.array([[1,1,1,1,1,1,1],
                [1,2,2,2,1,1,1],
                [1,2,3,2,1,3,1],
                [1,2,3,2,1,1,1],
                [1,2,2,2,1,1,1],
                [1,1,1,1,1,1,1]],np.float)

# KNOWLEDGE
t_graph=sgi.core.graph_factory("C<B<A")
p_graph=sgi.core.graph_factory("A<B<C")
#tp_model=sgi.core.TPModel(t_graph,[p_graph])

# REGIONS
regionA=np.where(image>=1,1,0)
regionB=sgi.core.fill_region(np.where(image==2,1,0))
regionC=np.logical_and(regionB,np.where(image==3,1,0))

###############################
#SITUATION WITH 'INEXACT GRAPH MATCHING'
#SEGMENTED REGION C HAS AN OUTLIER (i.e. value '3' surrounded by ones)
###############################
'''
class TestRecognitionUseCase4(unittest.TestCase):
    def setUp(self):
        self.residues=[ np.where(image==i,1,0) for i in [1,2] ]
        self.residues+=[ np.where(image>=3,1,0) ]

    ####################################
    #   EXECUTE TWICE THE FUNCTION: SHOULD NOT MODIFY RESIDUES AND NEW RESIDUES
    ####################################
    def test01(self):
        built_t_graph,new_residues=sgi.core.topological_graph_from_residues(self.residues)
        self.assertEqual(len(self.residues),3)
        self.assertEqual(len(new_residues),4)
        built_t_graph,new_residues=sgi.core.topological_graph_from_residues(self.residues)
        self.assertEqual(len(self.residues),3)
        self.assertEqual(len(new_residues),4)

    ####################################
    #   ISOMORPHISMS AND SUBISOMORPHISMS
    ####################################
    def test02(self):
        #Topological graph
        built_t_graph,new_residues=sgi.core.topological_graph_from_residues(self.residues)
        #but sub-isomorphism
        subiso=sgi.core.find_subgraph_isomorphims(built_t_graph,t_graph)
        self.assertEqual(subiso,[{0: 'A', 1: 'B', 3: 'C'}])


    ####################################
    #   IDENTIFICATION
    ####################################
    def test03(self):
        tp_model.set_image(image)
        tp_model.set_region('A',regionA)
        tp_model.set_targets(['B','C'])
        tp_model.identify_from_residues(self.residues)
        self.assertTrue(np.array_equal(tp_model.get_region('A'),regionA))
        self.assertTrue(np.array_equal(tp_model.get_region('B'),regionB))
        self.assertTrue(np.array_equal(tp_model.get_region('C'),regionC))

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRecognitionUseCase4)
    unittest.TextTestRunner(verbosity=2).run(suite)
'''