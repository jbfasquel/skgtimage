#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as skgti
import numpy as np

# IMAGE
image=np.array([[0.0, 0.2, 0.0, 0.0, 0.0],
                [0.2, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.1, 1.4, 0.9, 0.0],
                [0.0, 1.0, 1.3, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]])

label=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 4, 1, 0],
                [0, 1, 5, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 6, 0, 0],
                [0, 0, 0, 0, 0]])

###############################
#SITUATION WITH 'INEXACT GRAPH MATCHING'
#SEGMENTED REGION C HAS AN OUTLIER (i.e. value '3' surrounded by ones)
###############################
class TestTopology(unittest.TestCase):
    def setUp(self):
        self.built_t_graph,self.built_p_graph=skgti.core.from_labelled_image(image,label)
    def check(self,nodes,edges):
        self.assertEqual(len(set(nodes)-set(self.built_t_graph.nodes())),0)
        self.assertEqual(len( set(edges)-set(self.built_t_graph.edges()) ),0)
    ####################################
    #   EXECUTE TWICE THE FUNCTION: SHOULD NOT MODIFY RESIDUES AND NEW RESIDUES
    ####################################
    def test01(self):
        skgti.core.merge_nodes_topology(self.built_t_graph,0,1)
        self.check([1,2,3,4],[(2,1),(3,1),(4,1)])
    def test02(self):
        skgti.core.merge_nodes_topology(self.built_t_graph,1,0)
        self.check([0,2,3,4],[(2,0),(3,0),(4,0)])
    def test03(self):
        skgti.core.merge_nodes_topology(self.built_t_graph,2,3)
        self.check([0,1,3,4],[(1,0),(3,1),(4,0)])
    def test04(self):
        skgti.core.merge_nodes_topology(self.built_t_graph,1,2)
        self.check([0,2,3,4],[(2,0),(3,2),(4,0)])
    def test05(self):
        skgti.core.merge_nodes_topology(self.built_t_graph,1,4)
        self.check([0,2,3,4],[(2,4),(3,4),(4,0)])
    def test06(self):
        skgti.core.merge_nodes_topology(self.built_t_graph,4,1)
        self.check([0,1,2,3],[(2,1),(3,1),(1,0)])





if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTopology)
    unittest.TextTestRunner(verbosity=2).run(suite)
