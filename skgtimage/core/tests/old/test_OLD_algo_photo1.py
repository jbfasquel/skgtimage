#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as skit
import numpy as np
from skgtimage.core.tests import data

class TestAlgoPhoto1(unittest.TestCase):
    def setUp(self):
        self.g=skit.core.IrDiGraph(None,image=data.image)
        self.g.add_nodes_from(['A','B','C','D','E'])
        self.g.add_edge('B','A')
        self.g.add_edge('C','B')
        self.g.add_edge('D','B')
        self.g.add_edge('E','A')
        self.g.add_edge('B','E')
        self.g.add_edge('E','B')
        self.g.add_edge('D','C')

    #############################
    #             A*
    #        ^         ^
    #        |         |
    #        B   <->   E*
    #        ^         ^
    #        |         |
    #        C   <-    D
    #Search for segmented successors of B -> A and not [A,B]
    #############################
    def test01(self):
        self.g.set_region('A',data.A)
        self.g.set_region('E',data.E)
        self.assertEqual(set(skit.core.segmented_successors(self.g,'B')),set('A'))

    #############################
    #             A*
    #        ^         ^
    #        |         |
    #        B*   <->   E
    #        ^         ^
    #        |         |
    #        C*   <-    D
    #Search for segmented predecessors of E -> C and not B
    #############################
    def test02(self):
        self.g.set_region('B',data.B)
        self.g.set_region('C',data.C)
        self.assertEqual(set(skit.core.segmented_predecessors(self.g,'E')),set('C'))
        
        
if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAlgoPhoto1)
    unittest.TextTestRunner(verbosity=2).run(suite)
