#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as skit
import numpy as np
import data

class TestAlgoPhoto2(unittest.TestCase):
    def setUp(self):
        self.g=skit.core.IrDiGraph(None,image=data.image)
        self.g.add_nodes_from(['A','B','C','D','E'])
        self.g.add_edge('B','A')
        self.g.add_edge('C','B')
        self.g.add_edge('D','C')
        self.g.add_edge('A','D')
        self.g.add_edge('E','C')

    #############################
    #        A   <-    B
    #        |         ^
    #        v         |
    #        D   ->    C <- E
    #############################
    def test01(self):
        self.assertEqual(set(skit.core.recursive_predecessors(self.g, 'A')),set(['C', 'B', 'E', 'D']))
        self.assertEqual(set(skit.core.recursive_successors(self.g, 'A')),set(['C', 'B', 'D']))
        self.assertEqual(set(skit.core.recursive_segmented_successors(self.g, 'B')),set())
        self.assertEqual(set(skit.core.recursive_segmented_predecessors(self.g, 'B')),set())

    #############################
    #        A   <-    B*
    #        |         ^
    #        v         |
    #        D   ->    C <- E
    #############################
    def test02(self):
        self.g.set_region('B',data.B)
        self.assertEqual(set(skit.core.recursive_segmented_successors(self.g, 'A')),set('B'))
        self.assertEqual(set(skit.core.recursive_segmented_predecessors(self.g, 'A')),set('B'))

    #############################
    #        A   <-    B*
    #        |         ^
    #        v         |
    #        D   ->    C* <- E*
    #############################
    def test03(self):
        self.g.set_region('B',data.B)
        self.g.set_region('C',data.C)
        self.g.set_region('E',data.E)
        self.assertEqual(set(skit.core.recursive_segmented_successors(self.g, 'A')),set(['B','C']))
        self.assertEqual(set(skit.core.recursive_segmented_predecessors(self.g, 'A')),set(['B','C','E']))

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAlgoPhoto2)
    unittest.TextTestRunner(verbosity=2).run(suite)
