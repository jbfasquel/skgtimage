#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import unittest
import skgtimage as skit
import numpy as np
import data

class TestAlgoPhoto3(unittest.TestCase):
    def setUp(self):
        self.g=skit.core.IrDiGraph(None,image=data.image)
        self.g.add_nodes_from(['A','B','C','D','E'])
        self.g.add_edge('A','B')
        self.g.add_edge('B','C');self.g.add_edge('C','B')
        self.g.add_edge('C','D');self.g.add_edge('D','C')
        self.g.add_edge('D','E')

        #############################
        #                    E
        #                    ^
        #                    |
        #        B <-> C <-> D
        #        ^
        #        |
        #        A
        #############################
    def test01(self):
        #self.assertEqual(set(skit.core.recursive_predecessors(self.g,'C')),set(['A', 'B', 'D']))
        self.assertEqual(set(skit.core.recursive_predecessors(self.g,'C')),set(['A']))
        #self.assertEqual(set(skit.core.recursive_successors(self.g,'C')),set(['B', 'E', 'D']))
        self.assertEqual(set(skit.core.recursive_successors(self.g,'C')),set(['E']))
        self.assertEqual(set(skit.core.recursive_brothers(self.g,'C')),set(['B','D']))

        #############################
        #                    E
        #                    ^
        #                    |
        #        B* <-> C <-> D
        #        ^
        #        |
        #        A
        #############################
    def test02(self):
        self.g.set_region('B',data.B)
        self.assertEqual(set(skit.core.recursive_segmented_brothers(self.g,'C')),set(['B']))


if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAlgoPhoto3)
    unittest.TextTestRunner(verbosity=2).run(suite)
