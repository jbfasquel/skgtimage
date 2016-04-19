#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

class TestGraphSearchFiltered(unittest.TestCase):
    def setUp(self):
        self.g=sgi.core.IrDiGraph()
        self.g.add_nodes_from(['A','B','C','D','E','F','G'])

    ####################################
    #   A ->  B  ->  C
    ####################################
    def test01(self):
        self.g.add_edge('A','B')
        self.g.add_edge('B','C')
        self.assertEqual(set(sgi.core.recursive_segmented_successors(self.g,'A')),set())
        self.assertEqual(set(sgi.core.first_segmented_successors(self.g,'A')),set())

    ####################################
    #   A ->  B  ->  C*
    ####################################
    def test02(self):
        self.g.add_edge('A','B')
        self.g.add_edge('B','C')
        self.g.set_region('C',np.array([])) #already segmented region
        self.assertEqual(set(sgi.core.recursive_segmented_successors(self.g,'A')),set(['C']))

        self.assertEqual(set(sgi.core.first_segmented_successors(self.g,'A')),set(['C']))

        self.assertEqual(set(sgi.core.recursive_successors_until_first_segmented(self.g,'A')),set(['B']))
        self.assertEqual(set(sgi.core.recursive_predecessors_until_first_segmented(self.g,'C')),set(['A','B']))

    ####################################
    #   A ->  B*  ->  C*
    ####################################
    def test03(self):
        self.g.add_edge('A','B')
        self.g.add_edge('B','C')
        self.g.set_region('B',np.array([])) #already segmented region
        self.g.set_region('C',np.array([])) #already segmented region
        self.assertEqual(set(sgi.core.recursive_segmented_successors(self.g,'A')),set(['B','C']))
        self.assertEqual(set(sgi.core.first_segmented_successors(self.g,'A')),set(['B']))

    ####################################
    #   A ->  B*  ->  C
    ####################################
    def test04(self):
        self.g.add_edge('A','B')
        self.g.add_edge('B','C')
        self.g.set_region('B',np.array([])) #already segmented region
        self.assertEqual(set(sgi.core.recursive_segmented_successors(self.g,'A')),set(['B']))

    ####################################
    #   A* ->  B  ->  C
    ####################################
    def test05(self):
        self.g.add_edge('A','B')
        self.g.add_edge('B','C')
        self.g.set_region('A',np.array([])) #already segmented region
        self.assertEqual(set(sgi.core.recursive_segmented_predecessors(self.g,'C')),set(['A']))

    ####################################
    #   A* ->  B*  ->  C
    #                  ^
    #                  |
    #   D* ->  E  -----
    ####################################
    def test06(self):
        self.g.add_edge('A','B')
        self.g.add_edge('B','C')
        self.g.add_edge('D','E')
        self.g.add_edge('E','C')
        self.g.set_region('A',np.array([])) #already segmented region
        self.g.set_region('B',np.array([])) #already segmented region
        self.g.set_region('D',np.array([])) #already segmented region

        self.assertEqual(set(sgi.core.recursive_segmented_predecessors(self.g,'C')),set(['A', 'B','D']))
        self.assertEqual(set(sgi.core.first_segmented_predecessors(self.g,'C')),set(['B','D']))
        self.assertEqual(set(sgi.core.first_segmented_predecessors(self.g,'B')),set(['A']))

        self.assertEqual(set(sgi.core.recursive_successors_until_first_segmented(self.g,'A')),set())
        self.assertEqual(set(sgi.core.recursive_predecessors_until_first_segmented(self.g,'C')),set(['E']))

    ####################################
    #   A* ->  B*  ->  C <-> D* <-> E -> F -> G*
    ####################################
    def test07(self):
        self.g.add_edge('A','B')
        self.g.add_edge('B','C')
        self.g.add_edge('C','D');self.g.add_edge('D','C')
        self.g.add_edge('D','E');self.g.add_edge('E','D')
        self.g.add_edge('E','F')
        self.g.add_edge('F','G')

        self.g.set_region('A',np.array([])) #already segmented region
        self.g.set_region('B',np.array([])) #already segmented region
        self.g.set_region('D',np.array([])) #already segmented region
        self.g.set_region('G',np.array([])) #already segmented region

        self.assertEqual(set(sgi.core.first_segmented_predecessors(self.g,'F')),set(['D']))
        self.assertEqual(set(sgi.core.first_segmented_predecessors(self.g,'E')),set(['B']))
        self.assertEqual(set(sgi.core.first_segmented_successors(self.g,'E')),set(['G']))

'''
    ####################################
    #   A ->  B  ->  C
    #                ^
    #                |
    #   D ->  E  ----
    ####################################
    def test02(self):
        self.g.add_edge('A','B')
        self.g.add_edge('B','C')
        self.g.add_edge('D','E')
        self.g.add_edge('E','C')
        self.assertEqual(set(sgi.core.recursive_successors(self.g,'A')),set(['B', 'C']))
        self.assertEqual(set(sgi.core.recursive_predecessors(self.g,'C')),set(['A', 'B','D','E']))

    ####################################
    #   A ->  B  ->  C
    #   ^            |
    #   |            |
    #    ------------
    ####################################
    def test03(self):
        self.g.add_edge('A','B')
        self.g.add_edge('B','C')
        self.g.add_edge('C','A')
        self.assertEqual(set(sgi.core.recursive_successors(self.g,'A')),set(['B', 'C']))
        self.assertEqual(set(sgi.core.recursive_predecessors(self.g,'C')),set(['A', 'B']))

    ####################################
    #   A ->  B  <->  C <-> D -> E
    ####################################
    def test04(self):
        self.g.add_edge('A','B')
        self.g.add_edge('B','C');self.g.add_edge('C','B')
        self.g.add_edge('C','D');self.g.add_edge('D','C')
        self.g.add_edge('D','E')

        self.assertEqual(set(sgi.core.recursive_successors(self.g,'A')),set(['B', 'C','D','E']))
        self.assertEqual(set(sgi.core.recursive_brothers(self.g,'B')),set(['C', 'D']))
        self.assertEqual(set(sgi.core.recursive_brothers(self.g,'C')),set(['B', 'D']))
        self.assertEqual(set(sgi.core.recursive_brothers(self.g,'D')),set(['B', 'C']))


    ####################################
    #   A ->  B  <->  C <-> D -> E -> G
    #   ^                        |
    #   |                        |
    #    --------- F <-----------
    ####################################
    def test05(self):
        self.g.add_edge('A','B')
        self.g.add_edge('B','C');self.g.add_edge('C','B')
        self.g.add_edge('C','D');self.g.add_edge('D','C')
        self.g.add_edge('D','E')
        self.g.add_edge('E','F')
        self.g.add_edge('F','A')
        self.g.add_edge('E','G')

        #Standard successors/predecessors without taking care about "brothers"
        self.assertEqual(set(sgi.core.recursive_successors(self.g,'E')),set(['A', 'D', 'C', 'F', 'B','G']))
        self.assertEqual(set(sgi.core.recursive_predecessors(self.g,'E')),set(['A', 'D', 'C', 'F', 'B']))

        #Brothers
        self.assertEqual(set(sgi.core.recursive_brothers(self.g,'E')),set())
        self.assertEqual(set(sgi.core.recursive_brothers(self.g,'B')),set(['C','D']))


        #Standard successors/predecessors without "brothers"
        self.assertEqual(set(sgi.core.recursive_nonbrother_successors(self.g,'E')),set(['A', 'D', 'C', 'F', 'B','G']))
        self.assertEqual(set(sgi.core.recursive_nonbrother_predecessors(self.g,'E')),set(['A', 'D', 'C', 'F', 'B']))
        self.assertEqual(set(sgi.core.recursive_nonbrother_successors(self.g,'B')),set(['A', 'F', 'E','G']))
        self.assertEqual(set(sgi.core.recursive_nonbrother_predecessors(self.g,'B')),set(['A', 'F', 'E']))
'''

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGraphSearchFiltered)
    unittest.TextTestRunner(verbosity=2).run(suite)
