#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi

class TestParsing(unittest.TestCase):
    def setUp(self):
        pass
    ####################################
    #   A ->  B  ->  C
    ####################################
    def test01(self):
        g=sgi.core.graph_factory(" A,B < C")
        self.assertEqual(set(g.nodes()),set(['A','B','C']))
        self.assertEqual(set(g.edges()),set([('B','C'), ('A','C')]))
        g=sgi.core.graph_factory(" A < C > B")
        self.assertEqual(set(g.nodes()),set(['A','B','C']))
        self.assertEqual(set(g.edges()),set([('B','C'), ('A','C')]))
        g=sgi.core.graph_factory(" A < C; C > B")
        self.assertEqual(set(g.nodes()),set(['A','B','C']))
        self.assertEqual(set(g.edges()),set([('B','C'), ('A','C')]))
        g=sgi.core.graph_factory(" A < C; B < C")
        self.assertEqual(set(g.nodes()),set(['A','B','C']))
        self.assertEqual(set(g.edges()),set([('B','C'), ('A','C')]))

    def test02(self):
        g=sgi.core.graph_factory(" node1,node2 < node3")
        self.assertEqual(set(g.nodes()),set(['node1', 'node3', 'node2']))
        self.assertEqual(set(g.edges()),set([('node1', 'node3'), ('node2', 'node3')]))

    def test03(self):
        g=sgi.core.graph_factory(" A < B == C < D")
        self.assertEqual(set(g.nodes()),set(['C', 'B', 'D', 'A']))
        self.assertEqual(set(g.edges()),set([('C', 'B'), ('C', 'D'), ('B', 'C'), ('A', 'B')]))
        g=sgi.core.graph_factory(" A < B ; B == C; C < D")
        self.assertEqual(set(g.nodes()),set(['C', 'B', 'D', 'A']))
        self.assertEqual(set(g.edges()),set([('C', 'B'), ('C', 'D'), ('B', 'C'), ('A', 'B')]))




if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParsing)
    unittest.TextTestRunner(verbosity=2).run(suite)
