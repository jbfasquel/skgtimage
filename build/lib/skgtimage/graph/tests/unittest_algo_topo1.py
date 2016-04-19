#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import unittest
import skgtimage as skit
import numpy as np
import data

#############################
#            A
#          /  \
#        B     E
#       / \
#      C   D
#############################
class TestAlgoTopo1(unittest.TestCase):
    """
    Test of core search algorithms and residue computation, assuming all regions are segmented
    """
    def setUp(self):
        self.g=skit.core.IrDiGraph(None,image=data.image)
        self.g.add_nodes_from(['A','B','C','D','E'])
        self.g.add_edge('B','A')
        self.g.add_edge('C','B')
        self.g.add_edge('D','B')
        self.g.add_edge('E','A')
        
    #############################
    #            A
    #          /  \
    #        B     E
    #       / \
    #      C   D
    #############################
    def test01(self):        
        self.assertEqual(set(skit.core.recursive_successors(self.g,'C')),set(['B','A']))
        self.assertEqual(set(skit.core.recursive_predecessors(self.g,'A')),set(['B', 'E', 'C', 'D']))

    #############################
    #            A*
    #          /  \
    #        B*     E*
    #       / \
    #      C*   D*
    #############################
    def test02(self):
        self.g.set_region('A',data.A);self.g.set_region('B',data.B)
        self.g.set_region('C',data.C);self.g.set_region('D',data.D)
        self.g.set_region('E',data.E)
        self.assertTrue(np.array_equal(skit.core.residue(self.g,'A'),data.A-(data.B+data.E)))
        self.assertTrue(np.array_equal(skit.core.residue(self.g,'B'),data.B-(data.C+data.D)))

    #############################
    #            A*
    #          /  \
    #        B     E*
    #       / \
    #      C   D
    #############################
    def test03(self):
        self.g.set_region('A',data.A)
        self.g.set_region('E',data.E)
        #Segmented successors/predecessors
        self.assertEqual(skit.core.segmented_successors(self.g,'B'),['A'])
        self.assertEqual(skit.core.segmented_successors(self.g,'C'),['A'])
        self.assertEqual(skit.core.segmented_successors(self.g,'D'),['A'])
        self.assertEqual(skit.core.segmented_successors(self.g,'E'),['A'])
        #ROI
        roi=skit.core.roi(self.g,'D')
        self.assertTrue(np.array_equal(roi,data.A-data.E))
        #Number of classes (potentially of similar photometry)
        self.assertEqual(skit.core.regionlist_in_roi(self.g,'B'),['A', 'C', 'B', 'D'])
    
    #############################
    #            A*
    #          /  \
    #        B     E*
    #       / \
    #      C   D*
    #############################
    def test04(self):            
        self.g.set_region('A',data.A)
        self.g.set_region('E',data.E)
        self.g.set_region('D',data.D)
        self.assertEqual(skit.core.regionlist_in_roi(self.g,'B'),['A', 'C', 'B'])
        
if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAlgoTopo1)
    unittest.TextTestRunner(verbosity=2).run(suite)
