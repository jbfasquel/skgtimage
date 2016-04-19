#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np
from skgtimage.core.tests.data import *

###############################
#      C -> B -> A <- F
#           ^
#           |
#  E-> D ----
###############################
class TestGraphROIResidueClasses(unittest.TestCase):
    def setUp(self):
        self.t_graph=skgti.core.IrDiGraph()
        self.t_graph.add_nodes_from(['A','B','C','D','E','F'])
        self.t_graph.add_edge('B','A');self.t_graph.add_edge('C','B');self.t_graph.add_edge('D','B')
        self.t_graph.add_edge('E','D')
        self.t_graph.add_edge('F','A')

    ####################################
    #   RESIDUE
    ####################################
    def test01(self):
        #Residue of a non segmented region -> None
        self.assertEqual(sgi.core.residue(self.t_graph,'A'),None)
        #Residue of a segmented region
        self.t_graph.set_region('A',A)
        self.assertTrue(np.array_equal(sgi.core.residue(self.t_graph,'A'),A))
        self.t_graph.set_region('F',F)
        self.assertTrue(np.array_equal(sgi.core.residue(self.t_graph,'A'),A-F))
        self.t_graph.set_region('E',E)
        self.assertTrue(np.array_equal(sgi.core.residue(self.t_graph,'A'),A-E-F))
        self.t_graph.set_region('B',B)
        self.assertTrue(np.array_equal(sgi.core.residue(self.t_graph,'B'),B-E))
    ####################################
    #   ROI
    ####################################
    def test02(self):
        #Roi without segmented region nor image set -> exception should be raised
        self.assertTrue(np.array_equal(sgi.core.roi_for_target(self.t_graph,'A',image.shape),np.ones(image.shape,dtype=image.dtype)))
        #Roi without segmented region but with a image set
        self.assertTrue(np.array_equal(sgi.core.roi_for_target(self.t_graph,'E',image.shape),np.ones(image.shape,dtype=image.dtype)))
        #Roi with segmented regions
        self.t_graph.set_region('A',A)
        self.assertTrue(np.array_equal(sgi.core.roi_for_target(self.t_graph,'E',image.shape),A))

        self.assertTrue(np.array_equal(sgi.core.roi_for_targets(self.t_graph,['B','C','E'],image.shape),A))

        self.t_graph.set_region('B',B)
        self.assertTrue(np.array_equal(sgi.core.roi_for_target(self.t_graph,'E',image.shape),B))
        self.assertTrue(np.array_equal(sgi.core.roi_for_target(self.t_graph,'F',image.shape),A-B))
        self.t_graph.set_region('E',E)
        self.assertTrue(np.array_equal(sgi.core.roi_for_target(self.t_graph,'D',image.shape),B-E))

    ####################################
    #   CLASSES
    ####################################
    def test03(self):
        self.t_graph.set_region('B',B)
        #Roi without segmented region nor image set -> exception should be raised
        #with self.assertRaises(Exception): sgi.core.classes_for_target(self.t_graph,'A')

        self.assertEqual(set(sgi.core.classes_for_target(self.t_graph,'D')),set(['B','C','D','E']))
        self.assertEqual(set(sgi.core.classes_for_targets(self.t_graph,['D','C'])),set(['B','C','D','E']))
        self.t_graph.set_region('E',E)
        self.assertEqual(set(sgi.core.classes_for_target(self.t_graph,'D')),set(['B','C','D']))
        self.assertEqual(set(sgi.core.classes_for_targets(self.t_graph,['D','C'])),set(['B','C','D']))


if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGraphROIResidueClasses)
    unittest.TextTestRunner(verbosity=2).run(suite)
