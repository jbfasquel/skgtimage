#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

# IMAGE
image=np.array([[0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 2, 2, 2, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 2, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0]])

# INITIAL RESIDUES
initial_residues=[np.where(image==i,1,0) for i in range(0,np.max(image)+1)]

# EXPECTED RESIDUES
res0=np.array([ [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1]])
res1=np.where(image==1,1,0)
res2=np.array([ [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]])

res3=np.array([ [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]])

res4=np.array([ [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]])
###############################
#SITUATION WITH 'INEXACT GRAPH MATCHING'
#SEGMENTED REGION C HAS AN OUTLIER (i.e. value '3' surrounded by ones)
###############################
class TestTopologyFactory(unittest.TestCase):
    ####################################
    #   EXECUTE TWICE THE FUNCTION: SHOULD NOT MODIFY RESIDUES AND NEW RESIDUES
    ####################################
    def test01(self):
        t_graph,residues=sgi.core.topological_graph_from_residues_refactorying(initial_residues)
        self.assertEqual(len(residues),5)
        '''
        for n in residues:
            print(n.astype(np.uint8))
        '''
        self.assertTrue(np.array_equal(res0,residues[0]))
        self.assertTrue(np.array_equal(res1,residues[1]))
        self.assertTrue(np.array_equal(res2,residues[2]))
        self.assertTrue(np.array_equal(res3,residues[3]))
        self.assertTrue(np.array_equal(res4,residues[4]))
        '''
        print(t_graph.nodes())
        print(t_graph.edges())
        '''
        self.assertEqual(set(t_graph.nodes()),set([0, 1, 2, 3, 4]))
        self.assertEqual(set(t_graph.edges()),set([(1, 0), (2, 1), (3, 1), (4, 3)]))






if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTopologyFactory)
    unittest.TextTestRunner(verbosity=2).run(suite)
