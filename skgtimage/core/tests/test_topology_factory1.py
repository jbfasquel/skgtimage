#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

# IMAGE
image=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 2, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])

# INITIAL RESIDUES
initial_residues=[np.where(image==i,1,0) for i in range(0,np.max(image)+1)]

# EXPECTED RESIDUES
res0=np.where(image==0,1,0)
res1=np.where(image==1,1,0)
res2=np.where(image==2,1,0)

###############################
#SITUATION WITH 'INEXACT GRAPH MATCHING'
#SEGMENTED REGION C HAS AN OUTLIER (i.e. value '3' surrounded by ones)
###############################
class TestTopologyFactory1(unittest.TestCase):
    ####################################
    #   EXECUTE TWICE THE FUNCTION: SHOULD NOT MODIFY RESIDUES AND NEW RESIDUES
    ####################################
    def test01(self):
        t_graph,residues=sgi.core.topological_graph_from_residues(initial_residues)
        self.assertEqual(len(residues),3)
        self.assertTrue(np.array_equal(res0,residues[0]))
        self.assertTrue(np.array_equal(res1,residues[1]))
        self.assertTrue(np.array_equal(res2,residues[2]))
        print(t_graph.nodes())
        print(t_graph.edges())

        self.assertEqual(set(t_graph.nodes()),set([0, 1, 2]))
        self.assertEqual(set(t_graph.edges()),set([(1, 0), (2, 1)]))






if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTopologyFactory1)
    unittest.TextTestRunner(verbosity=2).run(suite)
