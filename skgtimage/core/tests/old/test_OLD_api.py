#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import unittest
import skgtimage as skit
import numpy as np
from skgtimage.core.tests import data

class TestDigraph1(unittest.TestCase):
    #INITIALIZATION OF BASES AND ROLES
    def setUp(self):
        self.g=skit.core.IrDiGraph(None,image=data.image)
        
    #CHECK GRAPH ATTRIBUTE "IMAGE" 
    def test01(self):
        self.assertTrue(np.array_equal(self.g.graph['image'],data.image))
    #CHECK NODE ATTRIBUTE "IMAGE"
    def test02(self):
        self.g.add_node('A')
        self.assertEqual(self.g.get_region('A'), None)
        self.g.set_region('A',data.A)
        self.assertNotEqual(self.g.get_region('A'), None)
        self.assertTrue(np.array_equal(self.g.get_region('A'),data.A))
        self.g.add_node('B',mykey='myvalue')
        self.assertEqual(self.g.node['B']['mykey'],'myvalue')

        
if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDigraph1)
    unittest.TextTestRunner(verbosity=2).run(suite)
