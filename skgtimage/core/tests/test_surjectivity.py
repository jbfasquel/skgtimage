#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np


nodes1=[0, 1, 2, 3, 4]
nodes2=[0, 1, 2, 3, 4,5]
common_iso1=[{0: 'liver', 1: 'tumor', 3: 'vessel'}, {0: 'liver', 2: 'tumor', 3: 'vessel'}, {0: 'liver', 3: 'vessel', 4: 'tumor'}]
common_iso2=[{0: 'liver', 1: 'tumor', 3: 'vessel'}, {1: 'liver', 2: 'tumor', 3: 'vessel'}, {0: 'liver', 3: 'vessel', 4: 'tumor'}] #invalid set


class TestSurjectivity(unittest.TestCase):
    ####################################
    #   UNMATCHED
    ####################################
    def test01(self):
        result=sgi.core.unmatched_nodes(common_iso1,nodes1)
        self.assertEqual(result,set())
        result=sgi.core.unmatched_nodes(common_iso1,nodes2)
        self.assertEqual(result,set([5]))
    ####################################
    #   SURJECTION
    ####################################
    def test02(self):
        pass
        '''
        self.assertNotEquals(sgi.core.find_sub_surjection(common_iso1),None)
        self.assertEquals(sgi.core.find_sub_surjection(common_iso1),{'vessel': {3}, 'liver': {0}, 'tumor': {1, 2, 4}})
        self.assertEquals(sgi.core.find_sub_surjection(common_iso2),None)
        '''

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSurjectivity)
    unittest.TextTestRunner(verbosity=2).run(suite)
