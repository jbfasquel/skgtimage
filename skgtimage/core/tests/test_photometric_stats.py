#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

image_gray=np.array([ [ 0,0,0,0,0,0],
                      [ 0,1,2,0,0,0],
                      [ 0,1,2,0,0,0],
                      [ 0,1,2,0,4,5],
                      [ 0,0,0,0,6,7],
                      [ 0,0,0,0,0,0],
                    ])

region_B=np.where( image_gray>3,1,0)
region_A=np.where( image_gray>0,1,0)-region_B

regions=region_A+region_B

image_color=np.dstack((image_gray,image_gray+1,image_gray+2))

class TestPhotometricStats(unittest.TestCase):
    ##############
    # IN GRAY, FROM LIST OF REGIONS
    ##############
    def test01(self):
        #Mean
        self.assertEqual(sgi.core.region_stat(image_gray,region_A),1.5)
        self.assertEqual(sgi.core.region_stat(image_gray,region_B),5.5)
        ord_indices=sgi.core.sort_region_indices_by_stat(image_gray,[region_A,region_B],return_stats=False)
        self.assertTrue(np.array_equal(ord_indices[0],1))
        self.assertTrue(np.array_equal(ord_indices[1],0))
        #Dev
        self.assertEqual(sgi.core.region_stat(image_gray,region_A,fct=np.std),0.5)
        self.assertAlmostEqual(sgi.core.region_stat(image_gray,region_B,fct=np.std),1.1,places=1)
    ##############
    # IN COLOR, FROM LIST OF REGIONS
    ##############
    def test02(self):
        #Mean
        #self.assertEqual(sgi.core.region_stat(image_color,region_A,gray=False,component=0),[1.5])
        #self.assertEqual(sgi.core.region_stat(image_color,region_A,gray=False),[1.5])
        #a=sgi.core.region_stat(image_color,region_A,gray=False)
        self.assertTrue(np.array_equal(sgi.core.region_stat(image_color,region_A,mc=True),np.array([1.5,2.5,3.5])))
        a=sgi.core.region_stat(image_color,region_B,mc=True)[0]
        self.assertTrue(np.array_equal(sgi.core.region_stat(image_color,region_B,mc=True),np.array([5.5,6.5,7.5])))
        #self.assertEqual(sgi.core.region_stat(image_color,region_B,gray=False,component=0),5.5)

        #Std
        #a=sgi.core.region_stat(image_color,region_A,fct=np.std,gray=False)
        self.assertTrue(np.array_equal(sgi.core.region_stat(image_color,region_A,fct=np.std,mc=True),np.array([0.5,0.5,0.5])))
        #self.assertEqual(sgi.core.region_stat(image_color,region_A,fct=np.std,gray=False,component=0),0.5)
        self.assertAlmostEqual(sgi.core.region_stat(image_color,region_B,fct=np.std,mc=True)[0],1.1,places=1)


if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhotometricStats)
    unittest.TextTestRunner(verbosity=2).run(suite)
