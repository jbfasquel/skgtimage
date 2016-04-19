#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

image=np.array([[1,1,1,1,1,1,1],
                [1,2,2,2,1,1,1],
                [1,2,3,2,1,3,1],
                [1,2,3,2,1,3,1],
                [1,2,2,2,1,1,1],
                [1,1,1,1,1,1,1]])

regionA=np.ones(image.shape)
regionB=np.array([  [0,0,0,0,0,0,0],
                    [0,1,1,1,0,0,0],
                    [0,1,1,1,0,0,0],
                    [0,1,1,1,0,0,0],
                    [0,1,1,1,0,0,0],
                    [0,0,0,0,0,0,0]])

regionC=np.array([  [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0],
                    [0,0,1,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0]])

regionD=np.array([  [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0],
                    [0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0]])



class TestParametersCase2(unittest.TestCase):
    def setUp(self):
        self.tp_model=sgi.core.TPModel()
        self.tp_model.set_topology("C<B<A;D<A")
        self.tp_model.set_photometry(["A<B<C=D"])
        self.tp_model.set_image(image)

    #######################
    #REGION A SEGMENTED
    #######################
    def test01(self):
        self.tp_model.set_region('A',regionA)
        self.tp_model.set_targets(['B'])
        #roi
        self.assertTrue(np.array_equal(self.tp_model.roi(),regionA))
        #roi image
        roi_image=np.ma.array(image, mask=np.logical_not(regionA))
        self.assertTrue(np.array_equal(self.tp_model.roi_image(),roi_image))
        self.assertTrue(np.array_equal(self.tp_model.roi_image().mask,roi_image.mask))

        # PARAMETERS FROM KNOWLEDGE : NUMBER OF CLUSTERS
        nb=self.tp_model.number_of_clusters()
        self.assertEqual(nb,3)

        # PARAMETERS FROM KNOWLEDGE : SEEDING CONSTRAINTS
        constraints=self.tp_model.intervals_for_clusters()
        self.assertEqual(list(constraints.values()),[[[1, 3]], [[1, 3]], [[1, 3]]])
    #######################
    #REGION A+B SEGMENTED
    #######################
    def test02(self):
        self.tp_model.set_region('A',regionA)
        self.tp_model.set_region('B',regionB)
        self.tp_model.set_targets(['C'])
        #roi
        self.assertTrue(np.array_equal(self.tp_model.roi(),regionB))
        #roi image
        roi_image=np.ma.array(image, mask=np.logical_not(regionB))
        self.assertTrue(np.array_equal(self.tp_model.roi_image(),roi_image))
        self.assertTrue(np.array_equal(self.tp_model.roi_image().mask,roi_image.mask))

        # PARAMETERS FROM KNOWLEDGE : NUMBER OF CLUSTERS
        nb=self.tp_model.number_of_clusters()
        self.assertEqual(nb,2)

        # PARAMETERS FROM KNOWLEDGE : SEEDING CONSTRAINTS
        constraints=self.tp_model.intervals_for_clusters()
        self.assertEqual(list(constraints.values()),[[[2, 3]], [[2, 3]]])

    #######################
    #REGION A+B+D SEGMENTED
    #######################
    def test03(self):
        self.tp_model.set_region('A',regionA)
        self.tp_model.set_region('B',regionB)
        self.tp_model.set_region('D',regionD)
        self.tp_model.set_targets(['C'])
        #roi
        self.assertTrue(np.array_equal(self.tp_model.roi(),regionB))
        #roi image
        roi_image=np.ma.array(image, mask=np.logical_not(regionB))
        self.assertTrue(np.array_equal(self.tp_model.roi_image(),roi_image))
        self.assertTrue(np.array_equal(self.tp_model.roi_image().mask,roi_image.mask))

        # PARAMETERS FROM KNOWLEDGE : NUMBER OF CLUSTERS
        nb=self.tp_model.number_of_clusters()
        self.assertEqual(nb,2)

        # PARAMETERS FROM KNOWLEDGE : SEEDING CONSTRAINTS
        constraints=self.tp_model.intervals_for_clusters()
        condition1=(list(constraints.values())==[[[3.0, 3.0]], [[2.0, 3.0]]])
        condition2=(list(constraints.values())==[[[2.0, 3.0]], [[3.0, 3.0]]])
        self.assertTrue(condition1 or condition2)

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParametersCase2)
    unittest.TextTestRunner(verbosity=2).run(suite)
