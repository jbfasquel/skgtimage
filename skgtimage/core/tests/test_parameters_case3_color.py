#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

# IMAGE
c0=np.array([0,0,0]) #black
c1=np.array([1,1,1]) #white
c2=np.array([1,1,0]) #yellow 0.9 to avoid many digits in HSV
c3=np.array([0,1,1]) #green-blue
c4=np.array([1,0,0]) #red
c5=np.array([0,0,0]) #black
image=np.array([ [c0, c0, c0, c0, c0, c0, c0],
                 [c0, c1, c1, c1, c0, c3, c0],
                 [c0, c1, c2, c1, c0, c3, c0],
                 [c0, c1, c2, c1, c0, c0, c0],
                 [c0, c1, c5, c1, c0, c4, c0],
                 [c0, c1, c1, c1, c0, c4, c0],
                 [c0, c0, c0, c0, c0, c0, c0]])
regionC1=np.array([ [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]])

regionC3=np.array([ [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]])

regionC4=np.array([ [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0]])




class TestParametersCase3(unittest.TestCase):
    def setUp(self):
        self.image_hsv=sgi.utils.rgb2hsv(image)
        self.tp_model=sgi.core.TPModel()
        self.tp_model.set_topology("c2,c5<c1<c0;c3,c4<c0")
        self.tp_model.set_photometry(["c4<c2<c3;c1;c0=c5","c0=c5;c1<c2=c3=c4","c0,c5<c1=c2=c3=c4;c0=c5"]) #H,S,V, avoiding cycles
        self.tp_model.set_image(self.image_hsv)

    #######################
    #CONTEXT 1
    #######################
    def test01(self):
        self.tp_model.set_region('c0',np.ones(self.image_hsv.shape[0:2]))
        self.tp_model.set_region('c1',regionC1)
        self.tp_model.set_targets(['c2'])
        # PARAMETERS
        l_image=self.tp_model.roi_image() #;print(l_image[:,:,0]);print(l_image[:,:,1]);print(l_image[:,:,2])
        self.assertTrue(np.array_equal(np.logical_not(l_image.mask[:,:,0]),regionC1))
        nb=self.tp_model.number_of_clusters() #;print(nb)
        self.assertEqual(nb,3)
        intervals=self.tp_model.intervals_for_clusters()
        self.assertTrue(np.allclose(np.array(intervals['c2']),np.array([[0.0, 0.16666667], [0.0, 1.0], [0.0, 1.0]]),atol=0.001))
        self.assertTrue(np.allclose(np.array(intervals['c1']),np.array([[0.0, 0.16666667], [0.0, 1.0], [0.0, 1.0]]),atol=0.001))
        self.assertTrue(np.allclose(np.array(intervals['c5']),np.array([[0.0, 0.16666667], [0.0, 1.0], [0.0, 1.0]]),atol=0.001))
    #######################
    #CONTEXT 2
    #######################
    def test02(self):
        self.tp_model.set_region('c0',np.ones(self.image_hsv.shape[0:2]))
        self.tp_model.set_region('c1',regionC1)
        self.tp_model.set_region('c3',regionC3)
        self.tp_model.set_region('c4',regionC4)
        self.tp_model.set_targets(['c2'])
        # PARAMETERS
        l_image=self.tp_model.roi_image()
        nb=self.tp_model.number_of_clusters()
        intervals=self.tp_model.intervals_for_clusters()

        self.assertTrue(np.allclose(np.array(intervals['c2']),np.array([[0.0, 0.16666667], [1.0, 1.0], [1.0, 1.0]]),atol=0.001))
        self.assertTrue(np.allclose(np.array(intervals['c1']),np.array([[0.0, 0.16666667], [0.0, 1.0], [1.0, 1.0]]),atol=0.001))
        self.assertTrue(np.allclose(np.array(intervals['c5']),np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),atol=0.001))


if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParametersCase3)
    unittest.TextTestRunner(verbosity=2).run(suite)
