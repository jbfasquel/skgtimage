#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

image=np.array([])

tp_model=sgi.core.TPModel()
tp_model.set_topology("B,C,D<A")
tp_model.set_image(image)
tp_model.set_region('A',image)

class TestParametersCase1(unittest.TestCase):
    def test01(self):
        tp_model.set_photometry(["A<B<C<D"])
        n=tp_model.number_of_clusters()
        self.assertEqual(tp_model.number_of_clusters(),4)
        tp_model.set_photometry(["A<B<C=D"])
        self.assertEqual(tp_model.number_of_clusters(),3)
        tp_model.set_photometry(["A=B<C=D"])
        self.assertEqual(tp_model.number_of_clusters(),2)
        tp_model.set_photometry(["A=B=C=D"])
        self.assertEqual(tp_model.number_of_clusters(),1)

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParametersCase1)
    unittest.TextTestRunner(verbosity=2).run(suite)
