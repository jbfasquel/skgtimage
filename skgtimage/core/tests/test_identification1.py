#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as skgti
import numpy as np

#IMAGE
image=np.array([[0.0, 0.2, 0.0, 0.0, 0.0],
                [0.2, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.1, 1.4, 0.9, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]])

#Segmentation (labelled image)
label=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 4, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])

#A priori knowledge
t_desc="B<A"
p_desc="A<B"
#Expected result
rA=np.ones(image.shape)
rB=np.where(label>=1,1,0)
expected_id2r={'A':rA,'B':rB}

###############################
#AMBIGUITY ON TOPOLOGY BUT NOT IN PHOTOMETRY
#TOPOLOGY
#       B -> A
#       C -> A
#PHOTOMETRY
#       A -> B -> C
###############################

class TestIdentificationCase1(unittest.TestCase):
    def setUp(self):
        pass
    ####################################
    #   GRAPHS FROM RESIDUES
    ####################################
    def check(self,id2r):
        for i in id2r:
            expected_region=expected_id2r[i]
            obtained_region=id2r[i]
            self.assertTrue(np.array_equal(expected_region,obtained_region))

    def test01(self):
        id2r,m=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,verbose=False)
        self.check(id2r)

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIdentificationCase1)
    unittest.TextTestRunner(verbosity=2).run(suite)
