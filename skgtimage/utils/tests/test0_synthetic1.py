#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as skgti
import numpy as np

# IMAGE
image=np.array([[0.0, 0.2, 0.0, 0.0, 0.0],
                [0.2, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.1, 1.4, 0.9, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]])

label=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 4, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])

# KNOWLEDGE
t_desc="B<A"
p_desc="A<B"

#TRUTH
B=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])
A=np.ones((5,5))-B
###############################
#AMBIGUITY ON TOPOLOGY BUT NOT IN PHOTOMETRY
#TOPOLOGY
#       B -> A
#PHOTOMETRY
#       A -> B
###############################
class TestRecognitionUseCase1(unittest.TestCase):
    ####################################
    #   GRAPHS FROM RESIDUES
    ####################################
    def test01(self):
        id2r,matcher=skgti.utils.recognize(image,label,t_desc,p_desc)
        self.assertTrue(np.array_equal(id2r['A'],A))
        self.assertTrue(np.array_equal(id2r['B'],B))

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRecognitionUseCase1)
    unittest.TextTestRunner(verbosity=2).run(suite)
