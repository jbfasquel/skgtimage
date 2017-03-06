#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as skgti
import numpy as np

# IMAGE
image=np.array([[1,1,1,1,1,1,1],
                [1,2,2,2,1,1,1],
                [1,2,3,2,1,3.1,1],
                [1,2,3,2,1,3.1,1],
                [1,2,2,2,1,1,1],
                [1,1,1,1,1,1,1]],np.float)
label=np.array([[1,1,1,1,1,1,1],
                [1,2,2,2,1,1,1],
                [1,2,3,2,1,4,1],
                [1,2,3,2,1,4,1],
                [1,2,2,2,1,1,1],
                [1,1,1,1,1,1,1]],np.float)
# KNOWLEDGE
t_desc="C<B<A;D<A"
p_desc="A<B<C=D"

# REGIONS
A=np.where(label==1,1,0)
B=np.where(label==2,1,0)
C=np.where(label==3,1,0)
D=np.where(label==4,1,0)



###############################
#AMBIGUITY ON PHOTOMETRY BUT NOT IN TOPOLOGY
#+RESIDUES WITH C AND D MERGED
#TOPOLOGY
#       C -> B -> A
#            D -> A
#PHOTOMETRY
#       A -> B -> C <-> D
###############################

class TestRecognitionUseCase2(unittest.TestCase):

    def test01(self):
        id2r,matcher=skgti.utils.recognize(image,label,t_desc,p_desc)
        self.assertTrue(np.array_equal(id2r['A'],A))
        self.assertTrue(np.array_equal(id2r['B'],B))
        self.assertTrue(np.array_equal(id2r['C'],C))
        self.assertTrue(np.array_equal(id2r['D'],D))

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRecognitionUseCase2)
    unittest.TextTestRunner(verbosity=2).run(suite)
