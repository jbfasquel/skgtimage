#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as skgti
import numpy as np

# IMAGE
image=np.array([[1,1,1,1,1,1,1],
                [1,2,2,2,1,1,1],
                [1,2,3,2,1,3,1],
                [1,2,3,2,1,3,1],
                [1,2,2,2,1,1,1],
                [1,1,1,1,1,1,1]],np.float)
label=image
# KNOWLEDGE
t_desc="C<B<A;D<A"
p_desc="A<B<C=D"
#tp_model=sgi.core.TPModel(t_graph,[p_graph])

# REGIONS
A=np.where(image>=1,1,0)
B=skgti.core.fill_region(np.where(image==2,1,0))
C=np.logical_and(B,np.where(image>=3,1,0))
D=np.where(image==3,1,0)-C


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
        id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False,verbose=False)
        self.assertTrue(np.array_equal(id2r['A'],A))
        self.assertTrue(np.array_equal(id2r['B'],B))
        self.assertTrue(np.array_equal(id2r['C'],C))
        self.assertTrue(np.array_equal(id2r['D'],D))

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRecognitionUseCase2)
    unittest.TextTestRunner(verbosity=2).run(suite)
