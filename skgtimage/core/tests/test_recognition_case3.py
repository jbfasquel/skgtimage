#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

# IMAGE
image=np.array([[1,1,1,1,1,1,1],
                [1,2,2,2,1,1,1],
                [1,2,3,2,1,3,1],
                [1,2,3,2,1,4,1],
                [1,2,2,2,1,1,1],
                [1,1,1,1,1,1,1]],np.float)

# KNOWLEDGE
t_graph=sgi.core.graph_factory("C<B<A;D<A")
p_graph=sgi.core.graph_factory("A<B<C=D")
tp_model=sgi.core.TPModel(t_graph,[p_graph])

# REGIONS
regionA=np.where(image>=1,1,0)
regionB=sgi.core.fill_region(np.where(image==2,1,0))
regionC=np.logical_and(regionB,np.where(image==3,1,0))
regionD=np.where(image>=3,1,0)-regionC


###############################
#AMBIGUITY ON PHOTOMETRY BUT NOT IN TOPOLOGY
#+RESIDUES WITH C AND D MERGED
#TOPOLOGY
#       C -> B -> A
#            D -> A
#PHOTOMETRY : INCLUDED SLIGHT PHOTOMETRIC DIFFERENCE BETWEEN C AND D
#       A -> B -> C <-> D
###############################
class TestRecognitionUseCase3(unittest.TestCase):
    def setUp(self):
        self.residues=[ np.where(image==i,1,0) for i in [1,2] ]
        self.residues+=[ np.where(image>=3,1,0) ]

    ####################################
    #   EXECUTE TWICE THE FUNCTION: SHOULD NOT MODIFY RESIDUES AND NEW RESIDUES
    ####################################
    def test01(self):
        built_t_graph,new_residues=sgi.core.topological_graph_from_residues(self.residues)
        self.assertEqual(len(self.residues),3)
        self.assertEqual(len(new_residues),4)
        built_t_graph,new_residues=sgi.core.topological_graph_from_residues(self.residues)
        self.assertEqual(len(self.residues),3)
        self.assertEqual(len(new_residues),4)


    ####################################
    #   TOPOLOGICAL GRAPH FROM RESIDUES
    ####################################
    def test02(self):
        #Topological graph
        built_t_graph,new_residues=sgi.core.topological_graph_from_residues(self.residues)
        self.assertEqual(len(self.residues),3)
        self.assertEqual(len(new_residues),4)
        self.assertEqual(set(built_t_graph.nodes()),set([0, 1, 2, 3]))
        self.assertEqual(set(built_t_graph.edges()),set([(1, 0), (2, 0), (3, 1)]))
        #Check regions
        regions=sgi.core.regions_from_residues(built_t_graph,new_residues)
        for i in range(0,len(regions)): built_t_graph.set_region(i,regions[i])
        #skgti.io.save_graph('inter',built_t_graph,directory='tmp',save_regions=True)
        self.assertEqual(np.array_equal(built_t_graph.get_region(0),regionA),True)
        self.assertEqual(np.array_equal(built_t_graph.get_region(1),regionB),True)
        self.assertEqual(np.array_equal(built_t_graph.get_region(3),regionC),True)
        self.assertEqual(np.array_equal(built_t_graph.get_region(2),regionD),True)


    ####################################
    #   PHOTOMETRIC GRAPH FROM RESIDUES
    ####################################
    def test03(self):
        #New residues: required for photometric graph
        built_t_graph,new_residues=sgi.core.topological_graph_from_residues(self.residues)
        #Photometric graph
        n=sgi.core.number_of_brother_links(p_graph)
        self.assertEqual(n,1)
        built_p_graph=sgi.core.photometric_graph_from_residues(image,new_residues)
        sgi.core.build_similarities(image,new_residues,built_p_graph,n)

        self.assertEqual(set(built_p_graph.nodes()),set([0, 1, 2, 3]))
        self.assertEqual(set(built_p_graph.edges()),set([(0, 1), (1, 3), (2, 3), (3, 2)]))

    ####################################
    #   ISOMORPHISMS
    ####################################
    def test04(self):
        #Topological graph
        built_t_graph,new_residues=sgi.core.topological_graph_from_residues(self.residues)
        self.assertEqual(sgi.core.find_subgraph_isomorphims(built_t_graph,t_graph),[{0: 'A', 1: 'B', 2: 'D', 3: 'C'}])
        #########################
        #Photometric isomorphism: needs
        n=sgi.core.number_of_brother_links(p_graph)
        built_p_graph=sgi.core.photometric_graph_from_residues(image,new_residues)
        sgi.core.build_similarities(image,new_residues,built_p_graph,n)

        tmp_ref=sgi.core.transitive_closure(p_graph)
        built_p_graph=sgi.core.transitive_closure(built_p_graph)
        self.assertEqual(sgi.core.find_subgraph_isomorphims(built_p_graph,tmp_ref),[{0: 'A', 1: 'B', 2: 'C', 3: 'D'},{0: 'A', 1: 'B', 2: 'D', 3: 'C'}])

    ####################################
    #   IDENTIFICATION
    ####################################
    def test05(self):
        tp_model.set_image(image)
        tp_model.set_region('A',regionA)
        tp_model.set_targets(['B','C','D'])
        tp_model.identify_from_residues(self.residues)
        self.assertTrue(np.array_equal(tp_model.get_region('A'),regionA))
        self.assertTrue(np.array_equal(tp_model.get_region('B'),regionB))
        self.assertTrue(np.array_equal(tp_model.get_region('C'),regionC))
        self.assertTrue(np.array_equal(tp_model.get_region('D'),regionD))

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRecognitionUseCase3)
    unittest.TextTestRunner(verbosity=2).run(suite)
