#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

# IMAGE
image=np.array([[0.0, 0.2, 0.0, 0.0, 0.0],
                [0.2, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.1, 2.0, 0.9, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]])
#SEGMENTATION AS LABELS
label=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 4, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])

#SEGMENTATION AS REGIONS
residues=[np.where(label==i,1,0) for i in [0,1,4]]

#WITH BOUNDARY ARTEFACT
label_b=np.array([[0, 6, 0, 0, 0],
                [6, 1, 1, 1, 0],
                [0, 1, 4, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])


class TestFactoryFromLabelRegions(unittest.TestCase):
    ####################################
    #   EXECUTE TWICE THE FUNCTION: SHOULD NOT MODIFY RESIDUES AND NEW RESIDUES
    ####################################
    def check(self,built_t_graph,built_p_graph):
        #Check nodes and edges
        self.assertEqual(set(built_t_graph.nodes()),set([0, 1, 2]))
        self.assertEqual(set(built_t_graph.edges()),set([(1, 0), (2, 1)]))
        self.assertEqual(set(built_p_graph.nodes()),set([0, 1, 2]))
        self.assertEqual(set(built_p_graph.edges()),set([(0, 1), (1, 2)]))
        #Check regions
        self.assertTrue(np.array_equal(built_t_graph.get_region(0),residues[0]))
        self.assertTrue(np.array_equal(built_t_graph.get_region(1),residues[1]))
        self.assertTrue(np.array_equal(built_t_graph.get_region(2),residues[2]))
        #Check mean intensities
        self.assertEqual(built_p_graph.get_mean_residue_intensity(0),0.025)
        self.assertEqual(built_p_graph.get_mean_residue_intensity(1),1.0)
        self.assertEqual(built_p_graph.get_mean_residue_intensity(2),2.0)

    #From labels
    def test01(self):
        built_t_graph,built_p_graph=sgi.core.from_labelled_image(image,label)
        self.check(built_t_graph,built_p_graph)
        built_t_graph,built_p_graph=sgi.core.from_labelled_image(image,label,roi=np.ones(image.shape))
        self.check(built_t_graph,built_p_graph)
    #From regions (residues)
    def test02(self):
        built_t_graph,built_p_graph=sgi.core.from_regions(image,residues)
        self.check(built_t_graph,built_p_graph)

    #With boundary artefact
    def test03(self):
        built_t_graph,built_p_graph=sgi.core.from_labelled_image(image,label_b,None,False)
        self.assertEqual(set(built_t_graph.nodes()),set([0, 1, 2, 3]))
        self.assertEqual(set(built_p_graph.nodes()),set([0, 1, 2, 3]))
        #Correction
        #built_t_graph,built_p_graph=sgi.core.from_labelled_image(image,label_b,None,True,1)
        #self.check(built_t_graph,built_p_graph)





if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFactoryFromLabelRegions)
    unittest.TextTestRunner(verbosity=2).run(suite)
