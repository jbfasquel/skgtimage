#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

###############################
#
###############################
class TestIdentificationFeasability(unittest.TestCase):
    ###############################
    # Topology: B,C -> A
    # Photometry: A->B->C
    # Identification : feasible
    ###############################
    def test01(self):
        t_graph=sgi.core.graph_factory("B,C<A")
        p_graph=sgi.core.graph_factory("A<B<C")
        tp_model=sgi.core.TPModel(t_graph,[p_graph])
        tp_model.set_region('A',np.array([]))
        self.assertTrue(tp_model.is_identification_feasible(['B']))
    ###############################
    # Topology: B,C -> A
    # Photometry: A->B<->C
    # Identification : not feasible
    ###############################
    def test02(self):
        t_graph=sgi.core.graph_factory("B,C<A")
        p_graph=sgi.core.graph_factory("A<B=C")
        tp_model=sgi.core.TPModel(t_graph,[p_graph])
        tp_model.set_region('A',np.array([]))
        self.assertFalse(tp_model.is_identification_feasible(['B']))

    ###############################
    # Topology: C->B->A
    #              D--^
    # Photometry: A->B->C<->D
    # Identification : feasible
    ###############################
    def test03(self):
        t_graph=sgi.core.graph_factory("C<B<A;D<A")
        p_graph=sgi.core.graph_factory("A<B<C=D")
        tp_model=sgi.core.TPModel(t_graph,[p_graph])
        tp_model=sgi.core.TPModel(t_graph,[p_graph])
        tp_model.set_region('A',np.array([]))
        self.assertTrue(tp_model.is_identification_feasible(['B']))

    ###############################
    # Topology:    B->A
    #              C--^
    #              D--^
    # Photometry: A->B->C<->D
    # Identification : not feasible
    ###############################
    def test04(self):
        t_graph=sgi.core.graph_factory("B,C,D<A")
        p_graph=sgi.core.graph_factory("A<B<C=D")
        tp_model=sgi.core.TPModel(t_graph,[p_graph])
        tp_model=sgi.core.TPModel(t_graph,[p_graph])
        tp_model.set_region('A',np.array([]))
        self.assertFalse(tp_model.is_identification_feasible(['B']))



if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIdentificationFeasability)
    unittest.TextTestRunner(verbosity=2).run(suite)
