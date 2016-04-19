#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import unittest
import skgtimage as sgi
import numpy as np

#################
#Helper function
#################
def do_correspond(edges1,edges2):
    if len(edges1) != len(edges2): return False
    result=True
    for e in edges1:
        if e not in edges2: result=False
    return result

class TestGraphBrothers(unittest.TestCase):
    def setUp(self):
        pass

    ####################################
    #   A ->  B  <->  C
    ####################################
    def test01(self):
        ####################
        #TESTED GRAPH
        ####################
        ref_graph=sgi.core.IrDiGraph()
        ref_graph.add_node('A')
        ref_graph.add_node('B')
        ref_graph.add_node('C')
        #
        ref_graph.add_edge('A','B')
        ref_graph.add_edge('B','C')
        ref_graph.add_edge('C','B')

        ####################
        #EXPECTED RESULT
        ####################
        expected_graphs=[
                        [('B', 'C'), ('A', 'B')],
                        [('A', 'C'), ('C', 'B')]
                         ]
        ####################
        #CHECK
        ####################
        all_graphs=sgi.core.compute_possible_graphs(ref_graph)
        for exp in expected_graphs:
            found_matching=False
            for g in all_graphs:
                if do_correspond(exp,g.edges()) : found_matching=True
            self.assertTrue(found_matching,msg=str(exp)+" not found in\n"+str([g.edges() for g in all_graphs]))
    ####################################
    #   bottom ->A<->B->C<->D<->E->F
    ####################################
    def test02(self):
        ####################
        #TESTED GRAPH
        ####################
        ref_graph=sgi.core.IrDiGraph()
        ref_graph.add_node('bottom')
        #ref_graph.add_node('X')
        ref_graph.add_node('A')
        ref_graph.add_node('B')
        ref_graph.add_edge('bottom','A')
        ref_graph.add_edge('B','A')
        ref_graph.add_edge('A','B')
        ref_graph.add_node('C')
        ref_graph.add_edge('B','C')
        ref_graph.add_node('D')
        ref_graph.add_edge('C','D')
        ref_graph.add_edge('D','C')
        ref_graph.add_node('E')
        ref_graph.add_edge('D','E')
        ref_graph.add_edge('E','D')
        
        ref_graph.add_node('F')
        ref_graph.add_edge('E','F')
        
        ####################
        #EXPECTED RESULT
        ####################
        expected_graphs=[
            [('D', 'C'), ('A', 'E'), ('E', 'D'), ('B', 'A'), ('bottom', 'B'), ('C', 'F')],
            [('D', 'C'), ('A', 'B'), ('E', 'D'), ('B', 'E'), ('bottom', 'A'), ('C', 'F')],
            [('D', 'F'), ('A', 'E'), ('E', 'C'), ('B', 'A'), ('bottom', 'B'), ('C', 'D')],
            [('D', 'F'), ('A', 'B'), ('E', 'C'), ('B', 'E'), ('bottom', 'A'), ('C', 'D')],
            [('D', 'E'), ('A', 'D'), ('E', 'C'), ('B', 'A'), ('bottom', 'B'), ('C', 'F')],
            [('D', 'E'), ('A', 'B'), ('E', 'C'), ('B', 'D'), ('bottom', 'A'), ('C', 'F')],
            [('D', 'C'), ('A', 'D'), ('E', 'F'), ('B', 'A'), ('bottom', 'B'), ('C', 'E')],
            [('D', 'C'), ('A', 'B'), ('E', 'F'), ('B', 'D'), ('bottom', 'A'), ('C', 'E')],
            [('D', 'F'), ('A', 'C'), ('E', 'D'), ('B', 'A'), ('bottom', 'B'), ('C', 'E')],
            [('D', 'F'), ('A', 'B'), ('E', 'D'), ('B', 'C'), ('bottom', 'A'), ('C', 'E')],
            [('D', 'E'), ('A', 'C'), ('E', 'F'), ('B', 'A'), ('bottom', 'B'), ('C', 'D')],
            [('D', 'E'), ('A', 'B'), ('E', 'F'), ('B', 'C'), ('bottom', 'A'), ('C', 'D')]
                         ]
        ####################
        #CHECK
        ####################
        all_graphs=sgi.core.compute_possible_graphs(ref_graph)
        print(all_graphs)
        for exp in expected_graphs:
            found_matching=False
            for g in all_graphs:
                if do_correspond(exp,g.edges()) : found_matching=True
            self.assertTrue(found_matching,msg=str(exp)+" not found in\n"+str([g.edges() for g in all_graphs]))


