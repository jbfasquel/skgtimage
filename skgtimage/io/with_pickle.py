#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import networkx as nx

def save_graph_pickle(filename,g):
    nx.write_gpickle(g, filename)

def load_graph_pickle(filename):
    return nx.read_gpickle(filename)

