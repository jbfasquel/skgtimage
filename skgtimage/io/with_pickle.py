#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import networkx as nx
import pickle

def pickle_matcher(matcher,filename):
    file=open(filename, 'wb')
    pickle.dump(matcher,file)
    file.close()

def unpickle_matcher(filename):
    file=open(filename, 'rb')
    matcher=pickle.load(file)
    return matcher


def save_graph_pickle(filename,g):
    nx.write_gpickle(g, filename)

def load_graph_pickle(filename):
    return nx.read_gpickle(filename)

