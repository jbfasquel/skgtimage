#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


try:
    import pygraphviz
except:
    print("Warning: cannot import skgtimage.io.with_graphviz because pygraphviz is not installed")
else:
    from .test_io import *
