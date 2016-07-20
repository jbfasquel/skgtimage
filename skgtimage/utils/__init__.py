#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

from __future__ import absolute_import

from .color import *
from .factory import *
from .histogram import *
from .meanshift import *
from .roi import *
from .evaluation import *

try:
    import sklearn
except:
    print("Warning: cannot import kmeans because sklearn is not installed")
else:
    from .kmeans import *

