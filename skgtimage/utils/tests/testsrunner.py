#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause
__author__      = "Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France"
__copyright__   = "Copyright (C) 2015 Jean-Baptiste Fasquel"



import unittest

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover('.')
    unittest.TextTestRunner(verbosity=2).run(suite)

