#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import unittest
import skgtimage as skit
import data,os

class TestIO(unittest.TestCase):
    """
    Path problem
    """
    #INITIALIZATION OF BASES AND ROLES
    def setUp(self):
        self.g=skit.core.IrDiGraph(None,image=data.image)
        self.g.add_node('A')
        self.g.set_region('A',data.A)
        self.g.add_node('B',mykey='myvalue')
        self.current_module_path=skit.core.__path__[0]
        self.current_test_path=os.path.join(os.path.join(self.current_module_path,'tests/'),'dirio/')
        
        #os.path.dirname(skit.core)
    #CHECK NUMBER OF COMPOSITIONS PER BASE OBJECT 
    def test01(self):
        for f in os.listdir(self.current_test_path): os.remove(os.path.join(self.current_test_path,f))

        skit.core.save_pickle(os.path.join(self.current_test_path,'test.pkl'),self.g)
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'test.pkl')))
        reloaded=skit.core.load_pickle(os.path.join(self.current_test_path,'test.pkl'))

    def test02(self):
        for f in os.listdir(self.current_test_path): os.remove(os.path.join(self.current_test_path,f))

        skit.core.export_human_readable(self.current_test_path,self.g)
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'core.pkl')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'core.dot')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'core.svg')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'core.png')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'image.png')))
        

        
if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIO)
    unittest.TextTestRunner(verbosity=2).run(suite)
