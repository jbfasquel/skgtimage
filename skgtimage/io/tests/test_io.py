#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import unittest
import skgtimage as skit
import os

import numpy as np

image=np.ones((10,10),np.uint8)
for i in range(0,10): image[:,i]=10*i
#Region A
A=np.ones((10,10),np.uint8)
#Region B
B=np.zeros((10,10),np.uint8);B[1:9,1:5]=1
#Region C
C=np.zeros((10,10),np.uint8);C[2:4,2:4]=1
#Region D
D=np.zeros((10,10),np.uint8);D[6:8,2:4]=1
#Region E
E=np.zeros((10,10),np.uint8);E[1:5,6:9]=1


class TestIO(unittest.TestCase):
    #INITIALIZATION
    def setUp(self):
        self.t_graph=skit.core.IrDiGraph()
        self.t_graph.add_node('A')
        self.t_graph.set_region('A',A)
        self.t_graph.add_node('B',mykey='myvalue')

        self.p_graph=skit.core.IrDiGraph()
        self.p_graph.add_node('A')
        self.p_graph.add_node('B')
        self.p_graph.add_edge('A','B')
        self.p_graph.set_region('A',A)

        self.current_module_path=skit.io.__path__[0]
        self.current_test_path=os.path.join(os.path.join(self.current_module_path,'tests/'),'dirio/')


        #self.tp_model=skit.core.TPModel(self.t_graph,[self.p_graph])

    ############
    #SAVE GRAPH WITH PICKLE
    ############
    def test01(self):
        for f in os.listdir(self.current_test_path): os.remove(os.path.join(self.current_test_path,f))

        skit.io.save_graph_pickle(os.path.join(self.current_test_path,'test.pkl'),self.t_graph)
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'test.pkl')))
        reloaded=skit.io.load_graph_pickle(os.path.join(self.current_test_path,'test.pkl'))

    ############
    #SAVE GRAPH "HUMAN READABLE"
    ############
    def test02(self):
        for f in os.listdir(self.current_test_path): os.remove(os.path.join(self.current_test_path,f))

        #Save1
        skit.io.save_graph('test',self.t_graph,directory=self.current_test_path)
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'test.svg')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'test.png')))
        os.remove(os.path.join(self.current_test_path,'test.svg'))
        os.remove(os.path.join(self.current_test_path,'test.png'))
        #Save2
        skit.io.save_graph('test',self.t_graph,tree=False,directory=self.current_test_path)
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'test.svg')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'test.png')))
        os.remove(os.path.join(self.current_test_path,'test.svg'))
        os.remove(os.path.join(self.current_test_path,'test.png'))
        #Save3
        skit.io.save_graph('test',self.t_graph,['A'],directory=self.current_test_path)
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'test.svg')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'test.png')))
        os.remove(os.path.join(self.current_test_path,'test.svg'))
        os.remove(os.path.join(self.current_test_path,'test.png'))
        #Save4
        skit.io.save_graph('test',self.t_graph,[],directory=self.current_test_path)
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'test.svg')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'test.png')))
        os.remove(os.path.join(self.current_test_path,'test.svg'))
        os.remove(os.path.join(self.current_test_path,'test.png'))
        #Save5
        skit.io.save_graph('test',self.t_graph,[],directory=self.current_test_path,save_regions=True)
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'region_A.png')))
        os.remove(os.path.join(self.current_test_path,'region_A.png'))
        os.remove(os.path.join(self.current_test_path,'test.svg'))
        os.remove(os.path.join(self.current_test_path,'test.png'))

    ############
    #SAVE MODEL "HUMAN READABLE" (TOPOLOGICAL GRAPH, PHOTOMETRIC GRAPHS, IMAGE)
    ############
    def test03(self):
        pass
        '''
        for f in os.listdir(self.current_test_path): os.remove(os.path.join(self.current_test_path,f))

        skit.io.save_model(self.current_test_path,self.tp_model)
        self.assertTrue(not(os.path.exists(os.path.join(self.current_test_path,'image.png'))))

        for f in os.listdir(self.current_test_path): os.remove(os.path.join(self.current_test_path,f))
        self.tp_model.set_image(image)
        skit.io.save_model(self.current_test_path,self.tp_model)
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'image.png')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'region_A.png')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'topology_apriori.svg')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'topology_apriori.png')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'topology_context.svg')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'topology_context.png')))

        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'photometry_apriori_0.svg')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'photometry_apriori_0.png')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'photometry_context_0.svg')))
        self.assertTrue(os.path.exists(os.path.join(self.current_test_path,'photometry_context_0.png')))
        '''


        
if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIO)
    unittest.TextTestRunner(verbosity=2).run(suite)
