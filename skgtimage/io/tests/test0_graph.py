import unittest,os
import skgtimage as skgti
import numpy as np

class TestIOGraph(unittest.TestCase):
    ####################################
    #   GRAPHS FROM RESIDUES
    ####################################
    def test01(self):
        inclusion = "C<B<A"
        photometry = "C>B>A"
        image = np.array([[1, 1, 1, 1, 1, 1, 1],
                          [1, 2, 2, 2, 1, 1, 1],
                          [1, 2, 3, 2, 1, -5, 1],
                          [1, 2, 2, 2, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1]], np.int8)

        id2region, r = skgti.utils.recognize(image, image, inclusion, photometry, roi=None, verbose=True)
        t, p = r.intermediate_graphs[-1]
        #IO GRAPH
        skgti.io.save_graph(t, "graph", directory="save_io/")
        self.assertTrue(os.path.exists("save_io/graph.png"))
        #IO GRAPH SIMPLE
        skgti.io.save_graph_basic(t, "graph_simple", directory="save_io/")
        self.assertTrue(os.path.exists("save_io/graph_simple.png"))
        #IO MACHTING
        skgti.io.save_matching(t, r.ref_t_graph, [skgti.io.matching2links(r.matching), r.ordered_merges],
                               ['red', 'green'], name="matching", directory="save_io/", tree=True)
        self.assertTrue(os.path.exists("save_io/matching.png"))
        self.assertTrue(os.path.exists("save_io/matching.svg"))
        #IO MACHTING SIMPLE
        skgti.io.save_matching_basic(t, r.ref_t_graph, [skgti.io.matching2links(r.matching), r.ordered_merges],
                                     ['red', 'green'], name="matching_simple", directory="save_io/", tree=True)
        self.assertTrue(os.path.exists("save_io/matching_simple.png"))
        self.assertTrue(os.path.exists("save_io/matching_simple.svg"))
        #IO INTENSITIES
        skgti.io.save_intensities(p, directory="save_io", filename="intensities")
        self.assertTrue(os.path.exists("save_io/intensities.csv"))
        #IO REGIONS
        skgti.io.save_graph_regions(t, directory="save_io")
        for n in t.nodes():
            self.assertTrue(os.path.exists("save_io/region_" + str(n) + ".npy"))
            self.assertTrue(os.path.exists("save_io/region_" + str(n) + ".png"))
        #############################
        #REMOVE FOR NEXT TEST
        '''
        os.remove("save_io/graph.png");self.assertFalse(os.path.exists("save_io/graph.png"))
        os.remove("save_io/graph.svg");self.assertFalse(os.path.exists("save_io/graph.svg"))
        os.remove("save_io/graph_simple.png");self.assertFalse(os.path.exists("save_io/graph_simple.png"))
        os.remove("save_io/graph_simple.svg");self.assertFalse(os.path.exists("save_io/graph_simple.svg"))
        os.remove("save_io/matching.png");self.assertFalse(os.path.exists("save_io/matching.png"))
        os.remove("save_io/matching.svg");self.assertFalse(os.path.exists("save_io/matching.svg"))
        os.remove("save_io/matching_simple.png");self.assertFalse(os.path.exists("save_io/matching_simple.png"))
        os.remove("save_io/matching_simple.svg");self.assertFalse(os.path.exists("save_io/matching_simple.svg"))
        for n in t.nodes():
            os.remove("save_io/region_" + str(n) + ".npy");self.assertFalse(os.path.exists("save_io/region_" + str(n) + ".npy"))
            os.remove("save_io/region_" + str(n) + ".png");self.assertFalse(os.path.exists("save_io/region_" + str(n) + ".png"))
        os.remove("save_io/intensities.csv");self.assertFalse(os.path.exists("save_io/intensities.csv"))
        os.rmdir("save_io/")
        '''

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIOGraph)
    unittest.TextTestRunner(verbosity=2).run(suite)
