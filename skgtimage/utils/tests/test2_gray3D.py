import unittest
import numpy as np
#import scipy as sp;from scipy.misc import imread
import skgtimage as skgti

inclusion="tumor,vessel<liver"
photometry="tumor<liver<vessel"
truth_dir="data_gray3D/truth/"
class TestOnGrayscale3D(unittest.TestCase):

    def test01(self):
        """ Meanshift """
        image = np.load(truth_dir + "image.npy")
        roi = np.load("data_gray3D/roi.npy")
        segmentation = np.load("data_gray3D/meanshift_labelling.npy")
        # INTERPRETATION (PROPOSED METHOD)
        id2region, r = skgti.utils.recognize(image, segmentation, inclusion, photometry, roi=roi, bound_thickness=2, min_size=300,
                                 mc=False, verbose=True)
        # COMPARISON WITH TRUTH
        truth_t_graph, _ = skgti.io.from_dir(truth_dir, image, inclusion, photometry)
        result_t_graph = r.relabelled_final_t_graph
        map = skgti.core.get_node2mean(truth_t_graph, round=True)
        truth_image = truth_t_graph.get_labelled(mapping=map)
        result_image = result_t_graph.get_labelled(mapping=map)

        classif = skgti.utils.goodclassification_rate(result_image, truth_image, 3)
        region2sim = skgti.utils.compute_sim_between_graph_regions(result_t_graph, truth_t_graph, 2)
        print(classif, region2sim)
        self.assertAlmostEqual(classif, 0.95,2)
        self.assertAlmostEqual(region2sim['tumor'], 0.68, 2)
        self.assertAlmostEqual(region2sim['vessel'], 0.54, 2)
        self.assertAlmostEqual(region2sim['liver'], 0.90, 2)



if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOnGrayscale3D)
    unittest.TextTestRunner(verbosity=2).run(suite)
