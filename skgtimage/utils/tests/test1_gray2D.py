import unittest
import scipy as sp;from scipy.misc import imread
import skgtimage as skgti

inclusion="text<paper<file"
photometry="text<file<paper"
truth_dir="data_gray/truth/"
class TestOnGrayscale2D(unittest.TestCase):
    def setUp(self):
        self.image = imread(truth_dir + "image.png")[::2, ::2]

    def compare(self,r):
        self.truth_t_graph, _ = skgti.io.from_dir2(truth_dir, color=False)
        self.truth_t_graph.downsample(2)
        result_t_graph = r.relabelled_final_t_graph
        truth_image = self.truth_t_graph.get_labelled()
        result_image = result_t_graph.get_labelled()
        classif = skgti.utils.goodclassification_rate(result_image, truth_image, 3)
        region2sim = skgti.utils.compute_sim_between_graph_regions(result_t_graph, self.truth_t_graph, 2)
        return classif,region2sim

    def test01(self):
        """ Meanshift """
        segmentation = imread("data_gray/meanshift_labelling.png")
        id2region, r = skgti.utils.recognize(self.image, segmentation, inclusion, photometry, bg=True)
        # COMPARISON WITH TRUTH
        classif, region2sim=self.compare(r)
        self.assertAlmostEqual(classif, 0.975,3)
        self.assertAlmostEqual(region2sim['text'], 0.61, 2)
        self.assertAlmostEqual(region2sim['paper'], 0.99, 2)
        self.assertAlmostEqual(region2sim['file'], 0.98, 2)

    def test02(self):
        """ Quickshift (rag merge) """
        segmentation = imread("data_gray/quickshift_labelling.png")
        id2region, r = skgti.utils.recognize(self.image, segmentation, inclusion, photometry, bg=True,rag=20)
        # COMPARISON WITH TRUTH
        classif, region2sim = self.compare(r)
        #self.assertAlmostEqual(classif, 0.987, 3)
        self.assertAlmostEqual(region2sim['text'], 0.67, 2)
        self.assertAlmostEqual(region2sim['paper'], 0.99, 2)
        self.assertAlmostEqual(region2sim['file'], 0.99, 2)

    def test03(self):
        """ Meanshift + ROI """
        segmentation = imread("data_gray/meanshift_labelling.png")
        roi = skgti.core.fill_region(imread("data_gray/truth/region_file.png"))[::2, ::2]
        id2region, r = skgti.utils.recognize(self.image, segmentation, inclusion, photometry, roi=roi)
        # COMPARISON WITH TRUTH
        classif, region2sim = self.compare(r)
        self.assertAlmostEqual(classif, 0.994, 3)
        self.assertAlmostEqual(region2sim['text'], 0.61, 2)
        self.assertAlmostEqual(region2sim['paper'], 0.99, 2)
        self.assertAlmostEqual(region2sim['file'], 1, 2)


if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOnGrayscale2D)
    unittest.TextTestRunner(verbosity=2).run(suite)
