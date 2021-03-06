import unittest,os
import scipy as sp;from scipy.misc import imread
import skgtimage as skgti

inclusion="2E<2D;2G<F;2D,F,2H,2I<C<B<A;C,1D<B<A;1E,1G<1D;1H<1E"
photometry="B=F=2D=2H=1E=1G<2I=2E=1H=1D<C=2G=A"
truth_dir="data_color/truth/"

class TestOnColor2D(unittest.TestCase):
    def setUp(self):
        self.image = imread(truth_dir + "image.png")

    def compare(self,r):
        self.truth_t_graph, _ = skgti.io.from_dir(truth_dir, mc=True)
        classif, _, _ = skgti.utils.goodclassification_rate_graphs(r.relabelled_final_t_graph, self.truth_t_graph, r.roi, 3)
        region2sim = skgti.utils.similarity_indices_graph_regions(r.relabelled_final_t_graph, self.truth_t_graph, 2)
        return classif,region2sim

    def test01(self):
        """ Meanshift + ROI """
        segmentation = imread("data_color/meanshift_labelling.png")
        roi = skgti.core.fill_region(imread(truth_dir + "region_A.png"))
        id2region, r = skgti.utils.recognize(self.image, segmentation, inclusion, photometry, roi=roi,mc=True, bg=False, verbose=True)
        # COMPARISON WITH TRUTH
        classif, region2sim=self.compare(r)
        self.assertAlmostEqual(classif, 0.976,3)
        self.assertAlmostEqual(min(region2sim.values()), 0.89, 2)
        self.assertAlmostEqual(max(region2sim.values()), 1, 2)


    def test02(self):
        """ Meanshift """
        segmentation = imread("data_color/meanshift_labelling.png")
        id2region, r = skgti.utils.recognize(self.image, segmentation, inclusion, photometry, mc=True, bg=True, verbose=True)
        # COMPARISON WITH TRUTH
        classif, region2sim=self.compare(r)
        self.assertAlmostEqual(classif, 0.979,3) #not reproducible ?
        self.assertAlmostEqual(min(region2sim.values()), 0.89, 2)
        self.assertAlmostEqual(max(region2sim.values()), 0.99, 2)


    def test03(self):
        """ Meanshift + rag and phot merging """
        segmentation = imread("data_color/meanshift_labelling.png")
        id2region, r = skgti.utils.recognize(self.image, segmentation, inclusion, photometry, mc=True, bg=True,
                                             verbose=True,rag=60,merge=5)
        # COMPARISON WITH TRUTH
        classif, region2sim = self.compare(r)
        self.assertAlmostEqual(classif, 0.976, 3)
        self.assertAlmostEqual(min(region2sim.values()), 0.87, 2)
        self.assertAlmostEqual(max(region2sim.values()), 0.99, 2)


if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOnColor2D)
    unittest.TextTestRunner(verbosity=2).run(suite)
