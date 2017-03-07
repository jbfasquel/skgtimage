import unittest,os
import skgtimage as skgti
import numpy as np
import scipy as sp;from scipy import misc

inclusion="C<B<A"
photometry="C>B>A"
image_base=np.array([[1, 1, 1, 1, 1, 1, 1],
                  [1, 2, 2, 2, 1, 1, 1],
                  [1, 2, 3, 2, 1, 1, 1],
                  [1, 2, 2, 2, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1]], np.uint8)

class TestIOFromDir(unittest.TestCase):
    def setUp(self):
        """ Prepare """
        #######
        # GRAY2D
        #######
        self.save_dir_gray2D = "fromdir_gray2D/"
        if not os.path.exists(self.save_dir_gray2D): os.mkdir(self.save_dir_gray2D)
        self.image_gray2D = sp.ndimage.interpolation.zoom(image_base, 10, order=0)
        sp.misc.imsave(self.save_dir_gray2D + "image.png", self.image_gray2D * 50)
        np.save(self.save_dir_gray2D + "image.npy", self.image_gray2D)
        sp.misc.imsave(self.save_dir_gray2D + "region_A.png", np.where(self.image_gray2D == 1, 255, 0) + np.where(self.image_gray2D == 0, 255, 0))
        np.save(self.save_dir_gray2D + "region_A.npy", np.where(self.image_gray2D == 1, 255, 0) + np.where(self.image_gray2D == 0, 255, 0))
        sp.misc.imsave(self.save_dir_gray2D + "region_B.png", np.where(self.image_gray2D == 2, 255, 0))
        np.save(self.save_dir_gray2D + "region_B.npy", np.where(self.image_gray2D == 2, 255, 0))
        sp.misc.imsave(self.save_dir_gray2D + "region_C.png", np.where(self.image_gray2D == 3, 255, 0))
        np.save(self.save_dir_gray2D + "region_C.npy", np.where(self.image_gray2D == 3, 255, 0))
        #######
        # COLOR2D
        #######
        self.save_dir_color2D = "fromdir_color2D/"
        if not os.path.exists(self.save_dir_color2D): os.mkdir(self.save_dir_color2D)
        self.image_color2D = sp.ndimage.interpolation.zoom(image_base, 10, order=0)
        sp.misc.imsave(self.save_dir_color2D + "region_A.png", np.where(self.image_color2D == 1, 255, 0) + np.where(self.image_color2D == 0, 255, 0))
        np.save(self.save_dir_color2D + "region_A.npy", np.where(self.image_color2D == 1, 255, 0) + np.where(self.image_color2D == 0, 255, 0))
        sp.misc.imsave(self.save_dir_color2D + "region_B.png", np.where(self.image_color2D == 2, 255, 0))
        np.save(self.save_dir_color2D + "region_B.npy", np.where(self.image_color2D == 2, 255, 0))
        sp.misc.imsave(self.save_dir_color2D + "region_C.png", np.where(self.image_color2D == 3, 255, 0))
        np.save(self.save_dir_color2D + "region_C.npy", np.where(self.image_color2D == 3, 255, 0))
        # become color
        self.image_color2D = np.dstack(tuple([self.image_color2D for i in range(0, 3)]))
        sp.misc.imsave(self.save_dir_color2D + "image.png", self.image_color2D * 50)
        np.save(self.save_dir_color2D + "image.npy", self.image_color2D)
        #######
        # GRAY3D
        #######
        self.save_dir_gray3D = "fromdir_gray3D/"
        if not os.path.exists(self.save_dir_gray3D): os.mkdir(self.save_dir_gray3D)
        self.image_gray3D = np.dstack(tuple([np.ones(image_base.shape), image_base, np.ones(image_base.shape)]))
        self.image_gray3D = sp.ndimage.interpolation.zoom(self.image_gray3D, 10, order=0)
        np.save(self.save_dir_gray3D + "image.npy", self.image_gray3D)
        np.save(self.save_dir_gray3D + "region_A.npy", np.where(self.image_gray3D == 1, 255, 0) + np.where(self.image_gray3D == 0, 255, 0))
        np.save(self.save_dir_gray3D + "region_B.npy", np.where(self.image_gray3D == 2, 255, 0))
        np.save(self.save_dir_gray3D + "region_C.npy", np.where(self.image_gray3D == 3, 255, 0))

    ####################################
    #   GRAPHS FROM RESIDUES
    ####################################
    def test01(self):
        """ From dir: grayscale 2D"""
        t, p = skgti.io.from_dir(self.save_dir_gray2D, mc=False)
        self.assertTrue(set(t.nodes()) == set(["A", "B", "C"]))
        for i in t.nodes():
            self.assertTrue (t.get_region(i) is not None)
            self.assertTrue (t.get_mean_intensity(i) is not None)
            self.assertTrue (p.get_mean_intensity(i) is not None)

        skgti.io.save_graph(t,name="t",directory=self.save_dir_gray2D+"from_dir/")
        skgti.io.save_graph(p,name="p",directory=self.save_dir_gray2D+"from_dir/")
        skgti.io.save_graph_regions(t,directory=self.save_dir_gray2D+"from_dir/")
        skgti.io.save_intensities(t,directory=self.save_dir_gray2D+"from_dir/",filename="intensities_t")
        skgti.io.save_intensities(p,directory=self.save_dir_gray2D+"from_dir/",filename="intensities_p")


    def test02(self):
        """ From dir: color 2D"""
        t,p=skgti.io.from_dir(self.save_dir_color2D, mc=True)
        self.assertTrue(set(t.nodes())==set(["A","B","C"]))
        for i in t.nodes():
            self.assertTrue (t.get_region(i) is not None)
            self.assertTrue (t.get_mean_intensity(i) is not None)
            self.assertTrue (p.get_mean_intensity(i) is not None)
        skgti.io.save_graph(t,name="t",directory=self.save_dir_color2D+"from_dir/")
        skgti.io.save_graph(p,name="p",directory=self.save_dir_color2D+"from_dir/")
        skgti.io.save_graph_regions(t,directory=self.save_dir_color2D+"from_dir/")
        skgti.io.save_intensities(t,directory=self.save_dir_color2D+"from_dir/",filename="intensities_t")
        skgti.io.save_intensities(p,directory=self.save_dir_color2D+"from_dir/",filename="intensities_p")


    def test03(self):
        """ From dir: gray 3D"""
        t,p=skgti.io.from_dir(self.save_dir_gray3D, mc=False)
        self.assertTrue(set(t.nodes())==set(["A","B","C"]))
        for i in t.nodes():
            self.assertTrue (t.get_region(i) is not None)
            self.assertTrue (t.get_mean_intensity(i) is not None)
            self.assertTrue (p.get_mean_intensity(i) is not None)
        skgti.io.save_graph(t,name="t",directory=self.save_dir_gray3D+"from_dir/")
        skgti.io.save_graph(p,name="p",directory=self.save_dir_gray3D+"from_dir/")
        skgti.io.save_graph_regions(t,directory=self.save_dir_gray3D+"from_dir/",slices=list(range(0,self.image_gray3D.shape[2])))
        skgti.io.save_intensities(t,directory=self.save_dir_gray3D+"from_dir/",filename="intensities_t")
        skgti.io.save_intensities(p,directory=self.save_dir_gray3D+"from_dir/",filename="intensities_p")

    def test04(self):
        """ From dir: with description and/or image provided """
        t, p = skgti.io.from_dir(self.save_dir_gray2D, self.image_gray2D, inclusion, photometry, mc=False)
        self.assertTrue(set(t.nodes()) == set(["A", "B", "C"]))
        for i in t.nodes():
            self.assertTrue(t.get_region(i) is not None)
            self.assertTrue(t.get_mean_intensity(i) is not None)
            self.assertTrue(p.get_mean_intensity(i) is not None)

        t, p = skgti.io.from_dir(self.save_dir_gray2D, t_desc=inclusion, p_desc=photometry, mc=False)
        self.assertTrue(set(t.nodes()) == set(["A", "B", "C"]))
        for i in t.nodes():
            self.assertTrue(t.get_region(i) is not None)
            self.assertTrue(t.get_mean_intensity(i) is not None)
            self.assertTrue(p.get_mean_intensity(i) is not None)

    def test05(self):
        """ From dir: check exception is raised when regions are not mutually excluded """
        sp.misc.imsave(self.save_dir_gray2D + "region_B.png",np.where(self.image_gray2D > 1, 255, 0))
        #t, p = skgti.io.from_dir_new(self.save_dir_gray2D, mc=False)
        with self.assertRaises(Exception):
            t, p = skgti.io.from_dir(self.save_dir_gray2D, mc=False)

if __name__ == '__main__':
    #With verbose
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIOFromDir)
    unittest.TextTestRunner(verbosity=2).run(suite)
