import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

#########
# MISC INFORMATIONS
#########
truth_dir="Database/image03/truth_top/"
save_dir="Database/image03/top_meanshift_ok_5classes_versus4expected_refactorying/"

#########
# A PRIORI KNOWLEDGE
#########
t_desc="C,D<B<A;E,F<D;G<C;H<E"
p_desc="G=B<E=F<H=D<A=C"

#########
# IMAGE: COLOR AND GRAY
#########
image_rgb=sp.misc.imread(os.path.join(truth_dir,"image.png"))
roi=sp.misc.imread(os.path.join(truth_dir,"region_A.png"))
image=skgti.utils.rgb2gray(image_rgb)

#########
# MEANSHIFT ON COLOR IMAGE
#########
image_chsv=skgti.utils.rgb2chsv(image_rgb)
label=skgti.utils.mean_shift(image_chsv,0.1,roi,True,True) #0.1 OK

built_t_graph,built_p_graph=skgti.core.from_labelled_image_refactorying(image,label)
skgti.io.save_graph(built_p_graph,name="photo",directory=save_dir+"tmp/",tree=True)
skgti.io.save_graphregions(built_p_graph,directory=save_dir+"tmp/")
skgti.io.save_intensities(built_p_graph,directory=save_dir+"tmp/")

#skgti.io.plot_graph_with_regions_refactorying(built_t_graph);plt.show();quit()
# RECOGNITION
#id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=True,thickness=2,filtering=False,verbose=True)
#matcher=skgti.core.matcher_factory_refactorying(image,label,t_desc,p_desc,roi=roi,manage_bounds=True,thickness=2,filtering=False)
#matcher.compute_maching()

#skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
#skgti.io.pickle_matcher(matcher,save_dir+"matcher.pkl")
#quit()
matcher=skgti.io.unpickle_matcher(save_dir+"matcher.pkl")
#plt.imshow(matcher.built_p_graph.get_region(0));plt.show();quit()
matcher.compute_merge()
matcher.update_final_graph()
skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)