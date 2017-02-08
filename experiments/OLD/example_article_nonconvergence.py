import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

save_dir="../../Database/examples/tmp/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

image = np.array([[0, 0, 0, 0, 0],
                    [0, -1, 1, -1, 0],
                    [0, -1, -0.5, -1, 0],
                    [0, -1, -1, -1, 0],
                    [0, 0, 0, 0, 0]])

label = np.array([[0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 2, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]])

image=sp.ndimage.interpolation.zoom(image,zoom=4,order=0)
label=sp.ndimage.interpolation.zoom(label,zoom=4,order=0)
#print(img_lab)

t_model="B<A"
p_model="B>A"

id2r,matcher=skgti.core.recognize_regions(image,label,t_model,p_model,roi=None,verbose=True,bf=False)
#id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,verbose=True)

#skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
skgti.io.save_matcher_details(matcher,image,label,None,save_dir,True)

