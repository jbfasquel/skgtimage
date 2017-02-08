import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

truth_dir="../../Database/image00/test07/truth/"
save_dir="../../Database/image00/test07/"


# IMAGE
image=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.1, 1.1, 1.1, 0.0, 0.0],
                [0.0, 0.5, 1.1, 1.1, 1.1, 0.0, 0.0],
                [0.0, 0.0, 1.1, 1.1, 1.1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

label=np.array([[0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 3, 3, 3, 1, 0],
                [0, 2, 3, 3, 3, 1, 0],
                [0, 1, 3, 3, 3, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0]])

roi=np.array([[0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0]])

t_graph,p_graph=skgti.core.from_labelled_image(image, label, roi, manage_bounds=True, thickness=1)
print("ROI:\n",roi)
for n in t_graph.nodes():
    print(n,":\n",t_graph.get_region(n))

quit()
