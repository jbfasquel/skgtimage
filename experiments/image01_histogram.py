import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper

truth_dir="Database/image01/truth/"

t_desc="text<paper<file"
p_desc="text<file<paper"

image=sp.misc.imread(truth_dir+"image.png")
roi=sp.misc.imread(truth_dir+"roi.png")

t_graph,p_graph=skgti.io.from_dir(t_desc,p_desc,image,truth_dir)
#plt.imshow(p_graph.get_image());plt.show()
skgti.io.plot_graph_histogram(t_graph,p_graph,True)#;plt.show()
plt.savefig(truth_dir+"histograms.svg",format="svg",bbox_inches='tight')
plt.savefig(truth_dir+"histograms.png",format="png",bbox_inches='tight')
