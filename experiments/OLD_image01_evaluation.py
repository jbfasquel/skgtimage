import os
import numpy as np
import scipy as sp;from scipy import misc; from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti


root_name="image01"
dir="Database/image01/"
#save_dir="Database/image01/Evaluation/"


def eval(roi,labelled_image,graph_result,save_dir):
    ########
    # L_TRUTH
    #########
    truth_regions=skgti.utils.Regions()
    truth_regions.add(2,sp.misc.imread(os.path.join(dir,root_name+"_region2.png")))
    truth_regions.add(3,sp.misc.imread(os.path.join(dir,root_name+"_region3.png")))
    truth_regions.add(4,sp.misc.imread(os.path.join(dir,root_name+"_region4.png")))

    truth=truth_regions.combine([2,3,4],[150,255,50])
    l_truth=np.ma.array(truth, mask=np.logical_not(roi))
    #plt.imshow(truth,"gray");plt.show()


    ########
    # EVALUATION: APRES IDENTIFICATION
    #########
    graph_based_regions=skgti.utils.Regions()
    graph_based_regions.add(2,graph_result.get_region("file"))
    graph_based_regions.add(3,graph_result.get_region("paper"))
    graph_based_regions.add(4,graph_result.get_region("text"))
    graph_based_segmentation=graph_based_regions.combine([2,3,4],[150,255,50])
    graph_based_segmentation=np.ma.array(graph_based_segmentation, mask=np.logical_not(roi))

    classif,levels=skgti.utils.goodclassification_rate_unknown_label_ordering(graph_based_segmentation,l_truth)
    graph_based_segmentation=skgti.utils.remap_greylevels(graph_based_segmentation,levels)
    similarities=skgti.utils.similarity_indices(graph_based_segmentation,l_truth)
    print("GCR (%) : " , round(classif*100,3))
    print("Similarites : " , [round(i,3) for i in similarities])
    skgti.utils.save_to_csv(save_dir,'result_graph',round(classif*100,3),[round(i,3) for i in similarities])

    #########
    # EVALUATION : SANS IDENTIFICATION
    #########

    classif,levels=skgti.utils.goodclassification_rate_unknown_label_ordering(labelled_image,l_truth)
    tmp_labelled_image=skgti.utils.remap_greylevels(labelled_image,levels)
    similarities=skgti.utils.similarity_indices(tmp_labelled_image,l_truth)
    print("GCR (%) : " , round(classif*100,3))
    print("Similarites : " , [round(i,3) for i in similarities])
    skgti.utils.save_to_csv(save_dir,'result_standard',round(classif*100,3),[round(i,3) for i in similarities])
    sp.misc.imsave(os.path.join(save_dir,"result_standard.png"),(tmp_labelled_image.filled(0)).astype(np.uint8))





'''

#########
# KNOWLEDGE
#########
tp_model=skgti.core.TPModel()
tp_model.set_topology("text<paper<file;pen;clear_file")
tp_model.add_photometry("pen<clear_file<paper;pen=text;file=clear_file")
image=sp.misc.imread(os.path.join(dir,root_name+".png"))
#image=sp.ndimage.filters.median_filter(image, 5)
tp_model.set_image(image)

#########
# CONTEXT
#########
tp_model.set_region("pen",sp.misc.imread(os.path.join(dir,root_name+"_region0.png")))
tp_model.set_region("clear_file",sp.misc.imread(os.path.join(dir,root_name+"_region1.png")))
tp_model.set_region("file",sp.misc.imread(os.path.join(dir,root_name+"_region2.png")))

# PLOT
#skgti.io.plot_model(tp_model);plt.show()

#########
# SEGMENTATION
#########
tp_model.set_targets(['text','paper'])
labelled_image=sp.misc.imread(os.path.join(save_dir,root_name+"_kmeans_segmentation.png"))
roi=sp.misc.imread(os.path.join(dir,root_name+"_region2.png"))

labelled_image=skgti.core.manage_boundaries(labelled_image,roi)
labelled_image=np.ma.array(labelled_image, mask=np.logical_not(roi))
#plt.imshow(labelled_image,"gray");plt.show()


#########
# IDENTIFICATION
#########
# REFERENCE SUBGRAPHS
l=skgti.core.classes_for_targets(tp_model.t_graph,['text','paper'])
target2residues=skgti.core.identify_from_labels(image,labelled_image,tp_model.t_graph,tp_model.p_graphs[0],l)


for n in target2residues: tp_model.set_region(n,skgti.core.fill_region(target2residues[n]))


'''