import os,time
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti
import skimage; from skimage import filters
import helper

#########
# MISC INFORMATIONS
#########
root_save_dir="../../Database/image03/results_allposes_tmp7/";
if not os.path.exists(root_save_dir): os.mkdir(root_save_dir)

##########################################################################################
##########################################################################################
#
#              ROI NONE
#
##########################################################################################
##########################################################################################
t_desc = "2E<2D;2G<F;2D,F,2H,2I<C<B<A;C,1D<B<A;1E,1G<1D;1H<1E"
p_desc = "B=F=2D=2H=1E=1G<2I=2E=1H=1D<C=2G=A"

for pose_number in [7]:
    truth_dir="../../Database/image03/pose_"+str(pose_number)+"/downsampled3/truth_simplified/"
    image=sp.misc.imread(os.path.join(truth_dir,"image.png"))
    roi = skgti.core.fill_region(sp.misc.imread(os.path.join(truth_dir, "region_A.png")))
    print("STARTING POSE ", pose_number)
    param=0.09

    info="3-Pose"+str(pose_number)+"_ROI_1A_QS_"+str(param)+"/"
    #label = skgti.utils.quickshift(image, ratio=0.9, mc=True, roi=roi)
    label = skgti.utils.mean_shift(image,param,roi,True,True,sigma=0.5,rgb_convert=True)
    tmp = skimage.segmentation.mark_boundaries(image, label)
    sp.misc.imsave(root_save_dir+"QS0.png",tmp)
    param=5+++0
    label=skgti.utils.rag_merge(image, label, param, True, roi)
    tmp = skimage.segmentation.mark_boundaries(image, label)
    sp.misc.imsave(root_save_dir+"QS1_rag"+str(param)+".png",tmp)
    param2=0
    label = skgti.utils.merge_photometry_color(image, label, roi, param2)
    tmp = skimage.segmentation.mark_boundaries(image, label)
    sp.misc.imsave(root_save_dir + "QS2_rag" + str(param) + "+"+str(param2)+".png", tmp)
    #tmp = skimage.segmentation.mark_boundaries(image, label)
    nb = len(skgti.utils.grey_levels(label))
    print(nb)
    plt.imshow(tmp);plt.show();quit()

    try:
        id2region, recognizer = skgti.utils.recognize(image, label, t_desc, p_desc, mc=True, roi=roi, bg=True,prerag=75,premnoragmerging=60, verbose=True)
        result = helper.analyze(recognizer, root_save_dir + info, name=info, runtime=rt, truth_dir=truth_dir,iso_influence=False, slices=[], full=True)
        print("RESULT FOR POSE ",pose_number)
        for i in result: print(i, ":", result[i], "\n")
    except skgti.utils.RecognitionException as m_e:
        print('ERROR:', m_e.message)
        skgti.utils.save_recognizer_details(m_e.recognizer, root_save_dir + "Failed_"+info, full=True, slices=[])
##########################################################################################
##########################################################################################
#
#              CONCATENATE ALL
#
##########################################################################################
##########################################################################################
helper.concatenate(root_save_dir,"stats.csv")
helper.concatenate(root_save_dir,"stats_runtime.csv")
