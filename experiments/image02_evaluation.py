import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper

######################################
#CONTEXT
######################################
root_dir="Database/image02/"
t_desc="tumor,vessel<liver"
p_desc="tumor<liver<vessel"

######################################
#MEANSHIFT
######################################
save_dir="Database/image02/meanshift_bandwidth15/"
image_gray=np.load(os.path.join(root_dir,"image_filtered.npy")) #;helper.plot(image_gray,45);quit()
#vessel=np.load(os.path.join(root_dir+"truth/","region_vessel.npy"));helper.plot(vessel,45);quit()
#tumors=np.load(os.path.join(root_dir+"truth/","region_tumor.npy"));helper.plot(tumors,45);quit()


######
# COMPARISON WITH TRUTH
classif,region2sim=helper.compared_with_truth(image_gray,t_desc,p_desc,root_dir+"truth",save_dir+"06_relabelled_built_t_graph",save_dir+"07_eval_classif/",slices=[45])
print("%%%%%%%%%%%%%%%%%%%%%%")
print("Evaluation of classification of all region vs truth")
print("Our approach vs truth (good classif rate): ",classif)
print("Our approach vs truth (similarities): ", region2sim)


'''
######################################
#TOP REGION : MEANSHIFT WITH FILTERING 4 CLASSES
######################################
save_dir="Database/image03/top_meanshift_ok_filtering_4classes_versus4expected/"
image_gray=sp.misc.imread(os.path.join(save_dir,"01_context_image.png")) #;plt.imshow(image_gray,"gray");plt.show();quit()

######
# COMPARISON WITH TRUTH
classif,region2sim=helper.compared_with_truth(image_gray,t_desc,p_desc,root_dir+"truth_top",save_dir+"06_relabelled_built_t_graph",save_dir+"07_eval_classif/")
print("%%%%%%%%%%%%%%%%%%%%%%")
print("Evaluation of classification of all region vs truth")
print("Our approach vs truth (good classif rate): ",classif)
print("Our approach vs truth (similarities): ", region2sim)
######
# COMPARISON WITH RAW SEGMENTATION
segmentation=sp.misc.imread(os.path.join(save_dir,"01_context_segmentation.png")) #plt.imshow(segmentation,"gray");plt.show();quit()
truth_t_graph,truth_p_graph=skgti.io.from_dir(t_desc,p_desc,image_gray,root_dir+"truth_top")
result_t_graph,result_p_graph=skgti.io.from_dir(t_desc,p_desc,image_gray,save_dir+"06_relabelled_built_t_graph")

related_truth=skgti.utils.combine(truth_t_graph,['A','B','C','G','D','E','H','F'],[170,85,170,85,0,85,0,255])
related_result=skgti.utils.combine(result_t_graph,['A','B','C','G','D','E','H','F'],[170,85,170,85,0,85,0,255])

classif_result,classif_rawsegmentation=helper.compared_with_rawsegmentation(segmentation,related_result,related_truth,truth_t_graph,save_dir+"07_eval_vs_raw_seg/")
print("%%%%%%%%%%%%%%%%%%%%%%")
print("Ability of removing segmentation imperfection")
print("Raw segmentation vs truth (good classif rate)",classif_rawsegmentation)
print("Our approach vs truth (good classif rate)",classif_result)
'''
######################################
#TOP REGION : MEANSHIFT WITHOUT FILTERING 4 CLASSES
######################################
'''
save_dir="Database/image03/top_meanshift_nok_misclassif_4classes_versus4expected/"
image_gray=sp.misc.imread(os.path.join(save_dir,"01_context_image.png")) #;plt.imshow(image_gray,"gray");plt.show();quit()

######
# COMPARISON WITH TRUTH
classif,region2sim=helper.compared_with_truth(image_gray,t_desc,p_desc,root_dir+"truth_top",save_dir+"06_relabelled_built_t_graph",save_dir+"07_eval_classif/")
print("%%%%%%%%%%%%%%%%%%%%%%")
print("Evaluation of classification of all region vs truth")
print("Our approach vs truth (good classif rate): ",classif)
print("Our approach vs truth (similarities): ", region2sim)
######
# COMPARISON WITH RAW SEGMENTATION
segmentation=sp.misc.imread(os.path.join(save_dir,"01_context_segmentation.png")) #plt.imshow(segmentation,"gray");plt.show();quit()
truth_t_graph,truth_p_graph=skgti.io.from_dir(t_desc,p_desc,image_gray,root_dir+"truth_top")
result_t_graph,result_p_graph=skgti.io.from_dir(t_desc,p_desc,image_gray,save_dir+"06_relabelled_built_t_graph")

related_truth=skgti.utils.combine(truth_t_graph,['A','B','C','G','D','E','H','F'],[170,85,170,85,0,85,0,255])
related_result=skgti.utils.combine(result_t_graph,['A','B','C','G','D','E','H','F'],[170,85,170,85,0,85,0,255])

classif_result,classif_rawsegmentation=helper.compared_with_rawsegmentation(segmentation,related_result,related_truth,truth_t_graph,save_dir+"07_eval_vs_raw_seg/")
print("%%%%%%%%%%%%%%%%%%%%%%")
print("Ability of removing segmentation imperfection")
print("Raw segmentation vs truth (good classif rate)",classif_rawsegmentation)
print("Our approach vs truth (good classif rate)",classif_result)
'''

######################################
#TOP REGION : MEANSHIFT 5 CLASSES
######################################
'''
save_dir="Database/image03/top_meanshift_ok_5classes_versus4expected/"
image_gray=sp.misc.imread(os.path.join(save_dir,"01_context_image.png")) #;plt.imshow(image_gray,"gray");plt.show();quit()

######
# COMPARISON WITH TRUTH
classif,region2sim=helper.compared_with_truth(image_gray,t_desc,p_desc,root_dir+"truth_top",save_dir+"06_relabelled_built_t_graph",save_dir+"07_eval_classif/")
print("%%%%%%%%%%%%%%%%%%%%%%")
print("Evaluation of classification of all region vs truth")
print("Our approach vs truth (good classif rate): ",classif)
print("Our approach vs truth (similarities): ", region2sim)
'''