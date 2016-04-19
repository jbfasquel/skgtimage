import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

def composition_operator(input_truth_dir,names,operators):
    image=sp.misc.imread(os.path.join(input_truth_dir,"region_"+names[0]+".png"))
    for i in range(1,len(names)):
        tmp=sp.misc.imread(os.path.join(input_truth_dir,"region_"+names[i]+".png"))
        if operators[i] == '-': image-=tmp
        else: image+=tmp
    return image

######################################
#EVALUATION PAR COMPARAISON RELATIVE: CAS SEGMENTATION 5 classes
######################################
save_dir="Database/image03/meanshift_down3_test1/"
roi=sp.misc.imread(os.path.join(save_dir,"01_roi.png"))

#RAW
'''
labelled_image=sp.misc.imread(os.path.join(save_dir,"01_labelled_meanshift.png"))
roi=sp.misc.imread(os.path.join(save_dir,"01_roi.png"))
l_labelled_image=np.ma.array(labelled_image, mask=np.logical_not(roi))
residues=skgti.core.residues_from_labels(l_labelled_image)
tmp_dir=save_dir+"residues_after_meanshift"
for i in range(0,len(residues)):    sp.misc.imsave(os.path.join(tmp_dir,"label"+str(i)+".png"),255*residues[i].astype(np.uint8))
truth_regions=skgti.utils.Regions()
for i in range(0,5):
    truth_regions.add(i,sp.misc.imread(os.path.join(tmp_dir,"label"+str(i)+".png")))
truth=truth_regions.combine(range(0,5),range(1,6))
l_truth=np.ma.array(truth, mask=np.logical_not(roi))
sp.misc.imsave(os.path.join(tmp_dir,"all.png"),l_truth.filled(0))
'''
#TRUTH
'''
downsampling=3
input_truth_dir="Database/image03/Truth"
tmp_dir=save_dir+"residues_from_truth"
image=composition_operator(input_truth_dir,['A','B','C','D','F','G','H','I'],['+','-','+','-','-','+','-','-'])
image=image[740:1360,150:1700];image=image[::downsampling,::downsampling]
sp.misc.imsave(os.path.join(tmp_dir,"label"+str(0)+".png"),image)
image=composition_operator(input_truth_dir,['B','C','F','G'],['+','-','+','-'])
image=image[740:1360,150:1700];image=image[::downsampling,::downsampling]
sp.misc.imsave(os.path.join(tmp_dir,"label"+str(1)+".png"),image)
image=composition_operator(input_truth_dir,['D','E'],['+','-'])
image=image[740:1360,150:1700];image=image[::downsampling,::downsampling]
sp.misc.imsave(os.path.join(tmp_dir,"label"+str(2)+".png"),image)
image=composition_operator(input_truth_dir,['E','I'],['+','+'])
image=image[740:1360,150:1700];image=image[::downsampling,::downsampling]
sp.misc.imsave(os.path.join(tmp_dir,"label"+str(3)+".png"),image)
image=composition_operator(input_truth_dir,['H'],['+'])
image=image[740:1360,150:1700];image=image[::downsampling,::downsampling]
sp.misc.imsave(os.path.join(tmp_dir,"label"+str(4)+".png"),image)
truth_regions=skgti.utils.Regions()
for i in range(0,5):
    truth_regions.add(i,sp.misc.imread(os.path.join(tmp_dir,"label"+str(i)+".png")))
truth=truth_regions.combine(range(0,5),range(1,6))
l_truth=np.ma.array(truth, mask=np.logical_not(roi))
sp.misc.imsave(os.path.join(tmp_dir,"all.png"),l_truth.filled(0))
#truth=truth_regions.combine(range(0,5),[0,40,80,150,200])
plt.imshow(truth,"gray");plt.show()

plt.imshow(l_truth.filled(-1),"gray");plt.show()
'''


#RESULT
'''
input_truth_dir=save_dir+"final_t_graph"
tmp_dir=save_dir+"residues_from_graph"
image=composition_operator(input_truth_dir,['A','B','C','D','F','G','H','I'],['+','-','+','-','-','+','-','-'])
sp.misc.imsave(os.path.join(tmp_dir,"label"+str(0)+".png"),image)
image=composition_operator(input_truth_dir,['B','C','F','G'],['+','-','+','-'])
sp.misc.imsave(os.path.join(tmp_dir,"label"+str(1)+".png"),image)
image=composition_operator(input_truth_dir,['D','E'],['+','-'])
sp.misc.imsave(os.path.join(tmp_dir,"label"+str(2)+".png"),image)
image=composition_operator(input_truth_dir,['E','I'],['+','+'])
sp.misc.imsave(os.path.join(tmp_dir,"label"+str(3)+".png"),image)
image=composition_operator(input_truth_dir,['H'],['+'])
sp.misc.imsave(os.path.join(tmp_dir,"label"+str(4)+".png"),image)
truth_regions=skgti.utils.Regions()
for i in range(0,5):
    truth_regions.add(i,sp.misc.imread(os.path.join(tmp_dir,"label"+str(i)+".png")))
truth=truth_regions.combine(range(0,5),range(1,6))
l_truth=np.ma.array(truth, mask=np.logical_not(roi))
sp.misc.imsave(os.path.join(tmp_dir,"all.png"),l_truth.filled(0))
#truth=truth_regions.combine(range(0,5),[0,40,80,150,200])
#plt.imshow(truth,"gray");plt.show()
#plt.imshow(l_truth.filled(-1),"gray");plt.show()
'''

'''
#EVALUATION: TRUTH
tmp_dir=save_dir+"residues_from_truth"
truth=sp.misc.imread(os.path.join(tmp_dir,"all.png"))
l_truth=np.ma.array(truth, mask=np.logical_not(roi))


#EVALUATION: RAW VS TRUTH
tmp_dir=save_dir+"residues_after_meanshift"
raw=sp.misc.imread(os.path.join(tmp_dir,"all.png"))
l_raw=np.ma.array(raw, mask=np.logical_not(roi))
classif,levels=skgti.utils.goodclassification_rate_unknown_label_ordering(l_raw,l_truth)
graph_based_segmentation=skgti.utils.remap_greylevels(l_raw,levels)
similarities=skgti.utils.similarity_indices(l_raw,l_truth)
print("GCR (%) : " , round(classif*100,3))
print("Similarites : " , [round(i,3) for i in similarities])
skgti.utils.save_to_csv(save_dir,'result_raw',round(classif*100,3),[round(i,3) for i in similarities])

#EVALUATION: GRAPH VS TRUTH
tmp_dir=save_dir+"residues_from_graph"
graph=sp.misc.imread(os.path.join(tmp_dir,"all.png"))
l_graph=np.ma.array(graph, mask=np.logical_not(roi))
classif,levels=skgti.utils.goodclassification_rate_unknown_label_ordering(l_graph,l_truth)
graph_based_segmentation=skgti.utils.remap_greylevels(l_graph,levels)
similarities=skgti.utils.similarity_indices(l_graph,l_truth)
print("GCR (%) : " , round(classif*100,3))
print("Similarites : " , [round(i,3) for i in similarities])
skgti.utils.save_to_csv(save_dir,'result_graph',round(classif*100,3),[round(i,3) for i in similarities])
'''

######################################
#EVALUATION PAR COMPARAISON ABSOLUE: CAS SEGMENTATION 5 classes
######################################
####
#PREPARING RESULT
save_dir="Database/image03/meanshift_down3_test1/"
roi=sp.misc.imread(os.path.join(save_dir,"01_roi.png"))
input_result_dir=save_dir+"final_t_graph"
regions=skgti.utils.Regions()
ids=['A','B','C','D','H','F','I','E','G']
for id in ids:
    regions.add(id,sp.misc.imread(os.path.join(input_result_dir,"region_"+id+".png")))
result=regions.combine(ids,range(0+5,len(ids)+5))
l_result=np.ma.array(result, mask=np.logical_not(roi))
sp.misc.imsave(os.path.join(input_result_dir,"all.png"),l_result.filled(0))
#plt.imshow(l_result.filled(0),"gray");plt.show()

####
#PREPARING TRUTH
save_dir="Database/image03/meanshift_down3_test1/"
roi=sp.misc.imread(os.path.join(save_dir,"01_roi.png"))
input_truth_dir="Database/image03/Truth"
regions=skgti.utils.Regions()
ids=['A','B','C','D','H','F','I','E','G']
for id in ids:
    regions.add(id,sp.misc.imread(os.path.join(input_truth_dir,"region_"+id+".png")))
truth=regions.combine(ids,range(0+5,len(ids)+5))
downsampling=3
truth=truth[740:1360,150:1700];truth=truth[::downsampling,::downsampling]
l_truth=np.ma.array(truth, mask=np.logical_not(roi))
sp.misc.imsave(os.path.join(input_truth_dir,"all.png"),l_truth.filled(0))
#plt.imshow(l_truth.filled(0),"gray");plt.show()

####
#EVALUATION: GRAPH VS TRUTH
image_of_errors=np.where(l_result-l_truth!=0,255,0)
sp.misc.imsave(os.path.join(input_result_dir,"errors.png"),image_of_errors)
plt.imshow(image_of_errors,"gray");plt.show()
number_of_errors=np.count_nonzero(l_result-l_truth)
classif=1.0-np.count_nonzero(l_result-l_truth)/np.count_nonzero(roi)
similarities=skgti.utils.similarity_indices(l_result,l_truth)
print("GCR (%) : " , round(classif*100,3))
print("Similarites : " , [round(i,3) for i in similarities])
skgti.utils.save_to_csv(save_dir,'result_graph_abs',round(classif*100,3),[round(i,3) for i in similarities])


######################################
#PREPARE GROUNDTRUTH
######################################
'''
root_name="image03"
input="Database/image03/RawData"
output="Database/image03/RawTruth"

#REGION A
i_region=sp.misc.imread(os.path.join(input,root_name+"_region1.png"))
sp.misc.imsave(os.path.join(output,"regionA.png"),i_region)
#REGION B
i_region=sp.misc.imread(os.path.join(input,root_name+"_region2.png"))
i_region=255*skgti.core.fill_region(i_region).astype(np.uint8)
print(np.max(i_region))
sp.misc.imsave(os.path.join(output,"regionB.png"),i_region)
#REGION C
a=sp.misc.imread(os.path.join(output,"regionA.png"))
b=sp.misc.imread(os.path.join(output,"regionB.png"))
tmp_a=np.logical_and(a,b)
i_region=sp.misc.imread(os.path.join(input,root_name+"_region2.png"))/255
o_region=skgti.core.fill_region(tmp_a-i_region)
sp.misc.imsave(os.path.join(output,"regionC.png"),o_region.astype(np.uint8)*255)
#REGIONS F ET G
tmp=sp.misc.imread(os.path.join(output,"tmp.png"))/255
F=skgti.core.fill_region(tmp)
G=F-tmp
sp.misc.imsave(os.path.join(output,"regionF.png"),F.astype(np.uint8)*255)
sp.misc.imsave(os.path.join(output,"regionG.png"),G.astype(np.uint8)*255)
#REGIONS H ET I
i_region=sp.misc.imread(os.path.join(input,root_name+"_region4.png"))
sp.misc.imsave(os.path.join(output,"regionH.png"),i_region)
i_region=sp.misc.imread(os.path.join(input,root_name+"_region5.png"))
sp.misc.imsave(os.path.join(output,"regionI.png"),i_region)

#REGIONS D ET E
i_region=sp.misc.imread(os.path.join(input,root_name+"_region6.png"))
sp.misc.imsave(os.path.join(output,"regionD.png"),i_region)
i_region=sp.misc.imread(os.path.join(input,root_name+"_region7.png"))
sp.misc.imsave(os.path.join(output,"regionE.png"),i_region)

#plt.imshow(F,"gray");plt.show()
'''