import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

######################################
#DOWNLOAD FULLRESOLUTION -> DIVIDED BY 3
######################################
#############
#For general crop: dilated union of ROIs
#############
'''
input_bottom="Database/image03/FullResolution/Truth_bottom/"
input_top="Database/image03/FullResolution/Truth_top/"
ref_roi=sp.misc.imread(os.path.join(input_top,"region_A.png"))+sp.misc.imread(os.path.join(input_bottom,"region_A.png"))
#Required dilatation so that rois do not touch image border (for eventual further preprocessing)
ref_roi=255*sp.ndimage.morphology.binary_dilation(ref_roi,iterations=1).astype(np.uint8)


download=3
root_name="image03"
input="Database/image03/FullResolution/"
output="Database/image03/"
image=sp.misc.imread(os.path.join(input,root_name+".jpeg"))
image=skgti.utils.extract_subarray_rgb(image,ref_roi)
image=image[::download,::download,:]
sp.misc.imsave(os.path.join(output,root_name+".png"),image)
#Bottom
input="Database/image03/FullResolution/Truth_bottom/"
output="Database/image03/truth_bottom/"
for i in ['A','B','C','D','E','F','G','H','I']:
    image=sp.misc.imread(os.path.join(input,"region_"+i+".png"))
    image=skgti.utils.extract_subarray(image,ref_roi)
    image=image[::download,::download]
    sp.misc.imsave(os.path.join(output,"region_"+i+".png"),image)
#Top
input="Database/image03/FullResolution/Truth_top/"
output="Database/image03/truth_top/"
for i in ['A','B','C','D','E','F','G','H']:
    image=sp.misc.imread(os.path.join(input,"region_"+i+".png"))
    image=skgti.utils.extract_subarray(image,ref_roi)
    image=image[::download,::download]
    sp.misc.imsave(os.path.join(output,"region_"+i+".png"),image)
'''
######################################
#PREPARE GROUNDTRUTH TOP
######################################
'''
root_name="image03"
input="Database/image03/FullResolution/RawData"
output="Database/image03/FullResolution/Truth_top"
#REGION A
i_region=sp.misc.imread(os.path.join(input,root_name+"_region0.png"))
sp.misc.imsave(os.path.join(output,"region_A.png"),i_region)
#REGION B
i_region=sp.misc.imread(os.path.join(input,root_name+"_region8.png"))
i_region=255*skgti.core.fill_region(i_region).astype(np.uint8)
print(np.max(i_region))
sp.misc.imsave(os.path.join(output,"region_B.png"),i_region)

#REGION C,G
i_region=sp.misc.imread(os.path.join(input,root_name+"_region9.png"))
print(np.max(i_region))
o_region=255*skgti.core.fill_region(i_region).astype(np.uint8)
print(np.max(o_region))
sp.misc.imsave(os.path.join(output,"region_C.png"),o_region)
diff=o_region-i_region
sp.misc.imsave(os.path.join(output,"region_G.png"),diff)
#REGION D
i_region=sp.misc.imread(os.path.join(input,root_name+"_region10.png"))
sp.misc.imsave(os.path.join(output,"region_D.png"),i_region)
#REGION F
i_region=sp.misc.imread(os.path.join(input,root_name+"_region11.png"))
sp.misc.imsave(os.path.join(output,"region_F.png"),i_region)
#REGION E,H
i_region=sp.misc.imread(os.path.join(input,root_name+"_region12.png"))
print(np.max(i_region))
o_region=255*skgti.core.fill_region(i_region).astype(np.uint8)
print(np.max(o_region))
sp.misc.imsave(os.path.join(output,"region_E.png"),o_region)
diff=o_region-i_region
sp.misc.imsave(os.path.join(output,"region_H.png"),diff)
'''

######################################
#PREPARE GROUNDTRUTH BOTTOM
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
######################################
#PREPARE GROUNDTRUTH TOP: FILLED REGIONS -> RESIDUES
######################################
'''
input="Database/image03/truth_top_filled/"
output="Database/image03/truth_top/"
A=sp.misc.imread(os.path.join(input,"region_A.png"))
B=sp.misc.imread(os.path.join(input,"region_B.png"))
new=A-B
sp.misc.imsave(os.path.join(output,"region_A.png"),new)
C=sp.misc.imread(os.path.join(input,"region_C.png"))
D=sp.misc.imread(os.path.join(input,"region_D.png"))
new=B-C-D
sp.misc.imsave(os.path.join(output,"region_B.png"),new)
G=sp.misc.imread(os.path.join(input,"region_G.png"))
new=C-G
sp.misc.imsave(os.path.join(output,"region_C.png"),new)
sp.misc.imsave(os.path.join(output,"region_G.png"),G)
E=sp.misc.imread(os.path.join(input,"region_E.png"))
F=sp.misc.imread(os.path.join(input,"region_F.png"))
H=sp.misc.imread(os.path.join(input,"region_H.png"))
new=D-E-F
sp.misc.imsave(os.path.join(output,"region_D.png"),new)
new=E-H
sp.misc.imsave(os.path.join(output,"region_E.png"),new)
sp.misc.imsave(os.path.join(output,"region_F.png"),F)
sp.misc.imsave(os.path.join(output,"region_H.png"),H)
'''
######################################
#PREPARE GROUNDTRUTH BOTTOM: FILLED REGIONS -> RESIDUES
######################################
'''
input="Database/image03/truth_bottom_filled/"
output="Database/image03/truth_bottom/"
A=sp.misc.imread(os.path.join(input,"region_A.png"))
B=sp.misc.imread(os.path.join(input,"region_B.png"))
new=A-B
sp.misc.imsave(os.path.join(output,"region_A.png"),new)
C=sp.misc.imread(os.path.join(input,"region_C.png"))
new=B-C
sp.misc.imsave(os.path.join(output,"region_B.png"),new)

D=sp.misc.imread(os.path.join(input,"region_D.png"))
H=sp.misc.imread(os.path.join(input,"region_H.png"))
F=sp.misc.imread(os.path.join(input,"region_F.png"))
I=sp.misc.imread(os.path.join(input,"region_I.png"))
new=C-D-H-F-I
sp.misc.imsave(os.path.join(output,"region_C.png"),new)

E=sp.misc.imread(os.path.join(input,"region_E.png"))
new=D-E
sp.misc.imsave(os.path.join(output,"region_D.png"),new)

G=sp.misc.imread(os.path.join(input,"region_G.png"))
new=F-G
sp.misc.imsave(os.path.join(output,"region_F.png"),new)
'''
##########
#histogram top
truth_dir="Database/image03/truth_top/"
t_desc="C,D<B<A;E,F<D;G<C;H<E"
p_desc="G=B<E=F<H=D<A=C"
image_rgb=sp.misc.imread(os.path.join(truth_dir,"image.png"))
image=skgti.utils.rgb2gray(image_rgb)
t_graph,p_graph=skgti.io.from_dir(t_desc,p_desc,image,truth_dir)
#plt.imshow(p_graph.get_image());plt.show()
skgti.io.plot_graph_histogram(t_graph,p_graph,True)#;plt.show()
plt.savefig(truth_dir+"histograms.svg",format="svg",bbox_inches='tight')
plt.savefig(truth_dir+"histograms.png",format="png",bbox_inches='tight')
plt.gcf().clear()

##########
#histogram bottom
truth_dir="Database/image03/truth_bottom/"
t_desc="E<D;G<F;D,F,H,I<C<B<A"
p_desc="B=F<D=H<I=E<C=A=G"
image_rgb=sp.misc.imread(os.path.join(truth_dir,"image.png"))
image=skgti.utils.rgb2gray(image_rgb)
t_graph,p_graph=skgti.io.from_dir(t_desc,p_desc,image,truth_dir)
#plt.imshow(p_graph.get_image());plt.show()
skgti.io.plot_graph_histogram(t_graph,p_graph,True)#;plt.show()
plt.savefig(truth_dir+"histograms.svg",format="svg",bbox_inches='tight')
plt.savefig(truth_dir+"histograms.png",format="png",bbox_inches='tight')
plt.gcf().clear()

