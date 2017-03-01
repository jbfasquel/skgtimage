import os,sys; sys.path.append(os.path.abspath("../")) #for executation without having installed the package
import scipy as sp;from scipy.misc import imread,imsave
import skgtimage as skgti;from skgtimage.utils import meanshift,recognize
from skimage.segmentation import mark_boundaries



#A PRIORI KNOWLEDGE
inclusion="text<paper<file" #text is included within paper, itself included into file
photometry="text<file<paper" #text is darker than file, itself darker than paper

#INITIAL IMAGE
image=imread("image_gray.png")

#MEANSHIFT-BASED SEGMENTATION
print("Start segmentation...")
segmentation=meanshift(image, 10,verbose=True)
print("Segmentation finished")

#INTERPRETATION (PROPOSED METHOD)
print("Start recognition...")
id2region,r = recognize(image, segmentation, inclusion, photometry,bg=True)
print("Recognition finished")

#skgti.utils.save_recognizer_details(r,"save/")

#DISPLAY
import matplotlib.pyplot as plt
idplot=141
plt.subplot(idplot)
plt.imshow(mark_boundaries(image, segmentation));plt.title("Initial");plt.axis('off')
for id in id2region:
    idplot+=1
    plt.subplot(idplot);plt.imshow(id2region[id],"gray");plt.title(id);plt.axis('off')
plt.show()