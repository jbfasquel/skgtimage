#import os,sys; sys.path.append(os.path.abspath(os.path.pardir))
import os,sys; sys.path.append(os.path.abspath(os.path.curdir))
import scipy as sp;from scipy.misc import imread,imsave
import skgtimage as skgti;from skgtimage.utils import mean_shift,recognize
from skimage.segmentation import mark_boundaries

#A PRIORI KNOWLEDGE
inclusion="text<paper<file"
photometry="text<file<paper"

#IMAGE INITIALE
image=imread("image.png")

#MEANSHIFT-BASED SEGMENTATION
segmentation=mean_shift(image,10)

#INTERPRETATION (PROPOSED METHOD)
id2region,r = recognize(image, segmentation, inclusion, photometry,bg=True)

skgti.utils.save_recognizer_details(r,"save/")

#DISPLAY
import matplotlib.pyplot as plt
idplot=141
plt.subplot(idplot)
plt.imshow(mark_boundaries(image, segmentation));plt.title("Initial");plt.axis('off')
for id in id2region:
    idplot+=1
    plt.subplot(idplot);plt.imshow(id2region[id],"gray");plt.title(id);plt.axis('off')
plt.show()