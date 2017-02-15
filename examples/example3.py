import os,sys; sys.path.append(os.path.abspath("../")) #for executation without having installed the package
import scipy as sp;from scipy.misc import imread,imsave
import skgtimage as skgti;from skgtimage.utils import meanshift,recognize
from skimage.segmentation import mark_boundaries

#A PRIORI KNOWLEDGE
inclusion="2E<2D;2G<F;2D,F,2H,2I<C<B<A;C,1D<B<A;1E,1G<1D;1H<1E"
photometry="B=F=2D=2H=1E=1G<2I=2E=1H=1D<C=2G=A"

#INITIAL IMAGE
image=imread("image_color.png")

#MEANSHIFT-BASED SEGMENTATION
print("Start segmentation...")
segmentation=meanshift(image, 0.1, mc=True, sigma=0.5, rgb_convert=True)
print("Segmentation finished")

#INTERPRETATION (PROPOSED METHOD)
print("Start recognition...")
id2region,r = recognize(image, segmentation, inclusion, photometry,mc=True,bg=True)
print("Recognition finished")

#skgti.utils.save_recognizer_details(r,"save/")

#DISPLAY
import matplotlib.pyplot as plt
idplot=141
plt.subplot(idplot)
plt.imshow(mark_boundaries(image, segmentation));plt.title("Initial");plt.axis('off')
for id in ["1E","2E","F"]:
    idplot+=1
    plt.subplot(idplot);plt.imshow(id2region[id],"gray");plt.title(id);plt.axis('off')
plt.show()