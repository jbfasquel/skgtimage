import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skgtimage as skgti

# IMAGE
c0=np.array([0,0,0]) #black
c1=np.array([1,1,1]) #white
c2=np.array([1,1,0]) #yellow 0.9 to avoid many digits in HSV
c3=np.array([0,1,1]) #green-blue
c4=np.array([1,0,0]) #red
c5=np.array([0,0,0]) #black
image=np.array([ [c0, c0, c0, c0, c0, c0, c0],
                 [c0, c1, c1, c1, c0, c3, c0],
                 [c0, c1, c2, c1, c0, c3, c0],
                 [c0, c1, c2, c1, c0, c0, c0],
                 [c0, c1, c5, c1, c0, c4, c0],
                 [c0, c1, c1, c1, c0, c4, c0],
                 [c0, c0, c0, c0, c0, c0, c0]])

regionC1=np.array([ [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]])

regionC3=np.array([ [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]])

regionC4=np.array([ [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0]])

#image_hsv=matplotlib.colors.rgb_to_hsv(image) #foreground=np.where(image_hsv[:,:,2]==0,0,1)
image_hsv=skgti.utils.rgb2hsv(image)
print(image_hsv[:,:,0]);print(image_hsv[:,:,1]);print(image_hsv[:,:,2])#;quit()
#plt.imshow(image,interpolation='nearest');plt.show();plt.axis('off');quit()
#plt.imshow(image_hsv,interpolation='nearest');plt.show();plt.axis('off');plt.axis('off');quit()

############
#  KNOWLEDGE
############
tp_model=skgti.core.TPModel()
tp_model.set_topology("c2,c5<c1<c0;c3,c4<c0")
tp_model.set_photometry(["c4<c2<c3;c1;c0=c5","c0=c5;c1<c2=c3=c4","c0,c5<c1=c2=c3=c4;c0=c5"]) #H,S,V, avoiding cycles (confusing with brothers)
tp_model.set_image(image_hsv)

############
#SEGMENTATION
############
# CONTEXT
tp_model.set_region('c0',np.ones(image_hsv.shape[0:2]))
tp_model.set_region('c1',regionC1)
tp_model.set_region('c3',regionC3)
tp_model.set_region('c4',regionC4)
tp_model.set_targets(['c2'])


r=skgti.core.find_equals(tp_model.t_graph,tp_model.p_graphs[0],'c2')

# PARAMETERS
l_image=tp_model.roi_image() #;print(l_image[:,:,0]);print(l_image[:,:,1]);print(l_image[:,:,2])
nb=tp_model.number_of_clusters() #;print(nb)
intervals=tp_model.intervals_for_clusters() ;print("Intervals:",intervals)
# KMEANS SEGMENTATION
seeder=skgti.utils.KMeansSeeder(l_image,nb,intervals,mc=True,projection_functor=skgti.utils.hsv2chsv)
for i in range(0,20):
    centroids=seeder.generate()

labelled_image=skgti.utils.kmeans(l_image,nb,intervals=intervals,fct=skgti.utils.hsv2chsv,mc=True) #;print(labelled_image);quit()

############
# RECOGNITION
############
tp_model.identify_from_labels(labelled_image)

############
# PLOT
############
segmentations=[np.where(labelled_image+1==i,1,0) for i in range(np.min(labelled_image)+1,np.max(labelled_image)+2)]
skgti.io.plot_model(tp_model,segmentations)
plt.show();quit()
