import numpy as np
from skgtimage.core.photometry import region_stat,grey_levels

def rgb2chsv(image):
    return hsv2chsv(rgb2hsv(image))

def rgb2hsv(image):
    import matplotlib
    tmp_image=image.astype(np.float)
    if np.max(tmp_image) > 1.0 :
        tmp_image/=255.0
    return matplotlib.colors.rgb_to_hsv(tmp_image)

def rgb2gray(image):
    return 0.2125*image[:,:,0]+ 0.7154*image[:,:,1]+0.0721*image[:,:,2]

def hsv2chsv(data):
    """
    hsv -> s*v*cos(2pi*h),s*v*sin(2pi*h),v
    data can be a point (array [h,s,v])
    data can be rows of points
    data can be a 2D image h=image[:,:,0]; s=image[:,:,1], v=image[:,:,2]
    """
    #point
    if len(data.shape) == 1:
        result=np.zeros(data.shape,dtype=np.float)
        result[0]=data[1]*data[2]*np.cos(data[0]*2.0*np.pi)
        result[1]=data[1]*data[2]*np.sin(data[0]*2.0*np.pi)
        result[2]=data[2]
        return result
    #rows of points
    if len(data.shape) == 2:
        result=np.zeros(data.shape,dtype=np.float)
        result[:,0]=data[:,1]*data[:,2]*np.cos(data[:,0]*2.0*np.pi)
        result[:,1]=data[:,1]*data[:,2]*np.sin(data[:,0]*2.0*np.pi)
        result[:,2]=data[:,2]
        return result
    #image
    else:
        result=np.zeros(data.shape,dtype=np.float)
        result[:,:,0]=data[:,:,1]*data[:,:,2]*np.cos(data[:,:,0]*2.0*np.pi)
        result[:,:,1]=data[:,:,1]*data[:,:,2]*np.sin(data[:,:,0]*2.0*np.pi)
        result[:,:,2]=data[:,:,2]
        if type(data) == np.ma.MaskedArray :
            result=np.ma.masked_array(result,mask=data.mask)
        return result


def merge_photometry_color(image,label,roi,times,verbose=False):
    """
    Merge regions of similar photometry, even if there are not adjacent
    :param chsv_image:
    :param label:
    :param roi:
    :param times:
    :param verbose:
    :return:
    """
    ################
    #Precomputing properties
    chsv_image = rgb2chsv(image)
    levels=grey_levels(label,roi)
    nb_levels=len(levels)
    mean_for_each_level=[]
    size_for_each_level=[]
    for i in range(0,nb_levels):
        if roi is not None:
            roi_i = np.logical_and(np.where(label == levels[i], 1, 0),roi)
        else:
            roi_i = np.where(label == levels[i], 1, 0)
        #plt.imshow(roi_i,"gray");plt.show()
        ri = region_stat(chsv_image, roi_i, np.mean, mc=True)
        mean_for_each_level+=[ri]
        size_for_each_level+=[np.sum(roi_i)]

    ################
    #Computing adjacency matrix
    adj_matrix=np.ones((nb_levels,nb_levels)) #1 assumes to be higher than any distance
    for i in range(0,nb_levels):
        for j in range(0, nb_levels):
            if i < j:
                ri=mean_for_each_level[i]
                rj=mean_for_each_level[j]
                dist=np.sqrt((rj[0]-ri[0])**2+(rj[1]-ri[1])**2+(rj[2]-ri[2])**2)
                adj_matrix[i,j]=dist
                adj_matrix[j,i]=dist

    new_label=np.copy(label)
    for it in range(0,times):
        if verbose:
            print("Merge_photometry_color, remaining iterations:", times-it)
        #plt.imshow(new_label);plt.show()
        ################
        #Search minimal distance
        mini=np.min(adj_matrix)
        min_is,min_js=np.where(adj_matrix==mini)
        min_i=min_is[0]
        min_j=min_js[0]
        ################
        #Merging j+i -> i ; on vire j
        #Modification of label, adjency matrix, etc
        ################
        merging = (levels[min_i], levels[min_j])
        #Label j takes the label i
        roi_to_change=np.where(new_label==merging[1],1,0)
        new_label=np.ma.array(new_label, mask=roi_to_change).filled(merging[0])
        #Update
        #levels.pop(min_j)
        levels=np.delete(levels, min_j, 0)
        #nb_levels=len(levels)
        tmp_mean_i=mean_for_each_level[min_i]
        tmp_mean_j=mean_for_each_level[min_j]
        mean_for_each_level[min_i]=(size_for_each_level[min_i]*mean_for_each_level[min_i]+size_for_each_level[min_j]*mean_for_each_level[min_j])/(size_for_each_level[min_i]+size_for_each_level[min_j])
        mean_for_each_level.pop(min_j)
        size_for_each_level[min_i]=size_for_each_level[min_i]+size_for_each_level[min_j]
        size_for_each_level.pop(min_j)
        adj_matrix=np.delete(adj_matrix, min_j, 0)
        adj_matrix=np.delete(adj_matrix, min_j, 1)
        if min_i < min_j:
            new_index_min_i=min_i
        else:
            new_index_min_i=min_i-1
        r_min_i=mean_for_each_level[new_index_min_i]
        for n in range(0,len(levels)):
            if n != new_index_min_i:
                rn = mean_for_each_level[n]
                dist = np.sqrt((rn[0] - r_min_i[0]) ** 2 + (rn[1] - r_min_i[1]) ** 2 + (rn[2] - r_min_i[2]) ** 2)
                adj_matrix[new_index_min_i, n] = dist
                adj_matrix[n, new_index_min_i] = dist

    return new_label
