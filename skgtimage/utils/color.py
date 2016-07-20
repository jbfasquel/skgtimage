import numpy as np


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
