# *-* coding: iso-8859-1 *-*

import numpy as np
import math

def draw_ellipse(image,center,a,b,theta,value=200):
    angle=(theta * 2*np.pi)/360.0
    for x in range(0,image.shape[0]):
        for y in range(0,image.shape[1]):
            c1=((x-center[0])*math.cos(angle)+(y-center[1])*math.sin(angle))**2/a**2
            c2=((x-center[0])*math.sin(angle)-(y-center[1])*math.cos(angle))**2/b**2
            if c1+c2 <= 1:
                image[x,y]=value
def draw_cercle(image,center,r,value=200):
    draw_ellipse(image,center=center,a=r,b=r,theta=0,value=value)

def draw_sphere(image,center,r,value=200):
    for x in range(0,image.shape[0]):
        for y in range(0,image.shape[1]):
            for z in range(0,image.shape[2]):
                d=np.sqrt((x-center[0])**2+(y-center[1])**2+(z-center[2])**2)
                if d <= r : image[x,y,z]=value


def draw_square(image,center,size,value):
    return draw_rectangle(image,center,(size,size),value)

def draw_rectangle(image,center,size,value):
    x_size_tmp=int(np.round(size[0]/2.0))
    y_size_tmp=int(np.round(size[1]/2.0))
    roi=np.zeros((image.shape[0],image.shape[1]),np.bool)
    roi[center[0]-x_size_tmp:center[0]+x_size_tmp+1,center[1]-y_size_tmp:center[1]+y_size_tmp+1]=True
    indices=np.nonzero(roi)
    image[indices]=value
    return roi

def draw_gaussian(image,c_x=5,c_y=5,sigma_x=2.0,sigma_y=2.0,max=10,order=2,angle=0):
    """
    angle in degre
    """
    rad=(angle*np.pi)/180
    sigma_x=float(sigma_x)
    sigma_y=float(sigma_y)
    for x in range(0,image.shape[0]):
        for y in range(0,image.shape[1]):
            x_p=(x-c_x)*np.cos(rad)-(y-c_y)*np.sin(rad)
            y_p=(x-c_x)*np.sin(rad)+(y-c_y)*np.cos(rad)
            image[x,y]+=max*np.exp(-(((x_p/sigma_x)**2+(y_p/sigma_y)**2)**(order/2.0))/2.0)
            #image[x,y]+=max*np.exp(-(((x-c_x)*np.cos(rad)/sigma_x)**order+((y-c_y)*np.cos(rad)/sigma_y)**order)/2.0)
    #return amplitude_model


def create_image(size=(200,200),value=0,dtype=np.int32):
    image=np.zeros(size,dtype=dtype)
    image.fill(value)
    return image

def add_gaussian_noise(image,mean,std):
    #return (np.round(image+np.random.normal(mean,std,image.shape))).astype(image.dtype)
    return image+np.random.normal(mean,std,image.shape)


def add_normal_centered_noise(image,amplitude):
    noise=(np.random.rand(image.shape[0],image.shape[1]).astype(np.float)-0.5)*amplitude
    return image+noise


def add_salt_pepper_noise(image,percentage_of_noisy_pixels):
    x=np.random.random(size=image.shape)
    #thresh=.4
    white=np.where(x.flatten()>(1-percentage_of_noisy_pixels))[0]
    black=np.where(x.flatten()<percentage_of_noisy_pixels)[0]

    #print 'Pct white %4.4f black %4.4f' % (len(white)/float(len(x.flatten())),len(black)/float(len(x.flatten())))
    noiseimg=np.copy(image).flatten()
    noiseimg[white]=0
    noiseimg[black]=255
    noiseimg=noiseimg.reshape(image.shape)
    return noiseimg

def generate_directional_wave(size,amplitude,period,orientation=0,offset=0,type=np.int16):
    """
    image[i,j]=amplitude*cos(theta)+offset.
    theta=2pi*period
    orientation radian
    Exemple: image=skit.generate_directional_wave((50,50), 125,3.0, orientation=2*np.pi/6)
    Exemple: horizontal: image=skit.generate_directional_wave((50,50), 125,3.0, orientation=0)
    """
    image=np.zeros(size,dtype=type)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            #theta=np.mod(i,(2*np.pi*3)) #lignes horizontales
            #theta=np.mod(j*np.cos(orientation)+i*np.sin(orientation),(2*np.pi*period)) #ligne
            theta=2*np.pi*(j*np.cos(orientation)+i*np.sin(orientation))/period#ligne
            #theta=np.mod(np.sqrt(i**2+j**2),(2*np.pi*3)) #arc de cercle
            #theta=np.mod(j,(2*np.pi*2)) #lignes verticales
            image[i,j]+=amplitude*np.cos(theta)+offset
    return image

def generate_circular_wave(size,amplitude,period,center=(0,0),offset=0,type=np.int16):
    """
    doc
    """
    image=np.zeros(size,dtype=type)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            #theta=np.mod(i,(2*np.pi*3)) #lignes horizontales
            #theta=np.mod(j*np.cos(orientation)+i*np.sin(orientation),(2*np.pi*period)) #ligne
            #theta=np.mod(np.sqrt(i**2+j**2),(2*np.pi*period)) #arc de cercle
            theta=2*np.pi*np.sqrt((i-center[0])**2+(j-center[1])**2)/period #arc de cercle
            #theta=np.mod(j,(2*np.pi*2)) #lignes verticales
            image[i,j]+=amplitude*np.cos(theta)+offset
    return image


def generate_gaussian_image(shape=(11,11),c_x=5,c_y=5,sigma_x=2.0,sigma_y=2.0,max=10):
    sigma_x=float(sigma_x)
    sigma_y=float(sigma_y)
    amplitude_model=np.zeros(shape, dtype=np.float)
    for x in range(0,amplitude_model.shape[0]-1):
        for y in range(0,amplitude_model.shape[1]-1):
            amplitude_model[x,y]=max*np.exp(-(((x-c_x)/sigma_x)**2+((y-c_y)/sigma_y)**2)/2.0)
    return amplitude_model
