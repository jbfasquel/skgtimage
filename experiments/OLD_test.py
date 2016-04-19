import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt

'''
def press(event):
    print('press', event.key)
    import sys
    sys.stdout.flush()
    if event.key == 'x':
        visible = xl.get_visible()
        xl.set_visible(not visible)
        fig.canvas.draw()

fig, ax = plt.subplots()

fig.canvas.mpl_connect('key_press_event', press)

ax.plot(np.random.rand(12), np.random.rand(12), 'go')
xl = ax.set_xlabel('easy come, easy go')

plt.show()
quit()
'''

class Slicer:
    def __init__(self,image_3d,slice_index=0):
        self.slice_index=slice_index
        self.image_3d=image_3d
    def mouse(self,event):
        print(event.key,event.button)
        if event.button == "up": ds=1
        elif event.button == "down": ds=-1
        else: ds=0
        self.slice_index+=ds
        if self.slice_index < 0: self.slice_index = 0
        if self.slice_index == self.image_3d.shape[2]: self.slice_index=self.image_3d.shape[2]-1

        image2D=self.image_3d[:,:,self.slice_index]

        plt.imshow(np.rot90(image2D),'gray')
        plt.gcf().canvas.draw()

def plot(image3D,slice=0):
    #CALLBACK ASSIGNEMENT
    slicer=Slicer(image3D,slice_index=slice)
    plt.gcf().canvas.mpl_connect('scroll_event', slicer.mouse)
    #plt.gcf().canvas.mpl_connect('key_press_event', slicer.mouse)
    slice=image3D[:,:,0]
    plt.imshow(np.rot90(slice),'gray');plt.axis('off')
    plt.show()

image=np.zeros((3,3,3))
plot(image)