import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import skimage

plt.style.use('seaborn-dark')

class Draw():
    def __init__(self, x, y, z, images):
        self.fig = plt.figure(figsize=(14,7))   ; 
        self.axis = [self.fig.add_subplot(121, projection='3d'),
                     self.fig.add_subplot(122)]

        # self.fig.add_axes(self.ax)
        self.frame = 0

        self.axis[0].view_init(90,90)

        self.clear_figure(self.axis[0])
        self.fig.tight_layout()

        self.x = x
        self.y = y
        self.z = z
        self.images = images

        self.anim = 0
        self.paused = False
        self.clikked = 0
    
    def clear_figure(self, ax):
        # pass
        ax.cla()
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    def animate(self,i):
        self.clear_figure(self.axis[0])
        self.axis[1].cla()

        self.axis[0].scatter(self.x[i],self.y[i],self.z[i], linewidth = 3)
        self.axis[1].imshow(self.images[i])
        self.axis[1].invert_yaxis()
        self.axis[1].axis('off')

        plt.gca().invert_xaxis() 
        plt.gca().invert_yaxis()
        self.frame += 1
        self.frame %= len(X)

    def onclick(self,event):
        xx, yy = event.xdata,event.ydata
        distance = (self.x[self.frame] - xx)**2 + (self.y[self.frame] - yy)**2
        self.clikked = np.argmin(distance)

    def begin_drawings(self):
        self.anim = FuncAnimation(self.fig, self.animate,
                                       frames=len(X), interval=50, blit=False)
        cid = self.fig.canvas.mpl_connect('button_press_event',self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.toggle_pause)
        plt.show()

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.anim.resume()
        else:
            self.anim.pause()
        self.paused = not self.paused

X  = np.stack([ np.load(os.path.join("/home/holdee/aus_estimation/data/pre_processed/disfa/facemesh/SN030", x)) for x in sorted(os.listdir("/home/holdee/aus_estimation/data/pre_processed/disfa/facemesh/SN030"))])

images  = [ skimage.io.imread(os.path.join("/home/holdee/aus_estimation/data/pre_processed/disfa/images/crop/SN030", y)) for y in sorted(os.listdir("/home/holdee/aus_estimation/data/pre_processed/disfa/images/crop/SN030"))]

mean_x = np.mean(X[:,:,0])
mean_y = np.mean(X[:,:,1])
mean_z = np.mean(X[:,:,2])

std_x = np.std(X[:,:,0])
std_y = np.std(X[:,:,1])
std_z = np.std(X[:,:,2])

x = (X[:,:,0] - mean_x)/std_x
y = (X[:,:,1] - mean_y)/std_y
z = (X[:,:,2] - mean_z)/std_z

draw = Draw(x = x, y = y, z = z, images = images)
draw.begin_drawings()