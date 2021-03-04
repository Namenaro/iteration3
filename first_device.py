from utils import get_mnist_number

import numpy as np
import matplotlib.pyplot as plt

class FirstDeviceCreator:
    def __init__(self, hside, image):
        self.image = image
        self.hside = hside
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.result = None

    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        x = int(event.xdata + 0.5)
        y = int(event.ydata + 0.5)
        plt.scatter(x, y, s=100, c='red', marker='o', alpha=0.4)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.result = self.get_rect_by_center(x, y)

    def get_rect_by_center(self, centerx, centery):
        print ("centerer is " + str(centerx) + ", " + str(centery))
        minx = centerx - self.hside
        maxx = centerx + self.hside

        miny= centery - self.hside
        maxy = centery + self.hside
        if minx<0 or miny<0:
            return None

        return self.image[miny:maxy+1, minx:maxx+1 ]

    def create_device(self):
        plt.imshow(self.image, cmap='gray_r')
        plt.show()

        return self.result

if __name__ == "__main__":
    mnist_number = 1
    image = get_mnist_number(mnist_number)
    devcr = FirstDeviceCreator(4, image)
    matrix = devcr.create_device()
    print (matrix.shape)
    plt.imshow(matrix, cmap='gray_r')
    plt.show()