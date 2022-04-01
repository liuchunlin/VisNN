import numpy as np


class Heatmap:
    def __init__(self, width, height, minx, maxx, miny, maxy):
        self.width = width
        self.height = height
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        # generate x data
        x = np.linspace(minx, maxx, width, dtype=np.float32)
        y = np.linspace(miny, maxy, height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        self.xdata = np.hstack([xv.reshape([-1, 1]), yv.reshape(-1, 1)])

    def get_coord_of_data(self, X):
        x, y = X[:, 0], X[:, 1]
        ix = (x - self.minx) / (self.maxx - self.minx) * self.width
        ix = np.clip(np.int32(np.round(ix)), 0, self.width-1)
        iy = (y - self.miny) / (self.maxy - self.miny) * self.height
        iy = np.clip(np.int32(np.round(iy)), 0, self.height-1)
        return ix, iy

