import torch
import numpy as np
from survae.data.datasets.toy import PlaneDataset


class CornersDataset(PlaneDataset):
    '''Adapted from https://www.mathworks.com/matlabcentral/fileexchange/41459-6-functions-for-generating-artificial-datasets'''
    def _create_data(self):
        assert self.num_points % 8 == 0
        N = self.num_points
        scale = 1
        gapwidth = 1
        cornerwidth = 3

        xplusmin = torch.cat([torch.ones(N//4), -torch.ones(N//4), torch.ones(N//4), -torch.ones(N//4)])
        yplusmin = torch.cat([torch.ones(N//4), -torch.ones(N//2), torch.ones(N//4)])

        horizontal = torch.cat([xplusmin[::2] * gapwidth + xplusmin[::2] * scale * torch.rand(N//2),
                                yplusmin[::2] * gapwidth + cornerwidth * yplusmin[::2] * torch.rand(N//2)], dim=0)

        vertical = torch.cat([xplusmin[1::2] * gapwidth + cornerwidth * xplusmin[1::2] * torch.rand(N//2),
                              yplusmin[1::2] * gapwidth + yplusmin[1::2] * scale * torch.rand(N//2)], dim=0)

        data = torch.stack([horizontal, vertical], dim=-1)
        data[...,0] *= (2*torch.bernoulli(0.5*torch.ones(N))-1)
        data[...,1] *= (2*torch.bernoulli(0.5*torch.ones(N))-1)

        self.data = data


class EightGaussiansDataset(PlaneDataset):
    '''Adapted from https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py'''
    def _create_data(self):
        scale = 4.
        bias = np.pi / 8
        step = np.pi / 4
        centers = [(np.cos(bias + 0*step), np.sin(bias + 0*step)),
                   (np.cos(bias + 1*step), np.sin(bias + 1*step)),
                   (np.cos(bias + 2*step), np.sin(bias + 2*step)),
                   (np.cos(bias + 3*step), np.sin(bias + 3*step)),
                   (np.cos(bias + 4*step), np.sin(bias + 4*step)),
                   (np.cos(bias + 5*step), np.sin(bias + 5*step)),
                   (np.cos(bias + 6*step), np.sin(bias + 6*step)),
                   (np.cos(bias + 7*step), np.sin(bias + 7*step))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(self.num_points):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        self.data = torch.from_numpy(dataset)
