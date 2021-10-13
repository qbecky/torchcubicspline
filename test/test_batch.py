import sys as _sys
_sys.path.append("../torchcubicspline/")

import torch
from torchcubicspline import (natural_cubic_spline_coeffs, 
                              NaturalCubicSpline)
import matplotlib.pyplot as plt
from matplotlib import gridspec

def ToNumpy(tensor):
    return tensor.cpu().detach().numpy()

if __name__=="__main__":
    torch.manual_seed(1)

    batch_size, length, channels = 3, 7, 2
    # t = torch.linspace(0, 1, length).reshape(1, -1).repeat(batch_size, 1)
    t = torch.rand(batch_size, length)
    t[:, 0] = 0
    t = torch.cumsum(t, dim=1) / torch.sum(t, dim=1, keepdim=True)

    torch.manual_seed(2)
    xs = torch.rand(batch_size, length, channels)
    coeffs = natural_cubic_spline_coeffs(t, xs)
    old_spline = NaturalCubicSpline(coeffs)

    refined_t = torch.linspace(0, 1, 1000).unsqueeze(0).repeat(batch_size, 1)
    old_x_spline = old_spline.evaluate(refined_t)

    print("Coeffs size: {}".format(coeffs[0].shape))
    print("Splines positions shape: {}".format(old_x_spline.shape))

    gs = gridspec.GridSpec(nrows=1, ncols=1, width_ratios=[1], height_ratios=[1])
    fig = plt.figure(figsize=(8, 8))

    axTmp = plt.subplot(gs[0, 0])
    for x, x_spline in zip(xs, old_x_spline):
        axTmp.plot(ToNumpy(x_spline[:, 0]), ToNumpy(x_spline[:, 1]), c='b')
        axTmp.scatter(ToNumpy(x[:, 0]), ToNumpy(x[:, 1]), c='k')
    axTmp.set_title("Spline", fontsize=14)
    axTmp.grid()

    plt.show()
