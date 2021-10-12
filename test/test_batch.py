import sys as _sys
_sys.path.append("../torchcubicspline/")

import torch
from torchcubicspline import (natural_cubic_spline_coeffs, 
                              NaturalCubicSpline)

if __name__=="__main__":
    torch.manual_seed(2)

    batch_size, length, channels = 3, 7, 2
    t = torch.linspace(0, 1, length).reshape(1, -1).repeat(batch_size, 1)
    # t = torch.linspace(0, 1, length)
    x = torch.rand(batch_size, length, channels)
    coeffs = natural_cubic_spline_coeffs(t, x)
    old_spline = NaturalCubicSpline(coeffs)

    print("Coeffs size: {}".format(coeffs[0].shape))