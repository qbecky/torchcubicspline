from posixpath import join
import sys as _sys
_sys.path.append("../torchcubicspline/")

import numpy as np
import torch
from torchcubicspline import (natural_cubic_spline_coeffs, 
                              NaturalCubicSpline, NaturalCubicSplineWithVaryingTs)
import matplotlib.pyplot as plt
from matplotlib import gridspec

def ToNumpy(tensor):
    return tensor.cpu().detach().numpy()

def GetEdgesRegularTopology(nJx, nJy):
    edges  = [[nJx * j + i, nJx * j + i + 1] for j in range(nJy) for i in range(nJx-1)]
    edges += [[nJx * j + i, nJx * (j + 1) + i] for i in range(nJx) for j in range(nJy-1)]
    return torch.tensor(edges, dtype=torch.int32)

def MakeConstantSpeed(init_knots, multiplier):
    '''
    Input:
    - init_knots : an array of shape (?, n_knots, 2 or 3) containing the initial knot positions
    - multiplier : the factor by which the number of knot is multiplied

    Output:
    - new_splines : ? new splines
    - s_knots     : parameter associated to each knot shape (?, n_knots)
    - refined_s   : new knots' parameter (?, n_knots*multiplier)
    - new_knots   : new knots' (?, n_knots*multiplier, 2 or 3)
    '''
    init_ts     = torch.linspace(0., 1., init_knots.shape[1])
    init_coeffs = natural_cubic_spline_coeffs(init_ts, init_knots)
    splines     = NaturalCubicSpline(init_coeffs)
    
    # Start arc-length reparameterization
    refined_ts = torch.linspace(0, 1, (init_knots.shape[1] - 1) * multiplier + 1)
    new_knots  = splines.evaluate(refined_ts)
    lengths    = torch.norm(new_knots[:, 1:, :] - new_knots[:, :-1, :], dim=2)
    cum_len    = torch.cumsum(lengths, dim=1)
    refined_s  = torch.cat([torch.zeros(size=(init_knots.shape[0], 1)), cum_len], dim=1) / cum_len[:, -1].reshape(-1, 1)
    
    # New splines
    new_coeffs  = natural_cubic_spline_coeffs(refined_s, new_knots)
    new_splines = NaturalCubicSplineWithVaryingTs(new_coeffs)

    # Parameters associated to the old knots
    s_knots = refined_s[:, ::multiplier].contiguous()
    
    return new_splines, s_knots, refined_s, new_knots

def ComputeElasticRodsDiscretization(controlPoints, nJoints, subdivision, mult=10):
    '''
    Input:
    - controlPoints : shape (nRods, stride*nJoints, 3)
    - nJoints       : number of joints per rod segment
    - subdivision   : number of discretized points per rod segments
    - mult          : multiplier for making curves unit speed

    Output:
    - newSplines
    - discSPoints : shape (nRods, subdivision*(nJoints-1)) containing the curve parameters of the 
    '''

    nRods = controlPoints.shape[0]
    # controlPoints.shape[1] = stride * (nJoints - 1) + 1
    stride = int((controlPoints.shape[1] - 1) // (nJoints - 1))
    # stride = int(controlPoints.shape[1] // stride)

    ## Make them unit speed
    newSplines, newSKnots, _, _ = MakeConstantSpeed(controlPoints, mult)
    lenCurves = newSKnots[:, 1::stride] - newSKnots[:, :-1:stride]     # (nRods, nJoints-1)
    minLen    = torch.minimum(lenCurves[:, 1:], lenCurves[:, :-1])     # (nRods, nJoints-2)

    ## Compute the discretized points' position
    sharedLen = torch.zeros_like(lenCurves)                               # (nRods, nJoints-1)
    sharedLen[:, :-1] += minLen / (2 * subdivision)
    sharedLen[:, 1:]  += minLen / (2 * subdivision)                
    remSub             = subdivision * torch.ones_like(lenCurves)         # (nRods, nJoints-1)
    remSub[:, :-1]    -= 1
    remSub[:, 1:]     -= 1
    lenOfEdge          = (lenCurves - sharedLen) / remSub                 # (nRods, nJoints-1)

    ## It remains to add stuff at each joint
    allLenEdges = torch.zeros(size=(nRods, (nJoints-1) + (nJoints-2))) # Even indices: edge len at each rod segment. Odd indices: edge length at joints (shared)
    allLenEdges[:, 0::2] = lenOfEdge
    allLenEdges[:, 1::2] = minLen / (2 * subdivision)

    ## Get the discretized lengths
    repeats       = torch.zeros(size=(allLenEdges.shape[1],)).to(dtype=torch.int32)
    repeats[0::2] = remSub[0, :]
    repeats[1::2] = 2
    lenPerEdge    = torch.repeat_interleave(allLenEdges, repeats, dim=1)                                              # (nRods, (nJoints-1)*subdivision)

    discLenS      = torch.cumsum(torch.cat((torch.zeros(size=(lenPerEdge.shape[0], 1)), lenPerEdge), dim=1), dim=1)   # (nRods, 1 + (nJoints-1)*subdivision)
    discSPoints   = discLenS / discLenS[:, -1].reshape(-1, 1)

    return newSplines, discSPoints



def PrintSpeeds(splines, nSplines):
    t = torch.linspace(0, 1, 100).reshape(1, -1).repeat(nSplines, 1)
    
    gs = gridspec.GridSpec(nrows=1, ncols=1, width_ratios=[1], height_ratios=[1])
    fig = plt.figure(figsize=(8, 8))

    axTmp = plt.subplot(gs[0, 0])
    ptsSplines = splines.evaluate(t)
    speedSplines = torch.linalg.norm(ptsSplines[:, 1:, :] - ptsSplines[:, :-1, :], axis=2)
    for speed in speedSplines:
        axTmp.plot(np.linspace(0., 1., t.shape[1]-1), ToNumpy(speed))
    axTmp.set_title("Speed per curve", fontsize=14)
    axTmp.set_xlabel("Curve parameter", fontsize=14)
    axTmp.grid()

if __name__=="__main__":
    torch.manual_seed(1)

    nJx, nJy    = 4, 3
    mult        = 21 # To compute a constant length parameterization of the curves
    subdivision = 10

    # Grid like position of the joints
    scale_linkage = 1000

    edges_regular = GetEdgesRegularTopology(nJx, nJy)

    # Intialize the joints position
    torch.manual_seed(2)

    jointsTorch = torch.zeros(size=(nJx, nJy, 3))
    jointsTorch[:, :, 0]   = scale_linkage * torch.arange(nJx).reshape(-1, 1).repeat(1, nJy)
    jointsTorch[:, :, 1]   = scale_linkage * torch.arange(nJy).reshape(1, -1).repeat(nJx, 1)
    jointsTorch[:, :, :2] += 0.1 * scale_linkage * torch.randn(nJx, nJy, 2)
    jointsTorch.requires_grad = True

    # Initialize control points along horizontal rods
    yHorMidControl = jointsTorch[:-1, :, 1].detach() + scale_linkage/50 * torch.randn(nJx-1, nJy)
    yHorMidControl.requires_grad = True
    horMidControl = torch.zeros(size=(nJx-1, nJy, 3))
    horMidControl[:, :, 0] = (jointsTorch[1:, :, 0] + jointsTorch[:-1, :, 0]) / 2
    horMidControl[:, :, 1] = yHorMidControl

    # Initialize control point along vertical rods
    xVertMidControl = jointsTorch[:, :-1, 0].detach() + scale_linkage/50 * torch.randn(nJx, nJy-1)
    xVertMidControl.requires_grad = True
    vertMidControl = torch.zeros(size=(nJx, nJy-1, 3))
    vertMidControl[:, :, 0] = xVertMidControl
    vertMidControl[:, :, 1] = (jointsTorch[:, 1:, 1] + jointsTorch[:, :-1, 1]) / 2

    # Horizontal edges
    x             = torch.zeros(size=(2*nJx-1, nJy, 3))
    x[0::2, :, :] = jointsTorch
    x[1::2, :, :] = horMidControl

    ## Get the positions
    newSplinesHor, discSHor = ComputeElasticRodsDiscretization(torch.swapaxes(x, 0, 1), nJx, subdivision, mult=mult)
    discHor = newSplinesHor.evaluate(discSHor)

    # Vertical edges
    x             = torch.zeros(size=(nJx, 2*nJy-1, 3))
    x[:, 0::2, :] = jointsTorch
    x[:, 1::2, :] = vertMidControl

    ## Get the positions
    newSplinesVert, discSVert = ComputeElasticRodsDiscretization(x, nJy, subdivision, mult=mult)
    discVert = newSplinesVert.evaluate(discSVert)

    gs = gridspec.GridSpec(nrows=1, ncols=1, width_ratios=[1], height_ratios=[1])
    fig = plt.figure(figsize=(8, 8))

    axTmp = plt.subplot(gs[0, 0])
    # newJointsHor  = new_spline_hor.evaluate(new_s_knots_hor[:,::2]).reshape(-1, 3)
    # newJointsVert = new_spline_vert.evaluate(new_s_knots_vert[:,::2]).reshape(-1, 3)
    # axTmp.scatter(ToNumpy(newJointsHor[:, 0]), ToNumpy(newJointsHor[:, 1]), c='b', s=10)
    # axTmp.scatter(ToNumpy(newJointsVert[:, 0]), ToNumpy(newJointsVert[:, 1]), c='b', s=10)
    for xSpline in discHor:
        axTmp.plot(ToNumpy(xSpline[:, 0]), ToNumpy(xSpline[:, 1]), c='b')
        axTmp.scatter(ToNumpy(xSpline[:, 0]), ToNumpy(xSpline[:, 1]), c='b', s=10)
    for xSpline in discVert:
        axTmp.plot(ToNumpy(xSpline[:, 0]), ToNumpy(xSpline[:, 1]), c='k')
        axTmp.scatter(ToNumpy(xSpline[:, 0]), ToNumpy(xSpline[:, 1]), c='k', s=10)
    axTmp.set_title("Splines network", fontsize=14)
    axTmp.grid()

    PrintSpeeds(newSplinesHor, nJy)
    PrintSpeeds(newSplinesVert, nJx)

    plt.show()



