from posixpath import join
import sys as _sys
_sys.path.append("../torchcubicspline/")

import numpy as np
import torch
import torch.optim as optim
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

def ComputeStraightness(discPoints):
    '''
    Input:
    - discPoints : torch tensor of shape (nRods, nDisc, 3) containing the discretized point along the rod

    Output:
    - cosEdges : tensor of shape (nRods, nDisc-2) containing the cosine similarity between neighboring edges
    '''

    edges    = discPoints[:, 1:, :] - discPoints[:, :-1, :]                         # (nRods, nDisc-1, 3)
    dotEdges = torch.einsum('ijk, ijk -> ij', edges[:, 1:, :], edges[:, :-1, :])    # (nRods, nDisc-2)
    lenEdges = torch.linalg.norm(edges, dim=2)                                      # (nRods, nDisc-1)
    cosEdges = dotEdges / (lenEdges[:, 1:] * lenEdges[:, :-1])                      # (nRods, nDisc-2)
    return cosEdges

def ComputeShapePreservation(tensor1, tensor2):
    ''' 
    Input:
    - tensor1 : tensor to compare with tensor2
    - tensor2 : same size as tensor1

    Output:
    - similarity : similarity between current and target
    '''

    return 0.5 * torch.sum((tensor1 - tensor2) ** 2)

def Criteria(jointsTorch, yHorMidControl, xVertMidControl, returnSplines):
    '''
    Input:
    - jointsTorch     : tensor of shape (nJx, nJy, 3) containing the joints position
    - yHorMidControl  : tensor of shape (nJx-1, nJy) containing the control point y coordinate for horizontal edges
    - xVertMidControl : tensor of shape (nJx, nJy-1) containing the control point x coordinate for vertical edges
    - returnSplines   : whether we return splines as well or not

    Output:
    - straightHor   : measure of straightness of horizontal beams
    - straightVer   : measure of straightness of vertical beams
    - presJoints    : preservation energy at the joints
    - newSplinesHor : horizontal splines (arc-length parameterized)
    - newSplinesVer : vertical splines (arc-length parameterized)
    '''

    horMidControl = torch.zeros(size=(nJx-1, nJy, 3))
    horMidControl[:, :, 0] = (jointsTorch[1:, :, 0] + jointsTorch[:-1, :, 0]) / 2
    horMidControl[:, :, 1] = yHorMidControl

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

    straightHor = 1. - torch.mean(ComputeStraightness(discHor))
    straightVer = 1. - torch.mean(ComputeStraightness(discVert))
    presJoints  = ComputeShapePreservation(jointsTorch, jointsTorchInit)

    listOut = [straightHor, straightVer, presJoints]
    if returnSplines:
        listOut += [newSplinesHor, newSplinesVert]
    return listOut


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

    jointsTorchInit     = jointsTorch.detach().clone()
    yHorMidControlInit  = yHorMidControl.detach().clone()
    xVertMidControlInit = xVertMidControl.detach().clone()

    # Horizontal edges
    x             = torch.zeros(size=(2*nJx-1, nJy, 3))
    x[0::2, :, :] = jointsTorch
    x[1::2, :, :] = horMidControl

    ## Get the positions
    initSplinesHor, discSHor = ComputeElasticRodsDiscretization(torch.swapaxes(x, 0, 1), nJx, subdivision, mult=mult)
    discHor = initSplinesHor.evaluate(discSHor)

    # Vertical edges
    x             = torch.zeros(size=(nJx, 2*nJy-1, 3))
    x[:, 0::2, :] = jointsTorch
    x[:, 1::2, :] = vertMidControl

    ## Get the positions
    initSplinesVert, discSVert = ComputeElasticRodsDiscretization(x, nJy, subdivision, mult=mult)
    discVert = initSplinesVert.evaluate(discSVert)

    # gs = gridspec.GridSpec(nrows=1, ncols=1, width_ratios=[1], height_ratios=[1])
    # fig = plt.figure(figsize=(8, 8))

    # axTmp = plt.subplot(gs[0, 0])
    # newJointsHor  = new_spline_hor.evaluate(new_s_knots_hor[:,::2]).reshape(-1, 3)
    # newJointsVert = new_spline_vert.evaluate(new_s_knots_vert[:,::2]).reshape(-1, 3)
    # axTmp.scatter(ToNumpy(newJointsHor[:, 0]), ToNumpy(newJointsHor[:, 1]), c='b', s=10)
    # axTmp.scatter(ToNumpy(newJointsVert[:, 0]), ToNumpy(newJointsVert[:, 1]), c='b', s=10)
    # for xSpline in discHor:
    #     axTmp.plot(ToNumpy(xSpline[:, 0]), ToNumpy(xSpline[:, 1]), c='b')
    #     axTmp.scatter(ToNumpy(xSpline[:, 0]), ToNumpy(xSpline[:, 1]), c='b', s=10)
    # for xSpline in discVert:
    #     axTmp.plot(ToNumpy(xSpline[:, 0]), ToNumpy(xSpline[:, 1]), c='k')
    #     axTmp.scatter(ToNumpy(xSpline[:, 0]), ToNumpy(xSpline[:, 1]), c='k', s=10)
    # axTmp.set_title("Splines network", fontsize=14)
    # axTmp.grid()

    # PrintSpeeds(newSplinesHor, nJy)
    # PrintSpeeds(newSplinesVert, nJx)

    # plt.show()

    straightVer = 1. - torch.mean(ComputeStraightness(discVert))
    straightHor = 1. - torch.mean(ComputeStraightness(discHor))
    presJoints  = ComputeShapePreservation(jointsTorch, jointsTorchInit)

    loss = straightVer + straightHor + presJoints
    loss.backward()

    # Start the optimization

    jointsTorchOptim     = jointsTorch.detach().clone()
    yHorMidControlOptim  = yHorMidControl.detach().clone()
    xVertMidControlOptim = xVertMidControl.detach().clone()

    jointsTorchOptim.requires_grad = True
    yHorMidControlOptim.requires_grad = True
    xVertMidControlOptim.requires_grad = True

    descender = optim.Adam([jointsTorchOptim, yHorMidControlOptim, xVertMidControlOptim], lr=0.1*scale_linkage)
    optim_steps = 200
    weights     = [1., 1., 1e-3/(scale_linkage ** 2)]

    straightVerOptim = torch.zeros(size=(optim_steps+1,))
    straightHorOptim = torch.zeros(size=(optim_steps+1,))
    presJointsOptim  = torch.zeros(size=(optim_steps+1,))

    straightVerOptim[0] = straightVer.detach()
    straightHorOptim[0] = straightHor.detach()
    presJointsOptim[0]  = presJoints .detach()

    for i in range(optim_steps):
        
        descender.zero_grad()
        listOut = Criteria(jointsTorchOptim, yHorMidControlOptim, xVertMidControlOptim, returnSplines=True)
        loss = weights[0] * listOut[0] + weights[1] * listOut[1] + weights[2] * listOut[2]
        straightHorOptim[i+1] = weights[0] * listOut[0].detach()
        straightVerOptim[i+1] = weights[1] * listOut[1].detach()
        presJointsOptim[i+1]  = weights[2] * listOut[2] .detach()
        loss.backward()
        descender.step()

    initJointsNP = ToNumpy(jointsTorchInit).reshape(-1, 3)
    optJoints    = ToNumpy(jointsTorchOptim).reshape(-1, 3)

    # Final positions
    nDisc = 100
    tHor = torch.linspace(0., 1., nDisc).reshape(1, -1).repeat(nJy, 1)
    tVer = torch.linspace(0., 1., nDisc).reshape(1, -1).repeat(nJx, 1)
    horPts     = ToNumpy(listOut[3].evaluate(tHor))
    horPtsInit = ToNumpy(initSplinesHor.evaluate(tHor))
    verPts     = ToNumpy(listOut[4].evaluate(tVer))
    verPtsInit = ToNumpy(initSplinesVert.evaluate(tVer))

    gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 0.05, 2], height_ratios=[1])
    fig = plt.figure(figsize=(12, 4))

    axTmp = plt.subplot(gs[0, 0])
    axTmp.scatter(initJointsNP[:, 0], initJointsNP[:, 1], c='b', s=10, label='Initial')
    for horSpline in horPtsInit:
        axTmp.plot(horSpline[:, 0], horSpline[:, 1], c='b')
    for verSpline in verPtsInit:
        axTmp.plot(verSpline[:, 0], verSpline[:, 1], c='b')
    axTmp.scatter(optJoints[:, 0], optJoints[:, 1], c='k', s=10, label='Optimized')
    for horSpline in horPts:
        axTmp.plot(horSpline[:, 0], horSpline[:, 1], c='k')
    for verSpline in verPts:
        axTmp.plot(verSpline[:, 0], verSpline[:, 1], c='k')
    axTmp.set_title("Splines network", fontsize=14)
    axTmp.legend(fontsize=12)
    axTmp.grid()

    axTmp = plt.subplot(gs[0, 2])
    axTmp.plot(np.arange(optim_steps+1), ToNumpy(straightVerOptim), label='Straightness vertical')
    axTmp.plot(np.arange(optim_steps+1), ToNumpy(straightHorOptim), label='Straightness horizontal')
    axTmp.plot(np.arange(optim_steps+1), ToNumpy(presJointsOptim), label='Joints positions preservation')
    axTmp.set_title("Losses as optimization goes", fontsize=14)
    axTmp.legend(fontsize=12)
    axTmp.grid()

    plt.show()
    
