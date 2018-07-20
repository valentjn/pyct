#!/usr/bin/python3

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import pyct
import testfcns



xx = np.array([np.linspace(0, 1, 201)]).T

xx0 = np.linspace(0, 1, 65)
xx1 = np.linspace(0, 1, 65)
XX0, XX1 = np.meshgrid(xx0, xx1)
XX = np.column_stack((XX0.flatten(), XX1.flatten()))



if False:
  np.random.seed(342)
  d = 1
  basis = d * [{"type" : "hermiteValue", "degree" : 1}]
  l = [3]
  fX = np.random.rand(9)
  #dfX = 2 * np.random.rand(9, d) - 1
  dfX = 5 * np.ones((9, d))
  c = pyct.interpolateFullGrid(basis, l, fX)
  
  yy = pyct.evaluateFullGrid(basis, l, c, xx)
  yy = pyct.interpolateEvaluateBHCombinationFullGrid(basis, l, fX, dfX, xx)
  
  fig = plt.figure(figsize=(7, 7))
  ax = fig.gca()
  ax.plot(xx, yy, "k-")
  X = pyct.getFullGrid(l)
  ax.plot(X[:,0], fX, "k.")
  ax.plot(X[:,0], dfX[:,0], "r.")
  plt.show()
  
  
  
  np.random.seed(342)
  functionType = "branin"
  d = 2
  basis = d * [{"type" : "bSpline", "degree" : 1}]
  l = [1, 2]
  f = testfcns.getFunction(functionType, d)
  X = pyct.getFullGrid(l)
  shape = [len(pyct.getFullGridPoints1D(l1D)) for l1D in l]
  fX  = np.reshape(f(X), shape)
  
  #c = interpolateFullGrid(basis, l, fX)
  #YY = evaluateFullGrid(basis, l, c, XX)
  
  df = testfcns.getFunctionGradient(functionType, d)
  dfX = np.reshape(df(X), shape + [d])
  YY = pyct.interpolateEvaluateBHCombinationFullGrid(basis, l, fX, dfX, XX)
  
  #df = testfcns.getFunctionFirstDerivatives(functionType, d)
  #dfX = df(X)
  #dfX = {order : np.reshape(dfX[order], shape) for order in dfX}
  #YY = interpolateEvaluateDerivativeCombinationFullGrid(
  #    basis, l, fX, dfX, XX, derivatives="mixed")
  
  YY = np.reshape(YY, XX0.shape)
  
  fig = plt.figure(figsize=(7, 7))
  ax = fig.add_subplot(111, projection="3d")
  ax.plot_surface(XX0, XX1, YY)
  ax.plot(X[:,0], X[:,1], "k.", zs=fX.flatten())
  
  #ax.plot(X[:,0], X[:,1], "r.", zs=dfX[:,:,0].flatten())
  #ax.plot(X[:,0], X[:,1], "y.", zs=dfX[:,:,1].flatten())
  
  #ax.plot(X[:,0], X[:,1], "r.", zs=dfX[(1, 0)].flatten())
  #ax.plot(X[:,0], X[:,1], "y.", zs=dfX[(0, 1)].flatten())
  
  plt.show()
  
  
  
  functionType = "branin"
  d = 2
  basis = d * [{"type" : "hermiteValue", "degree" : 1}]
  n = 3
  f = testfcns.getFunction(functionType, d)
  X = pyct.getRegularSparseGridPoints(n, d)
  fX = f(X)
  
  #df = testfcns.getFunctionGradient(functionType, d)
  #dfX = df(X)
  #interpEvalFullGridFcn = (lambda l, fX, dfX, XX:
  #    interpolateEvaluateFullGrid(basis, l, fX, XX))
  #interpEvalFullGridFcn = pyct.InterpolatorEvaluatorFullGrid(basis)
  
  #df = testfcns.getFunctionGradient(functionType, d)
  #dfX = df(X)
  #interpEvalFullGridFcn = (
  #    pyct.InterpolatorEvaluatorBHCombinationFullGrid(basis))
  
  df = testfcns.getFunctionFirstDerivatives(functionType, d)
  dfX = df(X)
  interpEvalFullGridFcn = (
      pyct.InterpolatorEvaluatorDerivativeCombinationFullGrid(basis))
  
  YY = pyct.interpolateEvaluateCTCombination(
      interpEvalFullGridFcn, n, X, fX, dfX, XX)
  YY = np.reshape(YY, XX0.shape)
  
  fig = plt.figure(figsize=(7, 7))
  ax = fig.add_subplot(111, projection="3d")
  ax.plot_surface(XX0, XX1, YY)
  ax.plot(X[:,0], X[:,1], "k.", zs=fX.flatten())
  #ax.plot(X[:,0], X[:,1], "r.", zs=dfX[:,0])
  #ax.plot(X[:,0], X[:,1], "y.", zs=dfX[:,1])
  plt.show()



functionType = "branin"
d = 2
basis = d * [{"type" : "bSpline", "degree" : 1}]
n = 7
f = testfcns.getFunction(functionType, d)
X = pyct.getRegularSparseGridPoints(n, d)
fX = f(X)

np.random.seed(342)
NN = 10000
XX = np.random.random((NN, d))
fXX = f(XX)

df = testfcns.getFunctionGradient(functionType, d)
dfX = df(X)
interpEvalFullGridFcn = pyct.InterpolatorEvaluatorBHCombinationFullGrid(basis)

YY = pyct.interpolateEvaluateCTCombination(
    interpEvalFullGridFcn, n, X, fX, dfX, XX)
print(np.sqrt(np.mean((fXX - YY)**2)))
