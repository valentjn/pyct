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
  shape = [len(pyct.getFullGrid1D(l1D)) for l1D in l]
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
  X = pyct.getRegularSparseGrid(n, d)
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



def computeFullGridErrors(functionType, d, interpEvalFullGridFcn,
                          derivatives, nMax, NMax):
  errors = {}
  f = testfcns.getFunction(functionType, d)
  
  if derivatives == "none":
    pass
  elif derivatives == "simple":
    df = testfcns.getFunctionGradient(functionType, d)
  elif derivatives == "mixed":
    df = testfcns.getFunctionFirstDerivatives(functionType, d)
  else:
    raise ValueError("Unknown value for derivatives.")
  
  np.random.seed(342)
  NN = 10000
  XX = np.random.random((NN, d))
  fXX = f(XX)
  
  for n in range(nMax+1):
    l = n * np.ones((d,))
    X = pyct.getFullGrid(l)
    N = X.shape[0]
    if N > NMax: break
    fX  = f(X)
    dfX = (df(X) if derivatives != "none" else None)
    
    shape = [len(pyct.getFullGrid1D(l1D)) for l1D in l]
    fX = np.reshape(fX, shape)
    
    if isinstance(dfX, np.ndarray):
      dfX = np.reshape(dfX, shape + [dfX.shape[1]])
    elif isinstance(dfX, dict):
      dfX = {key : np.reshape(dfX[key], shape) for key in dfX}
    elif dfX is None:
      pass
    else:
      raise ValueError("Unknown dfX type")
    
    YY = interpEvalFullGridFcn(l, fX, dfX, XX)
    error = np.sqrt(np.mean((fXX - YY)**2))
    errors[N] = error
  
  return errors

def computeSparseGridErrors(functionType, d, interpEvalFullGridFcn,
                            derivatives, nMax, NMax):
  errors = {}
  f = testfcns.getFunction(functionType, d)
  
  if derivatives == "none":
    pass
  elif derivatives == "simple":
    df = testfcns.getFunctionGradient(functionType, d)
  elif derivatives == "mixed":
    df = testfcns.getFunctionFirstDerivatives(functionType, d)
  else:
    raise ValueError("Unknown value for derivatives.")
  
  np.random.seed(342)
  NN = 10000
  XX = np.random.random((NN, d))
  fXX = f(XX)
  
  for n in range(nMax+1):
    X = pyct.getRegularSparseGrid(n, d)
    N = X.shape[0]
    if N > NMax: break
    fX  = f(X)
    dfX = (df(X) if derivatives != "none" else None)
    YY = pyct.interpolateEvaluateCTCombination(
        interpEvalFullGridFcn, n, X, fX, dfX, XX)
    error = np.sqrt(np.mean((fXX - YY)**2))
    errors[N] = error
  
  return errors

def computeConvergenceOrders(errors):
  Ns = sorted(list(errors.keys()))
  orders = {}
  
  for prevN, N in zip(Ns[:-1], Ns[1:]):
    prevError, error = errors[prevN], errors[N]
    order = -np.log(error/prevError) / np.log(N/prevN)
    orders[N] = order
  
  return orders



errors = computeFullGridErrors("ackley", 2,
    pyct.InterpolatorEvaluatorBHCombinationFullGrid(
        2 * [{"type" : "bSpline", "degree" : 1}]),
    "simple", 10, 5000)
print(errors)
print(computeConvergenceOrders(errors))

import sys
sys.exit(0)

errors = computeSparseGridErrors("ackley", 2,
    pyct.InterpolatorEvaluatorBHCombinationFullGrid(
        2 * [{"type" : "bSpline", "degree" : 1}]),
    "simple", 10, 5000)
print(errors)
print(computeConvergenceOrders(errors))

errors = computeFullGridErrors("ackley", 2,
    pyct.InterpolatorEvaluatorDerivativeCombinationFullGrid(
        2 * [{"type" : "hermiteValue"}]),
    "mixed", 10, 5000)
print(errors)
print(computeConvergenceOrders(errors))

errors = computeSparseGridErrors("ackley", 2,
    pyct.InterpolatorEvaluatorDerivativeCombinationFullGrid(
        2 * [{"type" : "hermiteValue"}]),
    "mixed", 10, 5000)
print(errors)
print(computeConvergenceOrders(errors))
