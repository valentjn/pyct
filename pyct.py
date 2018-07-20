#!/usr/bin/python3

import abc
import functools
import multiprocessing

import numpy as np
import scipy.signal



def asvoid(arr):
  """
  View the array as dtype np.void (bytes)
  This views the last axis of ND-arrays as bytes so
  you can perform comparisons on the entire row.
  http://stackoverflow.com/a/16840350/190597 (Jaime, 2013-05)
  Warning: When using asvoid for comparison,
  note that float zeros may compare UNEQUALLY
  >>> asvoid([-0.]) == asvoid([0.])
  array([False], dtype=bool)
  """
  arr = np.ascontiguousarray(arr)
  return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

def in1d_index(a, b):
  # from https://stackoverflow.com/a/22700221
  voida, voidb = map(asvoid, (a, b))
  return np.where(np.in1d(voidb, voida))[0]

def rowSortIndices(A):
  return np.lexsort(A.T[::-1,:])


def getFullGridPoints1D(l, boundaryTreatment="BH"):
  if l == 0:
    if boundaryTreatment == "BH":
      x = np.array([0.5])  # de Baar/Harding
    elif boundaryTreatment == "JV":
      x = np.array([0.0, 1.0]) # Valentin
    else:
      raise ValueError("Unknown boundary treatment")
  else:
    x = np.linspace(0, 1, 2**l + 1)
  
  return x

def getFullGrid(l, indices=False):
  Xs = [getFullGridPoints1D(l1D) for l1D in l]
  if indices: Xs = [list(range(len(X))) for X in Xs]
  Xs = np.meshgrid(*Xs, indexing="ij")
  X = np.column_stack([X.flatten() for X in Xs])
  return X

def evaluateBasis1D(basis, l, i, XX, boundaryTreatment="BH"):
  getXXK = lambda XX, K: XX[K]
  
  if (l == 0) and (boundaryTreatment == "BH"):
    if basis["type"] == "hermiteDeriv":
      YY = np.zeros_like(XX)
    else:
      YY = np.ones_like(XX)
  else:
    XX = 2**l * XX - i
    YY = np.zeros_like(XX)
    
    if basis["type"] == "bSpline":
      p = basis["degree"]
      K = np.logical_and((XX > -(p+1)/2), (XX < (p+1)/2))
      XXK = getXXK(XX, K)
      YY[K] = scipy.signal.bspline(XXK, p)
      # TODO: implement not-a-knot basis
    elif basis["type"] == "hermiteValue":
      K = np.logical_and((XX > -1), (XX <= 0))
      XXK = XX[K]
      YY[K] = ((-2*XXK - 3) * XXK) * XXK + 1
      K = np.logical_and((XX > 0),  (XX < 1))
      XXK = XX[K]
      YY[K] = (( 2*XXK - 3) * XXK) * XXK + 1
    elif basis["type"] == "hermiteDeriv":
      K = np.logical_and((XX > -1), (XX <= 0))
      XXK = XX[K]
      YY[K] = ((XXK + 2) * XXK + 1) * XXK
      K = np.logical_and((XX > 0),  (XX < 1))
      XXK = XX[K]
      YY[K] = ((XXK - 2) * XXK + 1) * XXK
    else:
      raise ValueError("Unknown basis")
  
  return YY

def getInterpolationMatrix1D(basis, l):
  X = getFullGridPoints1D(l)
  N = len(X)
  A = np.zeros((N, N))
  
  for i in range(N):
    A[:,i] = evaluateBasis1D(basis, l, i, X)
  
  return A

def interpolateFullGrid1D(basis, l, fX):
  if ((basis["type"] == "hermiteValue") or
      ((basis["type"] == "bSpline") and (basis["degree"] == 1))):
    c = np.array(fX)
  elif basis["type"] == "hermiteDeriv":
    c = np.array(fX) / 2**l
  else:
    A = getInterpolationMatrix1D(basis, l)
    c = np.linalg.solve(A, fX)
  
  return c

def interpolateFullGrid(basis, l, fX):
  d = len(l)
  c = np.array(fX)
  I = getFullGrid(l, indices=True)
  
  for t in range(d):
    notT = list(range(t)) + list(range(t+1, d))
    curI = np.array(I)
    curI[:,t] = -1
    curI = np.unique(curI, axis=0)
    
    for i in curI:
      i = list(i)
      i[t] = np.s_[:]
      c[i] = interpolateFullGrid1D(basis[t], l[t], c[i])
  
  return c

def evaluateFullGrid(basis, l, c, XX):
  d = len(l)
  YY = np.zeros((XX.shape[0],))
  I = getFullGrid(l, indices=True)
  
  for i in I:
    curYY = c[tuple(i)] * np.ones_like(YY)
    for t in range(d):
      curYY *= evaluateBasis1D(basis[t], l[t], i[t], XX[:,t])
    YY += curYY
  
  return YY

def interpolateEvaluateFullGrid(basis, l, fX, XX):
  c = interpolateFullGrid(basis, l, fX)
  YY = evaluateFullGrid(basis, l, c, XX)
  return YY

def interpolateEvaluateBHCombinationFullGrid(basis, l, fX, dfX, XX):
  d = len(l)
  YY = np.zeros((XX.shape[0],))
  
  for t in range(-1, d):
    for q in range(2):
      curBasis = list(basis)
      curFX = fX
      coeff = 1
      
      if t == -1:
        coeff = 1-d
        if q == 1: continue
      else:
        if q == 0:
          curBasis[t] = {"type" : "hermiteValue"}
        else:
          curBasis[t] = {"type" : "hermiteDeriv"}
          curFX = dfX[d * [np.s_[:]] + [t]]
      
      c = interpolateFullGrid(curBasis, l, curFX)
      YY += coeff * evaluateFullGrid(curBasis, l, c, XX)
  
  return YY

def interpolateEvaluateDerivativeCombinationFullGrid(
    basis, l, fX, dfX, XX, derivatives="mixed"):
  d = len(l)
  YY = np.zeros((XX.shape[0],))
  
  if derivatives == "simple":
    orders = np.eye(d, dtype=int).tolist()
    if not isinstance(dfX, dict):
      dfX = {tuple(orders[t+1]) : dfX[d * [np.s_[:]] + [t]]
             for t in range(d)}
  elif derivatives == "mixed":
    assert isinstance(dfX, dict)
    orders = list(dfX.keys())
    assert len(orders) == 2**d - 1
  else:
    raise ValueError("Unknown derivatives")
  
  orders.append(d * [0])
  
  for order in orders:
    curBasis = list(basis)
    curFX = (fX if sum(order) == 0 else dfX[tuple(order)])
    
    for t in range(d):
      if order[t] == 1: curBasis[t] = {"type" : "hermiteDeriv"}
    
    c = interpolateFullGrid(curBasis, l, curFX)
    YY += evaluateFullGrid(curBasis, l, c, XX)
  
  return YY

def numberOfLevels(n, d, includeZero=True):
  return scipy.special.comb((n+d-1 if includeZero else n-1), d-1, exact=True)

def enumerateLevels(n, d, includeZero=True):
  def enumerateLevelsRecursive(n, d, t, L, k):
    if d > 1:
      for m in range(n+1):
        kNew = enumerateLevelsRecursive(n-m, d-1, t+1, L, k)
        L[k:kNew,t] = m
        k = kNew
    else:
      L[k,t] = n
      k += 1
    return k
  
  N = numberOfLevels(n, d, includeZero=includeZero)
  
  if includeZero:
    if n >= 0:
      L = np.zeros((N, d), dtype=int)
      k = enumerateLevelsRecursive(n, d, 0, L, 0)
      assert k == N
    else:
      L = np.zeros((0, d), dtype=int)
  else:
    L = enumerateLevels(n-d, d, includeZero=True) + 1
  
  return L

def getRegularSparseGridCTLevels(n, d):
  N = sum([numberOfLevels(n-q, d) for q in range(d)])
  L = np.zeros((N, d), dtype=int)
  k = 0
  
  for q in range(d):
    curL = enumerateLevels(n-q, d)
    L[k:k+curL.shape[0],:] = curL
    k += curL.shape[0]
  
  return L

def getRegularSparseGridPoints(n, d):
  L = getRegularSparseGridCTLevels(n, d)
  X = np.unique(np.vstack([getFullGrid(l) for l in L]), axis=0)
  return X

class AbstractInterpolatorEvaluatorFullGrid(abc.ABC):
  @abc.abstractmethod
  def __call__(l, fX, dfX, XX): pass

class InterpolatorEvaluatorFullGrid(
    AbstractInterpolatorEvaluatorFullGrid):
  def __init__(self, basis):
    self.basis = basis
  
  def __call__(self, l, fX, dfX, XX):
    YY = interpolateEvaluateFullGrid(self.basis, l, fX, XX)
    return YY

class InterpolatorEvaluatorBHCombinationFullGrid(
    AbstractInterpolatorEvaluatorFullGrid):
  def __init__(self, basis):
    self.basis = basis
  
  def __call__(self, l, fX, dfX, XX):
    YY = interpolateEvaluateBHCombinationFullGrid(self.basis, l, fX, dfX, XX)
    return YY

class InterpolatorEvaluatorDerivativeCombinationFullGrid(
    AbstractInterpolatorEvaluatorFullGrid):
  def __init__(self, basis, derivatives="mixed"):
    self.basis = basis
    self.derivatives = derivatives
  
  def __call__(self, l, fX, dfX, XX):
    YY = interpolateEvaluateDerivativeCombinationFullGrid(
        self.basis, l, fX, dfX, XX, derivatives=self.derivatives)
    return YY

def _interpEvalCTLevel(interpEvalFullGridFcn, n, X, fX, dfX, XX, l):
  d = X.shape[1]
  q = n - np.sum(l)
  coeff = (-1)**q * scipy.special.comb(d-1, q, exact=True)
  
  curX = getFullGrid(l)
  K = in1d_index(curX, X)
  curX = X[K,:]
  J = rowSortIndices(curX)
  K = K[J]
  curX = X[K,:]
  
  curShape = [len(getFullGridPoints1D(l1D)) for l1D in l]
  curFX = np.reshape(fX[K], curShape)
  
  if isinstance(dfX, np.ndarray):
    curDfX = np.reshape(dfX[K,:], curShape + [dfX.shape[1]])
  elif isinstance(dfX, dict):
    curDfX = {key : np.reshape(dfX[key][K], curShape) for key in dfX}
  else:
    raise ValueError("Unknown dfX type")
  
  YY = coeff * interpEvalFullGridFcn(l, curFX, curDfX, XX)
  return YY

def interpolateEvaluateCTCombination(
      interpEvalFullGridFcn, n, X, fX, dfX, XX):
  d = X.shape[1]
  L = getRegularSparseGridCTLevels(n, d)
  args = [interpEvalFullGridFcn, n, X, fX, dfX, XX]
  
  #YYs = [_interpEvalCTLevel(*args, l) for l in L]
  
  with multiprocessing.Pool() as pool:
    YYs = pool.map(functools.partial(_interpEvalCTLevel, *args), L)
  
  YY = sum(YYs)
  return YY
