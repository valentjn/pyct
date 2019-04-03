#!/usr/bin/python3

import numpy as np
import sympy as sp



functionStrings = {
  "ackley" : (lambda d: (
      "-20*exp(-({})/(5*sqrt({}))) - exp(1/{}*({})) + 20 + exp(1)".format(
          "+".join(["x{}".format(t) for t in range(d)]), d, d,
          "+".join(["cos(2*pi*x{})".format(t) for t in range(d)])),
      # TODO: changed domain!
      np.array([1*np.ones((d,)), 2*np.ones((d,))]))),
  "branin" : (
      "(x1 - 51/10*x0^2/(4*pi^2) + 5*x0/pi - 6)^2 + "
      "10*(1-1/(8*pi))*cos(x0) + 10", np.array([[-5, 0], [10, 15]])),
}



def getFunction(functionType, d):
  return getFunctionDerivative(functionType, d, np.zeros((d,)))

def getFunctionDerivative(functionType, d, order):
  variables = [sp.var("x{}".format(t)) for t in range(d)]
  functionString, bounds = (functionStrings[functionType](d)
      if callable(functionStrings[functionType]) else
      functionStrings[functionType])
  functionExpression = sp.sympify(functionString)
  innerDerivative = 1
  
  for t in range(d):
    if order[t] > 0:
      functionExpression = sp.diff(functionExpression, variables[t], order[t])
      innerDerivative *= (bounds[1,t] - bounds[0,t])**order[t]
  
  lambdaFcn = sp.utilities.lambdify(variables, functionExpression)
  resultFcn = (lambda X: innerDerivative * lambdaFcn(
      *(bounds[0] + X * (bounds[1] - bounds[0])).T))
  
  return resultFcn

def getFunctionFirstDerivatives(functionType, d):
  orders = d * [list(range(2))]
  orders = np.meshgrid(*orders, indexing="ij")
  orders = np.column_stack([order.flatten() for order in orders])
  orders = orders.tolist()
  orders = [tuple(order) for order in orders if sum(order) > 0]
  derivativeFcns = {order : getFunctionDerivative(functionType, d, order)
                    for order in orders}
  resultFcn = (lambda X: {order : derivativeFcns[order](X)
                          for order in orders})
  return resultFcn

def getFunctionGradient(functionType, d):
  derivativeFcns = [getFunctionDerivative(
      functionType, d, np.eye(1, d, t, dtype=int).flatten())
      for t in range(d)]
  resultFcn = (lambda X: np.column_stack(
      [derivativeFcn(X) for derivativeFcn in derivativeFcns]))
  
  return resultFcn
