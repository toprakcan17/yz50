import random

from graphviz import Digraph
from math import tanh, exp
import torch

 
class Value:
    def __init__(self, data, prev=(), op='', label=''):
        self.data = data
        self.prev = prev
        self.grad = 0
        self.op = op
        self._backward = lambda: None
        self.label = label
        self.id = str(id(self))
    def __repr__(self):
        return f'Value({self.data}, {self.op}, {self.id})'

    def __add__(self, target):
        if type(target) == int or type(target) == float: target = Value(target)
        result = Value(self.data+target.data, (self, target), '+')
        def backward():
           self.grad += result.grad
           target.grad += result.grad
        result._backward = backward
        return result 
    def __sub__(self, other):
      return self + (-1 * other)
    def __rsub__(self, other):
      return other + (-1 * self)
    def __mul__(self, target):
        if type(target) == int or type(target) == float: target = Value(target)
        
        result = Value(self.data*target.data, (self, target), '*')
        def backward():
          self.grad += target.data*result.grad
          target.grad += self.data * result.grad
        result._backward = backward
        return result
    def __rmul__ (self, other):
       return self*other
    def __pow__(self,n):
      result = Value(self.data**n, (self,),f'**{n}')
      def backward():
        self.grad += n*(self.data**(n-1))*result.grad
      result._backward = backward
      return result
    def __truediv__(self,other):
      return self*(other**-1)
    def exp(self):
       result = Value(exp(self.data), (self,), 'exp')
       def backward():
          self.grad += result.grad*exp(self.data)
       result._backward = backward
       return result
    def tanh(self):
        result =  Value(tanh(self.data), (self,), 'tanh')
        def backward():
           self.grad += (1 - tanh(self.data)**2) * result.grad
        result._backward = backward
        return result
    def backward(self):
      topological_order = []
      visited = set()
      def topological_sort(node):
        if node in visited: return
        visited.add(node)
        for j in node.prev:
            topological_sort(j)
        topological_order.append(node)
      topological_sort(self)

      for i in topological_order[::-1]:
        i._backward()


class Neuron:
  def __init__(self, n):
    self.weights = [Value(random.uniform(-1,1)) for _ in range(n)] 
    self.bias = Value(random.uniform(-1,1))
  def __call__(self, inputs):
    activation = Value(0)
    for i,j in enumerate(inputs):
      activation += j*self.weights[i]
    activation += self.bias
    return (activation).tanh()
  def parameters(self):
    return self.weights + [self.bias]

class Layer:
  def __init__(self, input_size, output_size):
    self.neurons = [Neuron(input_size) for _ in range(output_size)]
  def __call__(self,inputs):
    activations = []
    for i in self.neurons:
      activations.append(i(inputs))    
    return activations
  def parameters(self):
    return [param for neuron in self.neurons for param in neuron.parameters()]
class MLP:
  def __init__(self, input_size, output_sizes):
    self.layers = []
    for i in range(len(output_sizes)):
      if i == 0:
        self.layers.append(Layer(input_size, output_sizes[i]))
      else:
        self.layers.append(Layer(output_sizes[i-1], output_sizes[i]))
  def __call__(self, outputs):
    for layer in self.layers:
      outputs = layer(outputs)
    return outputs
  def parameters(self):
     return [parameter for layer in self.layers for parameter in layer.parameters()]
  
a = MLP(3, [3,3,1])

training_data = ([
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
], [1.0, -1.0, -1.0, 1.0])

def loss_function(train):
  sse = Value(0)
  for i,j in enumerate(train[0]):
    pred = a(j)[0]
    sse += (pred-Value(train[1][i]))**2
  return sse

def gradient_descent(abc, step_size):
  for parameter in abc.parameters():
      parameter.data-=parameter.grad*step_size
  for parameter in abc.parameters():
    parameter.grad = 0
step_count = 150
for i in range(step_count):
   loss = loss_function(training_data)
   loss.grad = 1
   loss.backward()
   print(f"Step {i+1}: {loss.data:.4f}")
   gradient_descent(a, 0.5)
