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
        result = Value(self.data+target.data, (self, target), '+')
        def backward():
           self.grad += result.grad
           target.grad += result.grad
        result._backward = backward
        return result 
    def __mul__(self, target):
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
           self.grad = 1 - tanh(self.data)**2 * result.grad
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
      print(topological_order)

      for i in topological_order[::-1]:
        i._backward()
def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      if v.prev:
        for child in v.prev:
            edges.add((child, v))
            build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n.op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n.op, label = n.op)
      # and connect this node to it
      dot.edge(uid + n.op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2.op)

  return dot


x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
bias = Value(6.8813735870195432, label='bias')
a1 = a2 = a = b = c = d = e = Value(0)
def compute():
  global a1, a2, a, b, c, d, e
  a1 = x1*w1
  a2 =x2*w2
  a = a1+a2
  b = a+bias
  b.label = 'b'
  #c = b.tanh() 
  d = ((b.exp()**2)+Value(1))
  d.label = 'd'
  c = ((b.exp()**2)+(Value(-1)))
  c.label = "c"
  e = c/d
  e.grad = 1


method = input("Tercih edilen yontemi giriniz: ('pytorch', 'sayisal turev', 'backward')")
compute()
if method == 'backward':
  e.backward()
  print(x2.grad)
elif method == 'pytorch':
  _x1 = torch.Tensor([2.0]).double()                ; _x1.requires_grad = True
  _x2 = torch.Tensor([0.0]).double()                ; _x2.requires_grad = True
  _w1 = torch.Tensor([-3.0]).double()               ; _w1.requires_grad = True
  _w2 = torch.Tensor([1.0]).double()                ; _w2.requires_grad = True
  _b = torch.Tensor([6.8813735870195432]).double()  ; _b.requires_grad = True
  n = _x1*_w1 + _x2*_w2 + _b
  o = torch.tanh(n)
  o.backward()
  print('---')
  print('x2', _x2.grad.item())
  print('w2', _w2.grad.item())
  print('x1', _x1.grad.item())
  print('w1', _w1.grad.item())
elif method == 'sayisal turev':
  aaa = e.data
  h = 0.0001
  x2.data += h
  compute()
  aaaa = e.data
  grad = (aaaa-aaa)/h
  print(grad)





#draw_dot(e).render('graph', view=True)

