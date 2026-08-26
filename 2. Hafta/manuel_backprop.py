from graphviz import Digraph
from math import tanh

class Value:
    def __init__(self, data, prev=None, op='', label=''):
        self.data = data
        self.prev = prev
        self.grad = 0
        self.op = op
        self.label = label
        self.id = str(id(self))

    def __repr__(self):
        return f'Value({self.data}, {self.op}, {self.id})'

    def __add__(self, target):
        return Value(self.data + target.data, (self, target), '+')

    def __mul__(self, target):
        return Value(self.data * target.data, (self, target), '*')
    def tanh(self):
        return Value(tanh(self.data), (self,), 'tanh')

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


w1 = Value(5)
w2 = Value(1)

x1 = Value(-1)
x2 = Value(4)

bias = Value(3)

a1 = x1*w1
a2 =x2*w2
a = a1+a2
b = a+bias
c = b.tanh() 
c.grad = 1
b.grad = 1 - tanh(b.data)**2
a.grad = 1*b.grad
bias.grad = 1*b.grad
a1.grad = 1*a.grad
a2.grad = 1*a.grad
x1.grad = w1.data*a1.grad
w1.grad = x1.data*a1.grad
x2.grad = w2.data*a2.grad
w2.grad = x2.data*a2.grad



draw_dot(c).render('graph', view=True)

