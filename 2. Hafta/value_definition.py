import graphviz
import uuid

class Value:
    def __init__(self, data, prev:set = None, op:bool = None): # 0 toplama 1 çarpma
        self.data = data
        self.prev = prev
        self.grad = 0
        self.op = op
        self.label = ''
        self.id = str(id(self))
    def __repr__(self):
        return f'Value({self.data}, {self.op}, {self.id})'
    def __add__(self, target):
        return Value(self.data + target.data,(self,target), 0)
    def __mul__(self,target):
        return Value(self.data * target.data,(self,target), 1)

from graphviz import Digraph

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
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }"
                  % (n.label, n.data, n.grad),
            shape='record'
        )
        if n.op is not None:
            op = '*' if n.op else '+'
            op_uid = uid + op
            dot.node(
                name=op_uid,
                label=op
            )
            dot.edge(op_uid, uid)

    for n1, n2 in edges:
        op = '*' if n2.op else '+'
        op_uid = str(id(n2)) + op
        dot.edge(str(id(n1)), op_uid)

    return dot


a = Value(2)
b = Value(8)
c=a+b
d=b*c

draw_dot(d).render('graph', view=True)