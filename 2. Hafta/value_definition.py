from graphviz import Digraph

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


def trace(root):
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
        dot.node(uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')

        if n.op:
            op_uid = uid + n.op
            dot.node(op_uid, label=n.op)
            dot.edge(op_uid, uid)

    for n1, n2 in edges:
        op_uid = str(id(n2)) + n2.op
        dot.edge(str(id(n1)), op_uid)

    return dot


a = Value(2)
b = Value(-3)
e = Value(10)
c = a * b
d = c + e
f = Value(-2)
L = f * d

draw_dot(L).render('graph', view=True)