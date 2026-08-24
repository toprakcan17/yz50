class Value:
    def __init__(self, data, prev:set = set(), op:bool = None): # 0 toplama 1 çarpma
        self.data = data
        self.prev = prev
        self.op = op
    def __str__(self):
        return f'Value({self.data} {self.prev})'
    def __add__(self, target):
        return Value(self.data + target.data,(self.data,target.data), 0)
    def __mul__(self,target):
        return Value(self.data * target.data,(self.data,target.data), 1)
