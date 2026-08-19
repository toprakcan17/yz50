import math

def sigmoid(x):
    return 1/(1+math.exp(-x))


input_activation = 0
weight = 0
bias = 0

output_activation = sigmoid(input_activation * weight + bias)

print(output_activation)