from math import pow
import copy

input_activations = [0.25,0.5,0.3]
weights = [
    [0.1,-0.15,0.2],
    [2,0.65,-0.3],
    [1.4,-0.12,-1.1]
    ]
biases = [2,0,3]
output_activations = [0,0,0]
truth_samples = [0,1,0]
gradient = list()

def compute_neural_network(input, weight, bias):
    output = list()
    for i in range(len(input)):
        sum = float()
        for k,l in enumerate(input):
            sum+=weight[i][k]*l
        output.append(max(sum+bias[i],0))
    return output

def loss_function(output,truth):
    loss = float()
    for i,j in enumerate(output):
            loss+=pow(j-truth[i], 2)
    return loss

def numerical_derivative(i,j,weight):
    w = copy.deepcopy(weight)
    a = loss_function(compute_neural_network(input_activations,w,biases), truth_samples)
    k=0.00001
    w[i][j]+=k
    b = loss_function(compute_neural_network(input_activations,w,biases), truth_samples)
    return (b-a)/k


def compute_gradient(w):
    gradient = list()
    for i,j in enumerate(w):  
        for k,_ in enumerate(j):
            gradient.append(numerical_derivative(i,k,w))
    return gradient
weights_new = copy.deepcopy(weights)

for i in range(1,25):
    gradient = compute_gradient(weights_new)
    step_size = 0.05
    for k,j in enumerate(gradient):
        weights_new[k//3][k%3] = weights_new[k//3][k%3]-step_size*j
    for l in biases: l = l-step_size
    loss = loss_function(compute_neural_network(input_activations,weights_new,biases), truth_samples)
    
    print(loss)