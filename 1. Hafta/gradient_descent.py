from math import pow
import copy

input_activations = [0.25,0.5,0.3]
weights = [
    [0.1,-0.15,0.2],
    [2,0.65,-0.3],
    [1.4,-0.12,-1.1]
    ]
biases = [2,0,3]
output_activations = list()
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

def numerical_derivative_neuron(i,j,weight):
    w = copy.deepcopy(weight)
    a = loss_function(compute_neural_network(input_activations,w,biases), truth_samples)
    k=0.00001
    w[i][j]+=k
    b = loss_function(compute_neural_network(input_activations,w,biases), truth_samples)
    return (b-a)/k

def numerical_derivative_bias(i,weight):
    bias = copy.deepcopy(biases)
    a = loss_function(compute_neural_network(input_activations,weight,bias), truth_samples)
    k=0.00001
    bias[i]+=k
    b = loss_function(compute_neural_network(input_activations,weight,bias), truth_samples)
    return (b-a)/k

def compute_gradient(w):
    gradient = list()
    for i,j in enumerate(w):  
        for k,_ in enumerate(j):
            gradient.append(numerical_derivative_neuron(i,k,w))
    return gradient

def compute_bias_gradient(weights_new):
    gradient = list()
    for i in range(len(biases)):
        gradient.append(numerical_derivative_bias(i, weights_new))
    return gradient

weights_new = copy.deepcopy(weights)

for i in range(1,25):
    gradient = compute_gradient(weights_new)
    bias_gradient = compute_bias_gradient(weights_new)
    step_size = 0.05
    for k,j in enumerate(gradient):
        weights_new[k//3][k%3] = weights_new[k//3][k%3]-step_size*j
    for l,m in enumerate(bias_gradient): biases[l] = biases[l]-step_size*m
    loss = loss_function(compute_neural_network(input_activations,weights_new,biases), truth_samples)
    
    print("{}. adım | Loss: {:.2f}".format(i, loss))
print("Son weights matrisi {}\nSon bias vektörü {}".format(weights_new, biases))