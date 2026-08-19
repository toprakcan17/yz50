import math

def sigmoid(x):
    return 1/(1+math.exp(-x))



input_activations = [0.25,0.5,0.3]
weights = [
    [0.1,-0.15,0.2],
    [0.2,0.4,-0.3],  
    [1.4,-0.12,-1.1]
    ]
biases = [1,-2,-0.7]
output_activations = [0,0,0]

for i,j in enumerate(output_activations):
    sum = float()
    for k,l in enumerate(input_activations):
        sum+=weights[i][k]*l
    output_activations[i]=sigmoid(sum+biases[i])

print(output_activations)
