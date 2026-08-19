import math

def sigmoid(x):
    return 1/(1+math.exp(-x))



input_activations = [0.2,0.5,0.3]
weights = [
    [0.1,-0.15,0.2],
    [0.2,0.4,-0.3],
    [1.4,-0.12,-1.1]
    ]
biases = [1,-2,-0.7]
output_activations = [0,0,0]
truth_samples = [0,1,0]

for i,j in enumerate(input_activations):
    sum = int()
    for k in weights[i]: 
        sum+=j*k
    sum+=biases[i]
    output_activations[i]=sigmoid(sum)


loss = int()
for i,_ in enumerate(output_activations):
    loss+=math.pow(output_activations[i]-truth_samples[i], 2)

print(loss)