from math import pow

input_activations = [0.25,0.5,0.3]
weights = [
    [0.1,-0.15,0.2],
    [2,0.65,-0.3],
    [1.4,-0.12,-1.1]
    ]
biases = [1,-2,-0.7]
output_activations = [0,0,0]
truth_samples = [0,1,0]

def compute_neural_network(input, output, weight, bias):
    for i,j in enumerate(output):
        sum = float()
        for k,l in enumerate(input):
            sum+=weight[i][k]*l
        output_activations[i]=max(sum+bias[i],0)
    return output

def loss_function(output,truth):
    loss = float()
    for i,j in enumerate(output):
            loss+=pow(j-truth[i], 2)
    return loss

for i in range(1,15):
    print("Loss: {}".format(loss_function(compute_neural_network(input_activations,output_activations,weights,biases), truth_samples)))
    weights[1][1]+=0.5 # Değiştirilecek parametreyi ve değişim miktarı buradan ayarlanıyor


