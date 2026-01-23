import random

class Neuron:
    def __init__(self, no_of_inputs:int):
        self.no_of_inputs=no_of_inputs
        self.weights=self.get_random_weights()
        self.bias=0
    
    def neuron_output(self, inputs:list): #get the ouput for a single neuron
        output=0
        for i in range (0, len(inputs)):
            output+=self.weights[i]*inputs[i]
        output+=self.bias
        return output
    
    def get_random_weights(self): #assign random weights initally
        weights=[]
        for i in range(0, self.no_of_inputs):
            weights.append(random.uniform(-0.2, 0.2))
        return weights
    

    

class Network:
    def __init__(self, input_neurons:int, output_neurons:int):
        self.input_neurons=input_neurons
        self.output_neurons=output_neurons
        self.hidden_layer_1=self.create_hidden_layer_1()
        self.hidden_layer_2=self.create_hidden_layer_2()
        self.output_layer=self.create_output_layer()
        
    def relu(self, input):
        output=max(0,input)
        return output
    
    def relu_derivative(raw_output):
        derivative=0 if (raw_output<=0) else 1
        return derivative
    
    def create_hidden_layer_1(self): #create first hidden layer with 64 neurons
        hidden_layer_1=[]
        for i in range (0,64):
            hidden_layer_1.append(Neuron(self.input_neurons))
        return hidden_layer_1
    
    def create_hidden_layer_2(self): #create second hidden layer with 64 neurons
        hidden_layer_2=[]
        for i in range(0, 64):
            hidden_layer_2.append(Neuron(64))
        return hidden_layer_2
    
    def create_output_layer(self):
        output_layer=[]
        for i in range(0, self.output_neurons):
            output_layer.append(Neuron(64)) #create output layer
        return output_layer

    def compute(self, input_list:list):
        if len(input_list)!=self.input_neurons: #fail safe
            print("Invalid Input")
            return
        
        global hidden_layer_1_outputs_raw
        hidden_layer_1_outputs_raw=[]

        for neuron in self.hidden_layer_1: # get raw outputs for firsts hidden layer
            output=neuron.neuron_output(input_list)
            hidden_layer_1_outputs_raw.append(output)
        
        global hidden_layer_1_outputs_activated
        hidden_layer_1_outputs_activated=[self.relu(i) for i in hidden_layer_1_outputs_raw] # get activated outputs for first hidden layer

        global hidden_layer_2_outputs_raw
        hidden_layer_2_outputs_raw=[]

        for neuron in self.hidden_layer_2: # get raw outputs for second hidden layer
            output=neuron.neuron_output(hidden_layer_1_outputs_activated)
            hidden_layer_2_outputs_raw.append(output)

        global hidden_layer_2_outputs_activated
        hidden_layer_2_outputs_activated=[self.relu(i) for i in hidden_layer_2_outputs_raw] # get activated outputs for second hidden layer

        global final_outputs_raw
        final_outputs_raw=[]

        for neuron in self.output_layer: #get final raw outputs
            output=neuron.neuron_output(hidden_layer_2_outputs_activated)
            final_outputs_raw.append(output)
        
        global final_outputs_activated
        final_outputs_activated=[self.relu(i) for i in final_outputs_raw]
        
        print(final_outputs_activated)
        return final_outputs_activated

    def loss(self, actual_result, predicted_result):
        loss_list=[((actual_result[i]-predicted_result[i])**2) for i in range(0, len(actual_result))]
        return loss_list
    
    def backprop_output_neurons(self, output_neurons, layer_inputs, raw_outputs, activated_outputs, actual_outputs, lr_weight=0.001, lr_bias=0.01):
        for i in range(0, len(output_neurons)):
            dl_dz=2*(activated_outputs[i]-actual_outputs[i])*self.relu_derivative(raw_outputs[i])
            for j in range(0, len(output_neurons[i].weights)):
                dz_dw=layer_inputs[j]
                dl_dw=dl_dz*dz_dw
                output_neurons[i].weights[j]-=lr_weight*dl_dw
            output_neurons[i].bias-=lr_bias*dl_dz