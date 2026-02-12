import numpy as np

def mcp_activation(net_input, threshold):
    
    return 1 if net_input >= threshold else 0

def mcp_neuron(inputs, weights, threshold):
   
    inputs = np.array(inputs)
    weights = np.array(weights)
    
    net_input = np.dot(inputs, weights)
    
    output = mcp_activation(net_input, threshold)
    
    return net_input, output

def implement_or_gate():
    
    print("\n--- Implementing Logical OR Gate ---")
    
    weights = [1, 1]  
    threshold = 1     
    
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1)
    ]
    
    print(f"Weights: {weights}, Threshold: {threshold}")
    print("Input (A, B) | Net Input | Output | Expected")
    print("-" * 38)
    
    for inputs, expected in test_cases:
        net_input, output = mcp_neuron(inputs, weights, threshold)
        print(f"  {inputs[0]}, {inputs[1]}    |   {net_input}      |    {output}   |   {expected}")

def implement_not_gate():
    
    print("\n--- Implementing Logical NOT Gate ---")
    
    weights = [-1]    
    threshold = 0     
    
    test_cases = [
        ([0], 1),
        ([1], 0)
    ]
    
    print(f"Weights: {weights}, Threshold: {threshold}")
    print("Input (A) | Net Input | Output | Expected")
    print("-" * 38)
    
    for inputs, expected in test_cases:
        net_input, output = mcp_neuron(inputs, weights, threshold)
        print(f"    {inputs[0]}   |    {net_input}      |    {output}   |   {expected}")


implement_or_gate()
implement_not_gate()