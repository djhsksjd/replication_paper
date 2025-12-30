import torch.nn as nn
import torch



user_input = input("input the digit seqerated by the space:  ")
print(user_input)
input_num = [int(num) for num in user_input.split()]
input_tensor = torch.tensor(input_num, dtype = torch.float32)

model = nn.Sequential(
    nn.Linear(len(input_num),1)
)

output = model(input_tensor)
print("the output is ", output)