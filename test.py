import numpy as np
value_list = np.array([1,2,3,4,5])
output=[]
total = 1
for value in value_list:
    total*=value

for value in value_list:
    output.append(total-value)

print(output)