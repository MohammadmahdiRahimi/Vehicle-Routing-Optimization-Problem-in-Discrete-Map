import pandas as pd
from mip import Model, xsum, minimize, BINARY
#load data
df = pd.read_csv("Data.csv", sep=';')
num = len(df)
weights = [[1000 for j in range(num)]for i in range(num)]
names = list(df['Place_name'])
Latitude = list(df['Latitude'])
Longitude = list(df['Longitude'])
neighbors = [[int(x) for x in arr.split(',')] for arr in df['Neighbors_indice']]
diatance = [[int(x) for x in arr.split(',')] for arr in df['Neighbor_weight']]
#create model
model = Model()
nodes = set(range(num))
#create matrix of weights
for temp_index in range(num):
    for temp_neighbor in range(len(neighbors[temp_index-1])):
        weights[temp_index-1][neighbors[temp_index-1][temp_neighbor]-1] = diatance[temp_index-1][temp_neighbor]
#fix model
x = [[model.add_var(var_type=BINARY) for j in nodes] for i in nodes]
model.objective = minimize(xsum(weights[i][j] * x[i][j] for i in nodes for j in nodes))
#show places name
print('Please select index of places.')
print('Places:')
for i in range(num):
    print(i+1, names[i])
#get index of places from user  
print('Enter starting point:')
start = int(input())-1
print('Enter destination:')
end = int(input())-1

#subject to:
for i in nodes - {start, end}:
    model += xsum(x[i][j] for j in nodes) == \
     xsum(x[j][i] for j in nodes)
model += xsum(x[start][j] for j in nodes) == xsum(x[j][start] for j in nodes) + 1
model += xsum(x[end][j] for j in nodes) == xsum(x[j][end] for j in nodes) - 1
model.optimize()
#show the answer
print('               Index              Name                Latitude               Longitude')
print('Starting point: ', start+1, '     ', names[start], '            ', Latitude[start], '         ', Longitude[start])
print('destination:    ', end+1, '     ', names[end], '            ', Latitude[end], '         ', Longitude[end])
if model.num_solutions:
    print('Best way is found:')
    print('Index in order: ', start+1, end = '')
    temp_node = start
    while(temp_node != end):
        for i in range(num):
            if(x[temp_node][i].x):
                temp_node = i
                break
        print(' ->', temp_node+1, end = '')
    print('\n')
    print('Names in order: ', names[start], end = '')
    temp_node = start
    while(temp_node != end):
        for i in range(num):
            if(x[temp_node][i].x):
                temp_node = i
                break
        print(' ->', names[temp_node], end = '')
    print('\n cost = ', model.objective_value)
    