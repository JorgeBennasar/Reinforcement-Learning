import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# states = {'1' = 0,'2' = 1,...,'11' = 10}
# actions = {'east' = 0,'west' = 1,'north' = 2,'south' = 3}

# The transition matrix is defined as:
# transitionMatrix[actions][current_state,next_state]

transitionMatrix = np.array([[[0,1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,0,1]],
                              [[1,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,0,0,1]],
                              [[1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0],
                              [1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,0,0,1,0,0]],
                              [[0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,0,1]]])

# Universal algorithm (every CID)

CID = [0,1,7,7,9,5,1,8]

x = CID[5]
y = CID[6]
z = CID[7]

my_j = ((z + 1) % 3) + 1
my_p = 0.25 + 0.5 * (x / 10)
my_discountFactor = 0.2 + 0.5 * (y / 10)
           
# The rewards are received when ENTERING the state assigned to them

selectReward = [0,0,0,0,0,0,0,0,0,0]
selectReward[(my_j - 1)] = 1

rewardVector = [-1+selectReward[0]*11,-1+selectReward[1]*11,
                -1+selectReward[2]*11,-1+selectReward[3]*11,
                -1+selectReward[4]*11,-1+selectReward[5]*11,
                -1+selectReward[6]*11,-1+selectReward[7]*11,
                -1+selectReward[8]*11,-1+selectReward[9]*11,-100]

discountFactor = 0

valueState9 = []
myValueState = [0,0,0,0,0,0,0,0,0,0,0]
myOptimalPolicy = [0,0,0,0,0,0,0,0,0,0,0]

theta = 0.05 # for evaluation of different 'p' and 'discountFactor'
gamma = 0.001 # threshold for 'delta'

while discountFactor < 1 + theta:
    
    p = 0
     
    while p < 1 + theta:
        
        q = (1 - p) / 3
        
        # Value iteration algorithm
    
        # Values of states j and '11' will always be 0, due to the fact that they
        # are terminal states and the reward is collected when we ENTER them
        
        valueState = [0,0,0,0,0,0,0,0,0,0,0]
        
        # 'delta' is the threshold to stop the algorithm
        
        delta = 1
        
        # 'optimalPolicy' is a vector which contain the optimal action for each state
        # (see 'states' and 'actions')
        # I initialize to be always optimal to move east
        
        optimalPolicy = [0,0,0,0,0,0,0,0,0,0,0]
        valueStateAux = [0,0,0,0,0,0,0,0,0,0,0]
    
        while delta > gamma:
            
            delta = 0
            
            for i in range(11):
                
                if i != (my_j - 1) and i != 10: # Values of states j and '11' are known
                    
                    valueStateAux[i] = valueState[i]
                    x = -1000 # Initial condition (ideally -inf)
                    
                    for j in range(4):
                        
                        y = 0
                        
                        # I use auxiliary parameters a, b and c to compute the
                        # stochasticity of actions
                        
                        if j == 0: a = 1; b = 2; c = 3
                        elif j == 1: a = 0; b = 2; c = 3
                        elif j == 2: a = 0; b = 1; c = 3
                        else: a = 0; b = 1; c = 2
                        
                        for k in range(11):
                            
                            y = (y + (p * transitionMatrix[j][i,k] *
                                (rewardVector[k] + discountFactor * valueStateAux[k]))
                                + (q * transitionMatrix[a][i,k] *
                                (rewardVector[k] + discountFactor * valueStateAux[k]))
                                + (q * transitionMatrix[b][i,k] *
                                (rewardVector[k] + discountFactor * valueStateAux[k]))
                                + (q * transitionMatrix[c][i,k] * 
                                (rewardVector[k] + discountFactor * valueStateAux[k])))
                        
                        if y >= x: x = y; optimalPolicy[i] = j
                            
                    valueState[i] = x

                dif = abs(valueState[i] - valueStateAux[i])
                        
                if delta < dif:
                    
                    delta = dif

        if (int((p + 0.001)*100)/100 == my_p) and \
            (int((discountFactor + 0.001)*100)/100 == my_discountFactor):
                
            # Because of the problem of checking equality with floats
            
            myValueState = valueState
            myOptimalPolicy = optimalPolicy
                                        
        valueState9.append(valueState[8])
        
        p = p + theta 
      
    discountFactor = discountFactor + theta

# Plots (for theta = 0.05):
    
X = [] # discountFactor
Y = [] # p
Z = valueState9

for i in range(21):
    
    for j in range(21):
        
        X.append(i/20)
        Y.append(j/20)
        
font = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 14,
        }

Axes3D = Axes3D
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(X, Y, Z, c = Z, cmap = 'Blues');
plt.title('value of state 9', fontdict=font)
plt.xlabel('discount factor', fontdict=font)
plt.ylabel('probability p', fontdict=font)

valueState9_p = []
valueState9_d = []

for i in range(441):
    
    if X[i] == my_discountFactor:
        
        valueState9_p.append(Z[i])
        
    if Y[i] == my_p:
        
        valueState9_d.append(Z[i])
        
font = {'family': 'serif',
        'color':  'darkblue',
        'weight': 'normal',
        'size': 14,
        }
    
plt.figure(figsize=(6,6))
plt.plot([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
          0.75,0.8,0.85,0.9,0.95,1],valueState9_p,'darkblue')
plt.title('relationship p - value of state 9', fontdict=font)
plt.xlabel('probability p', fontdict=font)
plt.ylabel('value of state 9', fontdict=font)

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 14,
        }
    
plt.figure(figsize=(6,6))
plt.plot([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
          0.75,0.8,0.85,0.9,0.95,1],valueState9_d,'darkred')
plt.title('relationship discount factor - value of state 9', fontdict=font)
plt.xlabel('discount factor', fontdict=font)
plt.ylabel('value of state 9', fontdict=font)
