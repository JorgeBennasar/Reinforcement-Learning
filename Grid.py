import numpy as np

# states = ["11" = 0,"12" = 1,"13" = 2,"21" = 3,"22" = 4,"23" = 5,"31" = 6,"32" = 7,"33" = 8] 

# Left: 0-8, Right: 9-17, Up: 18-26, Down: 27-35

transitionMatrix = np.array([[1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0],
                             [0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],
                             [0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],
                             [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0,0],
                             [0,0,0,0,0,0,0,1,0],[0,1,0,0,0,0,0,0,0],
                             [0,0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],
                             [0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],
                             [0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1],
                             [1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],
                             [0,0,1,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0],
                             [0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],
                             [0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],
                             [0,0,0,0,0,1,0,0,0],[0,0,0,1,0,0,0,0,0],
                             [0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],
                             [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0,0],
                             [0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]])

rewardVector = np.array([0,-1,-1,-1,-1,-1,-1,-1,-1])
valueState = np.array([0,0,0,0,0,0,0,0,0])
valueStateAux1 = np.array([0,0,0,0,0,0,0,0,0])
discountFactor = 1
delta = 1
optimalPolicy = np.array([0,0,0,0,0,0,0,0,0])

while delta > 0.5:
    
    delta = 0
    
    for i in range(9):
        
        valueStateAux1[i] = valueState[i]
        optimalPolicy[i] = 0
        valueStateAux3 = -10
        
        for j in range(4):
            
            valueStateAux2 = 0
            
            for k in range(9):
                
                valueStateAux2 = valueStateAux2 + transitionMatrix[j*9+i,k] * \
                (rewardVector[i] + discountFactor * valueStateAux1[k])
            
            if valueStateAux2 >= valueStateAux3:
                
                valueStateAux3 = valueStateAux2
                optimalPolicy[i] = j
                
        valueState[i] = valueStateAux3
                
        if delta < abs(valueState[i] - valueStateAux1[i]):
            
            delta = abs(valueState[i] - valueStateAux1[i])
            
            
    
                
        
                
  