import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

n_episodes = 100000

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
        }

class Easy21:

    def __init__(self, max_length=1000):
        self.max_length = max_length

    def reset(self):
        self.player_first_card_val = np.random.choice(10) + 1
        self.dealer_first_card_val = np.random.choice(10) + 1

        self.player_sum = self.player_first_card_val
        self.dealer_sum = self.dealer_first_card_val

        self.state = [self.dealer_first_card_val, self.player_sum]

        self.player_goes_bust = False
        self.dealer_goes_bust = False

        self.terminal = False

        return self.state

    def step(self, action):
        # action 1: hit   0: stick
        # color: 1: black   -1: red
        r = 0

        if action == 1:
            self.player_card_val = np.random.choice(10) + 1
            self.player_card_col = np.random.choice([-1, 1], p=[1./3., 2./3.])

            self.player_sum += (self.player_card_val * self.player_card_col)
            self.player_goes_bust = self.check_go_bust(self.player_sum)

            if self.player_goes_bust == 1:
                r = -1
                self.terminal = True

        if action == 0:
            self.terminal = True
            
            while self.dealer_sum < 17 and self.dealer_sum > 0:
                self.dealer_card_val = np.random.choice(10) + 1
                self.dealer_card_col = np.random.choice([-1, 1], 
                                                        p=[1./3., 2./3.])
    
                self.dealer_sum += (self.dealer_card_val*self.dealer_card_col)
                self.dealer_goes_bust = self.check_go_bust(self.dealer_sum)

            if self.dealer_goes_bust == 1: r = 1
            else:
                if self.player_sum > self.dealer_sum: r = 1
                elif self.player_sum < self.dealer_sum: r = -1

        if self.terminal: return 'Terminal', r, self.terminal
        else:
            self.state[1] = self.player_sum
            return self.state, r, self.terminal

    def check_go_bust(self, Sum):
        return bool(Sum > 21 or Sum < 1)

#%%    

## Monte Carlo -- one episode
        
def monte_carlo(Q, returns, count_state, count_state_action):

    actions = []
    s = env.reset()
    states = [s]

    while True:
        action_greedy = Q[s[0]-1, s[1]-1, :].argmax()
        count_state[s[0]-1, s[1]-1] += 1
        epsilon = count_constant / float(count_constant + count_state[s[0]-1, 
                                                                      s[1]-1])
        action = np.random.choice([action_greedy, 1 - action_greedy], 
                                  p=[1. - epsilon/2., epsilon/2.])
        actions.append(action)

        s, r, term = env.step(action=action)
    
        if term: break
        else: states.append(s)
        
    for t in range(len(states)):
        count_state_action[states[t][0]-1, states[t][1]-1, actions[t]] += 1
        returns[states[t][0]-1, states[t][1]-1, actions[t]] += r
        
        Q[states[t][0]-1, states[t][1]-1, actions[t]] =\
            returns[states[t][0]-1, states[t][1]-1, actions[t]] /\
            count_state_action[states[t][0]-1, states[t][1]-1, actions[t]]
    
    return Q, returns, count_state, count_state_action, r

## Monte Carlo
    
Q_MC = np.zeros([10, 21, 2]) # Q(s, a)
returns = np.zeros([10, 21, 2]) # empirical first-visit returns
count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
count_state = np.zeros([10, 21], dtype=int) # N(s)
count_constant = 100

env = Easy21()

mean_rewards_MC = []
mean_rewards_sq_MC = []
std_rewards_MC = []
episodes_vector_MC = []

for i_epi in range(n_episodes):
    Q_MC, returns, count_state, count_state_action, r =\
        monte_carlo(Q_MC, returns, count_state, count_state_action)
        
    episodes_vector_MC.append(i_epi+1)
    
    if i_epi == 0: 
        mean_rewards_MC.append(r)
        mean_rewards_sq_MC.append(r**2)
        std_rewards_MC.append(0)
    else:
        mean_rewards_MC.append(mean_rewards_MC[i_epi-1]+(1/(i_epi+1))*\
                               (r-mean_rewards_MC[i_epi-1]))
        mean_rewards_sq_MC.append(mean_rewards_sq_MC[i_epi-1]+(1/(i_epi+1))*\
                                  (r**2-mean_rewards_sq_MC[i_epi-1]))
        std_rewards_MC.append(np.sqrt(mean_rewards_sq_MC[i_epi]-\
                                      mean_rewards_MC[i_epi]**2)) 
    
V_MC = Q_MC.max(axis=2)

# For n_episodes = 200000, n = 50
    # mean = 0.0284063
    # std = 0.00224475

## Monte Carlo -- plot

plt.figure(1)
plt.plot(episodes_vector_MC, mean_rewards_MC, color='blue', label='mean reward')
plt.plot(episodes_vector_MC, std_rewards_MC, color='orange', label='standard deviation')
plt.title('mean reward and standard deviation per episode (MC)', fontdict=font)
plt.xlabel('episode', fontdict=font)
plt.legend()

s1 = np.arange(10)+1
s2 = np.arange(21)+1
ss1, ss2 = np.meshgrid(s1, s2, indexing='ij')

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ss1, ss2, V_MC, cmap=cm.coolwarm)

ax.set_xlabel("dealer's first card", fontdict=font)
ax.set_ylabel("player's sum", fontdict=font)
ax.set_zlabel("state value (MC)", fontdict=font)
plt.yticks([1, 5, 10])
plt.yticks([1, 7, 14, 21])
fig.colorbar(surf, shrink=0.6)
fig.tight_layout()

plt.show()

#%%  

## SARSA -- one episode
        
def SARSA(Q, count_state, count_state_action):

    s = env.reset()
    
    epsilon = count_constant / float(count_constant + count_state[s[0]-1, 
                                                                  s[1]-1])
    action_greedy = Q[s[0]-1, s[1]-1, :].argmax()
    action = np.random.choice([action_greedy, 1 - action_greedy], 
                              p=[1. - epsilon/2., epsilon/2.])

    while True:
        next_s, r, term = env.step(action=action)
        
        count_state_action[s[0]-1, s[1]-1, action] += 1
        alpha = count_constant / float(count_constant + count_state_action
                                       [s[0]-1, s[1]-1, action]**2)
        
        if term: 
            Q[s[0]-1, s[1]-1, action] = (1 - alpha) *\
                Q[s[0]-1, s[1]-1, action] + alpha*r
            break
        
        count_state[s[0]-1, s[1]-1] += 1
        epsilon = count_constant / float(count_constant + count_state[s[0]-1, 
                                                                      s[1]-1])
        action_greedy = Q[next_s[0]-1, next_s[1]-1, :].argmax()
        next_action = np.random.choice([action_greedy, 1 - action_greedy], 
                                       p=[1. - epsilon/2., epsilon/2.])
        
        Q[s[0]-1, s[1]-1, action] = (1 - alpha) * Q[s[0]-1, s[1]-1, action] +\
            alpha * (r + Q[next_s[0]-1, next_s[1]-1, next_action])

        s = next_s
        action = next_action
    
    return Q, count_state, count_state_action, r

## SARSA
    
Q_TD = np.zeros([10, 21, 2]) # Q(s, a)
count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
count_state = np.zeros([10, 21], dtype=int) # N(s)
count_constant = 100

env = Easy21()

mean_rewards_TD = []
mean_rewards_sq_TD = []
std_rewards_TD = []
episodes_vector_TD = []
    
for i_epi in range(n_episodes):
    Q_TD, count_state, count_state_action, r =\
        SARSA(Q_TD, count_state, count_state_action)
            
    episodes_vector_TD.append(i_epi+1)
        
    if i_epi == 0: 
        mean_rewards_TD.append(r)
        mean_rewards_sq_TD.append(r**2)
        std_rewards_TD.append(0)
    else:
        mean_rewards_TD.append(mean_rewards_TD[i_epi-1]+(1/(i_epi+1))*\
                               (r-mean_rewards_TD[i_epi-1]))
        mean_rewards_sq_TD.append(mean_rewards_sq_TD[i_epi-1]+(1/(i_epi+1))*\
                                  (r**2-mean_rewards_sq_TD[i_epi-1]))
        std_rewards_TD.append(np.sqrt(mean_rewards_sq_TD[i_epi]-\
                                      mean_rewards_TD[i_epi]**2))        
        
V_TD = Q_TD.max(axis=2)
        
# For n_episodes = 200000, n_alpha_constant (alpha = 0.05) = n_alpha_var = 50
    # mean_alpha_constant = -0.0087416, mean_alpha_var = -0.006968
    # std_alpha_constant = 0.0020108, std_alpha_var = 0.00201372
    
# For n_episodes = 50000, n_alpha_constant (alpha = 0.2) = n_alpha_var = 30
    # mean_alpha_constant = -0.0575727, mean_alpha_var = -0.039696
    # std_alpha_constant = 0.00496, std_alpha_var = 0.00342

## SARSA -- plot

plt.figure(3)
plt.plot(episodes_vector_TD, mean_rewards_TD, color='blue', label='mean reward')
plt.plot(episodes_vector_TD, std_rewards_TD, color='orange', label='standard deviation')
plt.title('mean reward and standard deviation per episode (SARSA)', fontdict=font)
plt.xlabel('episode', fontdict=font)
plt.legend()

# For n_episodes = 50000, n_alpha_constant (alpha = 0.2) = n_alpha_var = 30

means_alpha_constant_TD = [-0.06690,-0.05808,-0.05930,-0.05282,-0.06140,-0.04894,
                           -0.05568,-0.05376,-0.05518,-0.06566,-0.05636,-0.04796,
                           -0.05852,-0.05402,-0.06252,-0.05334,-0.05516,-0.05806,
                           -0.05672,-0.06000,-0.05534,-0.05268,-0.06254,-0.04842,
                           -0.05834,-0.06656,-0.06412,-0.05842,-0.06220,-0.05818]

means_alpha_var_TD = [-0.04254,-0.03602,-0.03934,-0.03790,-0.03864,-0.03718,
                      -0.04138,-0.03888,-0.04412,-0.03950,-0.03522,-0.04222,
                      -0.03966,-0.03624,-0.04412,-0.04478,-0.04420,-0.03970,
                      -0.04280,-0.03442,-0.03698,-0.04444,-0.04002,-0.03470,
                      -0.04084,-0.03652,-0.04152,-0.04606,-0.03348,-0.03746]

plt.figure(4)
res_1 = stats.probplot(means_alpha_constant_TD, plot=plt)
res_2 = stats.probplot(means_alpha_var_TD, plot=plt)
plt.title('comparison of data with normal distributions (SARSA)', fontdict=font)

mean_alpha_constant_TD = -0.0575727
mean_alpha_var_TD = -0.039696
std_alpha_constant_TD = 0.00496
std_alpha_var_TD = 0.00342

# 95% confidence interval
mean_alpha_constant_error_TD = 1.96*std_alpha_constant_TD
mean_alpha_var_error_TD = 1.96*std_alpha_var_TD

fig, ax = plt.subplots()
ax.bar([0, 1], [mean_alpha_constant_TD, mean_alpha_var_TD], 
       yerr=[mean_alpha_constant_error_TD, mean_alpha_var_error_TD],
       align='center', ecolor='black')
ax.set_ylabel('mean rewards', fontdict=font)
ax.set_title('comparison of mean rewards in SARSA (95% CI)', fontdict=font)
ax.set_xticks([0, 1])
ax.set_xticklabels(['alpha = 0.2', 'alpha = 100/(100+N(s,a)^2)'], fontdict=font)
ax.yaxis.grid(True)

s1 = np.arange(10)+1
s2 = np.arange(21)+1
ss1, ss2 = np.meshgrid(s1, s2, indexing='ij')

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ss1, ss2, V_TD, cmap=cm.coolwarm)

ax.set_xlabel("dealer's first card", fontdict=font)
ax.set_ylabel("player's sum", fontdict=font)
ax.set_zlabel("state value (SARSA)", fontdict=font)
plt.yticks([1, 5, 10])
plt.yticks([1, 7, 14, 21])
fig.colorbar(surf, shrink=0.6)
fig.tight_layout()

plt.show()

#%%  

## Q-Learning -- one episode
        
def Q_Learning(Q, count_state, count_state_action):

    s = env.reset()

    while True:
        action_greedy = Q[s[0]-1, s[1]-1, :].argmax()
        count_state[s[0]-1, s[1]-1] += 1
        epsilon = count_constant / float(count_constant + count_state[s[0]-1, 
                                                                      s[1]-1])
        action = np.random.choice([action_greedy, 1 - action_greedy], 
                                  p=[1. - epsilon/2., epsilon/2.])
        
        count_state_action[s[0]-1, s[1]-1, action] += 1
        alpha = count_constant / float(count_constant + count_state_action
                                       [s[0]-1, s[1]-1, action]**2)
            
        next_s, r, term = env.step(action=action)
        
        if term: 
            Q[s[0]-1, s[1]-1, action] += alpha * (r - 
                 Q[s[0]-1, s[1]-1, action])
            break

        Q[s[0]-1, s[1]-1, action] += alpha * (r + max(Q[next_s[0]-1, 
              next_s[1]-1, :]) - Q[s[0]-1, s[1]-1, action])
    
        s = next_s
    
    return Q, count_state, count_state_action, r

## Q-Learning
    
Q_Q = np.zeros([10, 21, 2]) # Q(s, a)
count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
count_state = np.zeros([10, 21], dtype=int) # N(s)
count_constant = 100

env = Easy21()
        
mean_rewards_Q = []
mean_rewards_sq_Q = []
std_rewards_Q = []
episodes_vector_Q = []
        
for i_epi in range(n_episodes):
            
    Q_Q, count_state, count_state_action, r =\
        Q_Learning(Q_Q, count_state, count_state_action)
                
    episodes_vector_Q.append(i_epi+1)
        
    if i_epi == 0: 
        mean_rewards_Q.append(r)
        mean_rewards_sq_Q.append(r**2)
        std_rewards_Q.append(0)
    else:
        mean_rewards_Q.append(mean_rewards_Q[i_epi-1]+(1/(i_epi+1))*\
                              (r-mean_rewards_Q[i_epi-1]))
        mean_rewards_sq_Q.append(mean_rewards_sq_Q[i_epi-1]+(1/(i_epi+1))*\
                                 (r**2-mean_rewards_sq_Q[i_epi-1]))
        std_rewards_Q.append(np.sqrt(mean_rewards_sq_Q[i_epi]-\
                                     mean_rewards_Q[i_epi]**2))   

V_Q = Q_Q.max(axis=2)
        
# For n_episodes = 200000, n_alpha_constant (alpha = 0.05) = n_alpha_var = 50
    # mean_alpha_constant = 0.0155148, mean_alpha_var = 0.0255174
    # std_alpha_constant = 0.0022207, std_alpha_var = 0.00274868
    
# For n_episodes = 50000, n_alpha_constant (alpha = 0.2) = n_alpha_var = 30
    # mean_alpha_constant = -0.05397, mean_alpha_var = -0.009097
    # std_alpha_constant = 0.00456, std_alpha_var = 0.00482

## Q-Learning -- plot

plt.figure(7)
plt.plot(episodes_vector_Q, mean_rewards_Q, color='blue', label='mean reward')
plt.plot(episodes_vector_Q, std_rewards_Q, color='orange', label='standard deviation')
plt.title('mean reward and standard deviation per episode (Q)', fontdict=font)
plt.xlabel('episode', fontdict=font)
plt.legend()

# For n_episodes = 50000, n_alpha_constant (alpha = 0.2) = n_alpha_var = 30

means_alpha_constant_Q = [-0.05426,-0.05484,-0.05652,-0.04754,-0.05090,-0.04756,
                          -0.05512,-0.04516,-0.06020,-0.04870,-0.04328,-0.05478,
                          -0.05494,-0.05278,-0.06132,-0.05660,-0.05656,-0.05484,
                          -0.05770,-0.05912,-0.05300,-0.05592,-0.05802,-0.05590,
                          -0.04554,-0.05320,-0.05830,-0.05946,-0.05542,-0.05162]

means_alpha_var_Q = [-0.01058,-0.01094,-0.00982,-0.01436,-0.00996,-0.00888,
                     -0.00870,-0.01388,-0.01078,0.00264,-0.00066,-0.01110,
                     -0.00846,-0.00632,-0.01628,-0.00408,-0.02072,-0.00596,
                     -0.00944,-0.01116,-0.01230,-0.00810,-0.00882,-0.01450,
                     -0.00056,-0.01360,-0.00208,-0.00758,-0.00892,-0.00700]

plt.figure(8)
res_1 = stats.probplot(means_alpha_constant_Q, plot=plt)
res_2 = stats.probplot(means_alpha_var_Q, plot=plt)
plt.title('comparison of data with normal distributions (Q)', fontdict=font)

mean_alpha_constant_Q = -0.05397
mean_alpha_var_Q = -0.009097
std_alpha_constant_Q = 0.00456
std_alpha_var_Q = 0.00482

# 95% confidence interval
mean_alpha_constant_error_Q = 1.96*std_alpha_constant_Q
mean_alpha_var_error_Q = 1.96*std_alpha_var_Q

fig, ax = plt.subplots()
ax.bar([0, 1], [mean_alpha_constant_Q, mean_alpha_var_Q], 
       yerr=[mean_alpha_constant_error_Q, mean_alpha_var_error_Q],
       align='center', ecolor='black')
ax.set_ylabel('mean rewards', fontdict=font)
ax.set_title('comparison of mean rewards in Q-Learning (95% CI)', fontdict=font)
ax.set_xticks([0, 1])
ax.set_xticklabels(['alpha = 0.2', 'alpha = 100/(100+N(s,a)^2)'], fontdict=font)
ax.yaxis.grid(True)

s1 = np.arange(10)+1
s2 = np.arange(21)+1
ss1, ss2 = np.meshgrid(s1, s2, indexing='ij')

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(ss1, ss2, V_Q, cmap=cm.coolwarm)

ax.set_xlabel("dealer's first card", fontdict=font)
ax.set_ylabel("player's sum", fontdict=font)
ax.set_zlabel("state value (Q)", fontdict=font)
plt.yticks([1, 5, 10])
plt.yticks([1, 7, 14, 21])
fig.colorbar(surf, shrink=0.6)
fig.tight_layout()

plt.show()

#%%  

# Comparison MC - SARSA - Q

plt.figure(11)
plt.plot(episodes_vector_MC, mean_rewards_MC, 'blue', label='MC')
plt.plot(episodes_vector_TD, mean_rewards_TD, 'orange', label='SARSA')
plt.plot(episodes_vector_Q, mean_rewards_Q, 'red', label='Q')
plt.title('mean reward per episode comparison', fontdict=font)
plt.xlabel('episode', fontdict=font)
plt.ylabel('mean reward', fontdict=font)
plt.legend()

plt.figure(12)
plt.plot(episodes_vector_MC, std_rewards_MC, 'blue', label='MC')
plt.plot(episodes_vector_TD, std_rewards_TD, 'orange', label='SARSA')
plt.plot(episodes_vector_Q, std_rewards_Q, 'red', label='Q')
plt.title('standard deviation of rewards per episode comparison', fontdict=font)
plt.xlabel('episode', fontdict=font)
plt.ylabel('standard deviation', fontdict=font)
plt.legend()

# Taking into account that n_episodes = 50000, n = 30 give results that follow
# a normal distribution, I assume n_episodes = 200000, n = 50 also do
# (central limit theorem)

# For n_episodes = 200000, n = 50

mean_MC = 0.0284063
std_MC = 0.00224475
mean_alpha_var_TD = -0.006968
std_alpha_var_TD = 0.00201372
mean_alpha_var_Q = 0.0255174
std_alpha_var_Q = 0.00274868

# 95% confidence interval
mean_error_MC = 1.96*std_MC
mean_alpha_var_error_TD = 1.96*std_alpha_var_TD
mean_alpha_var_error_Q = 1.96*std_alpha_var_Q

fig, ax = plt.subplots()
ax.bar([0, 1, 2], [mean_MC, mean_alpha_var_TD, mean_alpha_var_Q], 
       yerr=[mean_error_MC, mean_alpha_var_error_TD, mean_alpha_var_error_Q],
       align='center', ecolor='black')
ax.set_ylabel('mean rewards', fontdict=font)
ax.set_title('comparison of mean rewards (95% CI)', fontdict=font)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Monte-Carlo', 'SARSA', 'Q-Learning'], fontdict=font)
ax.yaxis.grid(True)