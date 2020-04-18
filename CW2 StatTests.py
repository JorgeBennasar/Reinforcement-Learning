import numpy as np

n_episodes = 200000
count_constant = 100

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
    
means_MC = []

for i in range(50):
    
    Q_MC = np.zeros([10, 21, 2]) # Q(s, a)
    returns = np.zeros([10, 21, 2]) # empirical first-visit returns
    count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
    count_state = np.zeros([10, 21], dtype=int) # N(s)
    
    env = Easy21()
    
    mean_rewards_MC = []

    for i_epi in range(n_episodes):
        Q_MC, returns, count_state, count_state_action, r =\
            monte_carlo(Q_MC, returns, count_state, count_state_action)
    
        if i_epi == 0: 
            mean_rewards_MC.append(r)
        else:
            mean_rewards_MC.append(mean_rewards_MC[i_epi-1]+(1/(i_epi+1))*\
                                   (r-mean_rewards_MC[i_epi-1])) 
        
    means_MC.append(mean_rewards_MC[i_epi])
    
mean_MC = np.mean(means_MC) 
std_MC = np.sqrt(np.var(means_MC)) 

#%%  

## SARSA -- one episode
        
def SARSA(Q, count_state, count_state_action, alpha_policy):

    s = env.reset()
    
    epsilon = count_constant / float(count_constant + count_state[s[0]-1, 
                                                                  s[1]-1])
    action_greedy = Q[s[0]-1, s[1]-1, :].argmax()
    action = np.random.choice([action_greedy, 1 - action_greedy], 
                              p=[1. - epsilon/2., epsilon/2.])

    while True:
        next_s, r, term = env.step(action=action)
        
        count_state_action[s[0]-1, s[1]-1, action] += 1
        
        if alpha_policy == 0:
            alpha = 0.2
        else: 
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
    
means_TD = [[],[]]
    
for alpha_policy in range(2):
    
    for i in range(30):
    
        Q_TD = np.zeros([10, 21, 2]) # Q(s, a)
        count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
        count_state = np.zeros([10, 21], dtype=int) # N(s)
        
        env = Easy21()
        
        mean_rewards_TD = []
    
        for i_epi in range(n_episodes):
            
            Q_TD, count_state, count_state_action, r =\
                SARSA(Q_TD, count_state, count_state_action, alpha_policy)
        
            if i_epi == 0: 
                mean_rewards_TD.append(r)
            else:
                mean_rewards_TD.append(mean_rewards_TD[i_epi-1]+(1/(i_epi+1))*\
                                       (r-mean_rewards_TD[i_epi-1]))       
        
        means_TD[alpha_policy].append(mean_rewards_TD[i_epi])
    
mean_alpha_ct_TD = np.mean(means_TD[0]) 
std_alpha_ct_TD = np.sqrt(np.var(means_TD[0])) 
mean_alpha_var_TD = np.mean(means_TD[1]) 
std_alpha_var_TD = np.sqrt(np.var(means_TD[1])) 

#%%  

## Q-Learning -- one episode
        
def Q_Learning(Q, count_state, count_state_action, alpha_policy):

    s = env.reset()

    while True:
        action_greedy = Q[s[0]-1, s[1]-1, :].argmax()
        count_state[s[0]-1, s[1]-1] += 1
        epsilon = count_constant / float(count_constant + count_state[s[0]-1, 
                                                                      s[1]-1])
        action = np.random.choice([action_greedy, 1 - action_greedy], 
                                  p=[1. - epsilon/2., epsilon/2.])
        
        count_state_action[s[0]-1, s[1]-1, action] += 1
        
        if alpha_policy == 0:
            alpha = 0.2
        else: 
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
    
means_Q = [[],[]]
    
for alpha_policy in range(2):
    
    for i in range(30):
    
        Q_Q = np.zeros([10, 21, 2]) # Q(s, a)
        count_state_action = np.zeros([10, 21, 2], dtype=int) # N(s, a)
        count_state = np.zeros([10, 21], dtype=int) # N(s)
        
        env = Easy21()
        
        mean_rewards_Q = []
        
        for i_epi in range(n_episodes):
            
            Q_Q, count_state, count_state_action, r =\
                Q_Learning(Q_Q, count_state, count_state_action, alpha_policy)
        
            if i_epi == 0: 
                mean_rewards_Q.append(r)
            else:
                mean_rewards_Q.append(mean_rewards_Q[i_epi-1]+(1/(i_epi+1))*\
                                      (r-mean_rewards_Q[i_epi-1])) 
                
        means_Q[alpha_policy].append(mean_rewards_Q[i_epi])
    
mean_alpha_ct_Q = np.mean(means_Q[0]) 
std_alpha_ct_Q = np.sqrt(np.var(means_Q[0])) 
mean_alpha_var_Q = np.mean(means_Q[1]) 
std_alpha_var_Q = np.sqrt(np.var(means_Q[1])) 