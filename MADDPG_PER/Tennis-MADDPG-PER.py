#!/usr/bin/env python
# coding: utf-8

# # Collaboration and Competition
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
# 
# ### 1. Start the Environment
# 
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# In[1]:


from unityagents import UnityEnvironment
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import torch.optim as optim


# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
# 
# - **Mac**: `"path/to/Tennis.app"`
# - **Windows** (x86): `"path/to/Tennis_Windows_x86/Tennis.exe"`
# - **Windows** (x86_64): `"path/to/Tennis_Windows_x86_64/Tennis.exe"`
# - **Linux** (x86): `"path/to/Tennis_Linux/Tennis.x86"`
# - **Linux** (x86_64): `"path/to/Tennis_Linux/Tennis.x86_64"`
# - **Linux** (x86, headless): `"path/to/Tennis_Linux_NoVis/Tennis.x86"`
# - **Linux** (x86_64, headless): `"path/to/Tennis_Linux_NoVis/Tennis.x86_64"`
# 
# For instance, if you are using a Mac, then you downloaded `Tennis.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Tennis.app")
# ```

# In[2]:


env = UnityEnvironment(file_name="/home/deeprl/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis.x86_64", seed=1)
env.reset()


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print(brain)


# ### 2. Examine the State and Action Spaces
# 
# In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.
# 
# The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 
# 
# Run the code cell below to print some information about the environment.

# In[4]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# ### 4. Create Critic and Actor models

# In[5]:




from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import torch.optim as optim

def hidden_init(layer):
    # source: The other layers were initialized from uniform distributions
    # [− 1/sqrt(f) , 1/sqrt(f) ] where f is the fan-in of the layer
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
    
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256,
                 fc2_units=128,fc3_units=128):
        """Initialize parameters and build model.
        :param state_size: int. Dimension of each state
        :param action_size: int. Dimension of each action
        :param seed: int. Random seed
        :param fc1_units: int. Number of nodes in first hidden layer
        :param fc2_units: int. Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # source: The low-dimensional networks had 2 hidden layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc1))
        # source: The final layer weights and biases of the actor and were
        # initialized from a uniform distribution [−3 × 10−3, 3 × 10−3]
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions.
        """
        # source: used the rectified non-linearity for all hidden layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # source The final output layer of the actor was a tanh layer,
        # to bound the actions
        return torch.tanh(self.fc4(x))
        
        
        
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, nb_agents, seed,
                 fcs1_units=256, fc2_units=128,fc3_units=128):
        """Initialize parameters and build model.
        :param state_size: int. Dimension of each state
        :param action_size: int. Dimension of each action
        :param seed: int. Random seed
        :param fcs1_units: int. Nb of nodes in the first hiddenlayer
        :param fc2_units: int. Nb of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear((state_size+action_size)*nb_agents, fcs1_units)#*nb_agents
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        # source: The final layer weights and biases of the critic were
        # initialized from a uniform distribution [3 × 10−4, 3 × 10−4]
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Build a critic (value) network that maps
        (state, action) pairs -> Q-values
        :param state: tuple.
        :param action: tuple.
        """
        xs = torch.cat((state, action.float()), dim=1)
        x = F.relu(self.fcs1(xs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# ### 5. Create Noise Generator

# In[6]:


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x)
        # dx += self.sigma * np.random.rand(*self.size)  # Uniform disribution
        dx += self.sigma * np.random.randn(self.size)  # normal distribution
        # dx += self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


# ### 6. Create Replay Buffer

# In[7]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, alpha, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float) : 0~1 indicating how much prioritization is used
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        
        self.alpha = max(0., alpha)  # alpha >= 0
        self.priorities = deque(maxlen=buffer_size) #priority for each element
        self._buffer_size = buffer_size
        self.cumulative_priorities = 0.
        self.eps = 1e-6
        self._indexes = []
        self.max_priority = 1.**self.alpha # max priority = 1
        
        
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        # exclude the value that will be discareded (first element)
        if len(self.priorities) >= self._buffer_size:
            self.cumulative_priorities -= self.priorities[0]
        # initialy include the max priority possible 
        self.priorities.append(self.max_priority)  # already use alpha
        # Add to the cumulative priorities abs(td_error)
        self.cumulative_priorities += self.priorities[-1]
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        i_len = len(self.memory)#current memory size
        na_probs = None
        if self.cumulative_priorities:
            na_probs = np.array(self.priorities)/self.cumulative_priorities
        l_index = np.random.choice(i_len,size=min(i_len, self.batch_size),p=na_probs)
        self._indexes = l_index

        experiences = [self.memory[ii] for ii in l_index]#Sampling experiances

        states_list = [torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        actions_list = [torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]
        next_states_list = [torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float().to(device) for index in range(num_agents)]            
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)        
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
       
        return (states_list, actions_list, rewards, next_states_list, dones)

    def _calculate_w(self, f_priority, current_beta, max_weight, i_n):
        #  wi= ((N x P(i)) ^ -β)/max(wi)
        f_wi = (i_n * f_priority/self.cumulative_priorities)
        return (f_wi ** -current_beta)/max_weight

    def get_weights(self, current_beta,device):
        '''
        Return the importance sampling  weights of the current sample based
        on the beta passed
        :param current_beta: float. fully compensates for the non-uniform
            probabilities P(i) if β = 1
        '''
        # calculate P(i) 
        i_n = len(self.memory)
       
        max_weight = ((i_n * min(self.priorities) / self.cumulative_priorities)) ** -current_beta

        this_weights = [self._calculate_w(self.priorities[ii], current_beta, max_weight,i_n)for ii in self._indexes]
        return torch.tensor(this_weights,device=device,dtype=torch.float).reshape(-1, 1)

    def update_priorities(self, td_errors):
        '''
        Update priorities of sampled transitions
        inspiration: https://bit.ly/2PdNwU9
        :param td_errors: tuple of torch.tensors. TD-Errors of last samples
        '''
        for i, f_tderr in zip(self._indexes, td_errors):
            #print(f_tderr)
            f_tderr = float((f_tderr[0]+f_tderr[1])/2.0)
            self.cumulative_priorities -= self.priorities[i]#removing old priorities 
            self.priorities[i] = ((abs(f_tderr) + self.eps) ** self.alpha)# transition priority: pi^α = (|δi| + ε)^α
            self.cumulative_priorities += self.priorities[i]#Update new priorities
        self.max_priority = max(self.priorities)
        self._indexes = []
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# ### 7. Create DDPG single agent

# In[8]:


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 250        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 0.0001        # learning rate of the actor
LR_CRITIC = 0.001        # learning rate of the critic
WEIGHT_DECAY = 0.0001        # L2 weight decay
NOISE_DECAY = 0.99

UPDATE_EVERY = 2
PER_ALPHA = 0.6         # importance sampling exponent
PER_BETA = 0.4          # prioritization exponent
max_t = 1000
sharedBuffer = PrioritizedReplayBuffer(action_size,BUFFER_SIZE, BATCH_SIZE, alpha = PER_ALPHA,seed= 0)

class DDPGAgent(object):
    '''
    Implementation of a DDPG agent that interacts with and learns from the
    environment
    '''

    def __init__(self, state_size, action_size, rand_seed,nb_agents):
        '''Initialize an MetaAgent object.
        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param meta_agents:  multi-agents to use
        :param rand_seed: int. random seed
        :param memory: ReplayBuffer object.
        '''
        self.state_size = state_size
        self.action_size = action_size

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, rand_seed).to(device)
        self.actor_target = Actor(state_size, action_size, rand_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, nb_agents, rand_seed).to(device)
        self.critic_target = Critic(state_size, action_size, nb_agents, rand_seed).to(device)
        # NOTE: the decay corresponds to L2 regularization
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)  # , weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, rand_seed)
        self.noise_decay = NOISE_DECAY
        
        self.alpha = PER_ALPHA
        self.initial_beta = PER_BETA
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def get_beta(self, t):
        """
        Return the current exponent β based on its schedul. Linearly anneal β
        from its initial value β0 to 1, at the end of learning.
        Params
        ======
        t (int) : Current time step in the episode
        
        return current_beta (float): Current exponent beta
        """
        f_frac = min(float(t) / max_t, 1.0)
        current_beta = self.initial_beta + f_frac * (1. - self.initial_beta)
        return current_beta
    
    def step(self,t):
        if len(sharedBuffer) > BATCH_SIZE:
            experiences = sharedBuffer.sample()
            self.learn(experiences, GAMMA,t)

    def act(self, states, add_noise=True):
        '''Returns actions for given states as per current policy.
        :param states: array_like. current states
        :param add_noise: Boolean. If should add noise to the action
        '''
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        # source: Select action at = μ(st|θμ) + Nt according to the current
        # policy and exploration noise
        if add_noise:
            actions += self.noise.sample()
            #self.noise_decay *= self.noise_decay 
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma,t):
        '''
        Update policy and value params using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        :param experiences: Tuple[torch.Tensor]. tuple of (s, a, r, s', done)
        :param gamma: float. discount factor
        '''
        states_list, actions_list, rewards, next_states_list, dones = experiences
                    
        next_states_tensor = torch.cat(next_states_list, dim=1).to(device)
        states_tensor = torch.cat(states_list, dim=1).to(device)
        actions_tensor = torch.cat(actions_list, dim=1).to(device)

        # --------------------------- update critic ---------------------------
        # Get predicted next-state actions and Q values from target models
        next_actions = [self.actor_target(states) for states in states_list]        
        next_actions_tensor = torch.cat(next_actions, dim=1).to(device)        
        Q_targets_next = self.critic_target(next_states_tensor, next_actions_tensor)        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))        
        # Compute critic loss: L = 1/N SUM{(yi − Q(si, ai|θQ))^2}
        Q_expected = self.critic_local(states_tensor, actions_tensor)#critic can access all knowledge
        
         # Compute importance-sampling weight wj
        f_currbeta = self.get_beta(t)
        weights = sharedBuffer.get_weights(current_beta=f_currbeta,device=device)
        
        # Compute TD-error δj 
        td_errors = Q_targets - Q_expected
        # Update transition priority pj
        sharedBuffer.update_priorities(td_errors)
        
        critic_loss = self.weighted_mse_loss(Q_expected, Q_targets, weights).mean()
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # suggested by Attempt 3, from Udacity
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # --------------------------- update actor ---------------------------
        # Compute actor loss: ∇θμ J ≈1/N  ∇aQ(s, a|θQ)|s=si,a=μ(si)∇θμ μ(s|θμ)
        # take the current states and predict actions
        actions_pred = [self.actor_local(states) for states in states_list]        
        actions_pred_tensor = torch.cat(actions_pred, dim=1).to(device)
        # -1 * (maximize) Q value for the current prediction
        actor_loss = -self.critic_local(states_tensor, actions_pred_tensor).mean()        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()        
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ---------------------- update target networks ----------------------
        # Update the critic target networks: θQ′ ←τθQ +(1−τ)θQ′
        # Update the actor target networks: θμ′ ←τθμ +(1−τ)θμ′
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def weighted_mse_loss(self,expected, target, weights):
        '''
        Return the weighted mse loss to be used by Prioritized experience replay
        :param input: torch.Tensor.
        :param target: torch.Tensor.
        :param weights: torch.Tensor.
        :return loss:  torch.Tensor.
        '''
        # source: http://
        # forums.fast.ai/t/how-to-make-a-custom-loss-function-pytorch/9059/20
        out = (expected-target)**2
        out = out * weights.expand_as(out)
        loss = out.mean(0)  # or sum over whatever dimensions
        return loss


# ### 8. Create Multi-agent DDPG

# In[9]:


class MADDPG_PER(object):
    '''
    Implementation of a MADDPG agent that interacts with and learns from the
    environment
    '''

    def __init__(self, state_size, action_size, nb_agents, random_seed):
        '''Initialize an MultiAgent object.
        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param nb_agents: int. number of agents to use
        :param rand_seed: int. random seed
        '''
        self.nb_agents = nb_agents
        self.action_size = action_size
        self.agents = [DDPGAgent(state_size,action_size,random_seed,nb_agents) for x in range(nb_agents)]# creating agents
    
    def step(self, states, actions, rewards, next_states, dones,t):
        sharedBuffer.add(states, actions, rewards, next_states, dones)
        for agent in self.agents:
            agent.step(t)

    def act(self, states, add_noise=True):
        actions = np.zeros([num_agents, action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index], add_noise)
        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def __len__(self):
        return self.nb_agents

    def __getitem__(self, key):
        return self.agents[key]
    
    def save_weights(self):
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor_PER.pth'.format(index+1))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic_PER.pth'.format(index+1))


# ### 9. Train Agent

# In[10]:


agent = MADDPG_PER(state_size=state_size, action_size=action_size,nb_agents = num_agents, random_seed=0)
scores_avg = []
scores_std = []

def maddpg(n_episodes=5000, max_t=1000, print_every=100):
    scores_window = deque(maxlen=print_every)
    scores = []
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] #reset the environment with every episode
        states = env_info.vector_observations 
        agent.reset()
        score = np.zeros(len(agent))
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name] #Taking one step
            next_states = env_info.vector_observations   
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones,t)
            states = next_states
            score += rewards
            if np.any(dones):
                break 
        scores.append(np.max(score))
        scores_window.append(np.max(score))
        scores_avg.append(np.mean(scores_window))
        scores_std.append(np.std(scores_window))
        print('\rEpisode {}\tAverage Score: {:.3f} \t Max Score: {:.3f}'.format(i_episode, np.mean(scores_window),np.max(scores_window)), end="")
   
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.3f} '.format(i_episode, np.mean(scores_window)))
            
        if np.mean(scores_window)>=0.5:
            agent.save_weights()
            print('\nEnvironment solved in {:d} Episodes \tAverage Score: {:.3f} '.format(i_episode, np.mean(scores_window)))
            break
            
    return scores

scores = maddpg()


# ### 10. Plot Results

# In[11]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# In[12]:


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores_avg)+1), scores_avg)
plt.ylabel('Average Score')
plt.xlabel('Episode #')
plt.show()


# ### 11. Watch smart agents

# In[11]:


env_info = env.reset(train_mode=False)[brain_name]
agent = MADDPG_PER(state_size=state_size, action_size=action_size,nb_agents = num_agents, random_seed=0)
for i in range(len(agent)):
    agent[i].actor_local.load_state_dict(torch.load('agent{}_checkpoint_actor_PER.pth'.format(i+1)))
    agent[i].critic_local.load_state_dict(torch.load('agent{}_checkpoint_critic_PER.pth'.format(i+1)))

states = env_info.vector_observations            # get the current state
scores = np.zeros(len(agent))                                          # initialize the score
for i in range(1000):
    actions = agent.act(states)                # select an action
    env_info = env.step(actions)[brain_name]        # send the action to the environment
    next_states = env_info.vector_observations   # get the next state
    rewards = env_info.rewards                  # get the reward
    dones = env_info.local_done                 # see if episode has finished
    scores += rewards                                # update the score
    states = next_states                             # roll over the state to next time step
    
    
print("Average Score: {}".format(np.mean(scores))) 


# In[ ]:




