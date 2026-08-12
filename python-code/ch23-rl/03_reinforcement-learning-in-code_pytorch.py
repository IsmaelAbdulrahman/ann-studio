# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 23: Deep reinforcement learning
# Section: Reinforcement learning in code
# Code example 3 of 4 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "rl"
# ====================================================================

import torch, torch.nn as nn, torch.nn.functional as F

def q_net(obs_dim, n_act):                          # a small MLP Q-network
    return nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, n_act))      # one Q-value per action

online = q_net(4, 2)                                 # e.g. CartPole: 4 obs, 2 actions
target = q_net(4, 2)
target.load_state_dict(online.state_dict())          # start the target as a copy
opt = torch.optim.Adam(online.parameters(), lr=1e-3)
gamma = 0.99

def dqn_step(batch):                                 # batch from the replay buffer
    s, a, r, s2, done = batch                        # shapes: (B,4)(B,)(B,)(B,4)(B,)
    q = online(s).gather(1, a[:, None]).squeeze(1)    # Q(s,a) for the actions taken
    with torch.no_grad():                            # no gradient through the target
        a_star = online(s2).argmax(1)                # Double DQN: online picks a'
        q_next = target(s2).gather(1, a_star[:, None]).squeeze(1)  # target scores it
        y = r + gamma * q_next * (1.0 - done)         # TD target (0 past a terminal s')
    loss = F.smooth_l1_loss(q, y)                     # Huber loss on the TD error
    opt.zero_grad(); loss.backward(); opt.step()      # one gradient step (Chapter 8)
    return loss.item()

# every C steps, refresh the frozen target:  target.load_state_dict(online.state_dict())
