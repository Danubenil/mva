from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import pickle
import joblib
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

state_dim = 6
action_dim = 4
path = "./FQI.joblib(-)"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = int(5000) # Number of samples
n_iterations = int(70)
n_epochs = 1
class ProjectAgent:

    def __init__(self):

        self.epsilon = 0.1
        #self.model = DQN().to(device)
        #self.target = DQN().to(device)
        self.observations = []


    def collect_samples(self, Qfuncs = None):
        s, _ = env.reset()
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for i in tqdm(range(N)):
            if Qfuncs is None:
                action = env.action_space.sample()
            else:
                action = self.act_qfun(s, Qfuncs)
            states.append(s)
            actions.append(action)
            next_state, r, done, trunc, _ = env.step(action)
            rewards.append(r)
            next_states.append(next_state)
            dones.append(done)
            if done or trunc:
                s, _ = env.reset()
                if done:
                    print("Done")
            else:
                s = next_state
        states = np.array(states)
        rewards = np.array(rewards)
        actions = np.array(actions).reshape((-1, 1))
        next_states = np.array(next_states)
        dones = np.array(dones)
        return states, actions, rewards, next_states, dones
    def act_qfun(self, observation, Qfuncs, use_random=False):
        self.observations.append(observation)

        #if use_random or self.epsilon < random.random():
        #    return random.randint(0, 3)
        Qvalues = np.zeros(env.action_space.n)
        for i, a in enumerate(range(env.action_space.n)):
            sa = np.append(observation, a).reshape((1, -1))
            Qvalues[i] = Qfuncs[-1].predict(sa)  
        return np.argmax(Qvalues)
    def act(self, observation, use_random=False):
        self.observations.append(observation)
        #if use_random or self.epsilon < random.random():
        #    return random.randint(0, 3)
        return self.act_qfun(observation, self.Qfuncs)
    def train_step(self, n_iterations, Qfuncs = None, gamma = 0.9):
        states, actions, rewards, next_states, dones  = self.collect_samples(Qfuncs = Qfuncs)
        Qfuncs = []
        states_actions = np.append(states, actions, axis = 1)
        target = rewards.copy()
        for i in tqdm(range(n_iterations)):
            if i != 0:
                Qnext = np.zeros((states.shape[0], actions.shape[0]))
                for a in range(actions.shape[1]):
                    actions_next = a * np.ones((states.shape[0], 1))
                    states_actions_next = np.append(next_states, actions_next, axis = 1)
                    Qnext[:, a] = Qfuncs[-1].predict(states_actions_next)
                target = rewards + gamma * (1 - dones) * np.max(Qnext, axis = 1)

            Q = RandomForestRegressor()
            Q.fit(states_actions, target)
            Qfuncs.append(Q)
        return Qfuncs
    def train(self, n_epochs = 6, gamma = 0.98,):
        Qfuncs = None
        for i in tqdm(range(n_epochs)):
            Qfuncs = self.train_step(n_iterations = n_iterations, Qfuncs = Qfuncs, gamma = gamma)
        self.Qfuncs = Qfuncs
        return Qfuncs
    def save(self, path):

        joblib.dump(self.Qfuncs[-1], path)

    def load(self):
        self.Qfuncs = [joblib.load(path)]
    
if __name__ == "__main__":
    agent = ProjectAgent()
    #states, _, _, _, _ = agent.collect_samples()
    Qfuncs = agent.train(n_epochs = n_epochs, gamma = 0.95)
    agent.save(path)
