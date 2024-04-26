import numpy as np
from scipy.optimize import minimize
import copy
import time
import torch
device = 'cuda:0'

import random
from collections import deque

# Define the memory buffer to store experience tuples
class Buffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        sampled_experiences = zip(*random.sample(self.buffer, batch_size))
        return sampled_experiences

    def __len__(self):
        return len(self.buffer)

# class ReplayBuffer2():
#     def __init__(self, buffer_size):
#         self.buffer = deque(maxlen=buffer_size)

#     def push(self, experience):
#         self.buffer.append(experience)

#     def sample(self, batch_size):
#         if len(self.buffer) < batch_size:
#             raise ValueError("Buffer does not have enough experiences for sampling.")
#         return list(self.buffer)[-batch_size:]

#     def __len__(self):
#         return len(self.buffer)

import numpy as np
from collections import deque

class Buffer2():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError("Buffer does not have enough experiences for sampling.")

        sampled_experiences = list(self.buffer)[-batch_size:]
        a_batch = np.array([exp[0] for exp in sampled_experiences])
        b_batch = np.array([exp[1] for exp in sampled_experiences])

        return a_batch, b_batch

    def __len__(self):
        return len(self.buffer)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = 'cuda:0'
class MLP_network(nn.Module):
    def __init__(self, in_dim, hidem_dims = [32], out_dim = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidem_dims[0]).to(torch.float64).to(device)
        self.fc2 = nn.Linear(hidem_dims[0], out_dim).to(torch.float64).to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMWithAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional = False).to(torch.float64).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(torch.float64).to(device)

        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1).to(torch.float64).to(device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(torch.float64).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(torch.float64).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Attention mechanism
        attention_weights = self.attention(torch.cat((out, out), dim=2))
        out = torch.sum(attention_weights * out, dim=1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class ReactorsInSeries:
    def __init__(self, k1, k2, k3, F0, V1, V2, V3, dt = 0.1):
        self.k1 = k1  # Reaction rate constant for reactor 1
        self.k2 = k2  # Reaction rate constant for reactor 2
        self.k3 = k3  # Reaction rate constant for reactor 3
        self.F0 = F0  # Inlet feed flow rate
        self.V1 = V1  # Volume of reactor 1
        self.V2 = V2  # Volume of reactor 2
        self.V3 = V3  # Volume of reactor 3
        self.dt = dt
        self.state = np.array([1.0, 0.5, 0.5, 350.0, 1.0, 0.5, 0.5, 350.0, 1.0, 0.5, 0.5, 350.0]).reshape([1,-1])
        
    def dynamics(self, x, u):# t, x, u
        # u = np.array([1.0, 2.0, 1.5, 0.5, 0.6, 0.8]).reshape([1,-1])

        H1, xA1, xB1, T1, H2, xA2, xB2, T2, H3, xA3, xB3, T3 = x
        Q1, Q2, Q3, Ff1, Ff2, FR = u

        # H1, xA1, xB1, T1, H2, xA2, xB2, T2, H3, xA3, xB3, T3 = self.state[0].tolist()
        # Q1, Q2, Q3, Ff1, Ff2, FR = u[0].tolist()
        
        # Assuming some arbitrary values for enthalpy changes
        delta_H_r1 = -50
        delta_H_r2 = -60
        delta_H_r3 = -70

        # Reactor 1
        r1 = self.k1 * (xA1 * xB1)**0.5
        dH1dt = (self.F0 + FR - Ff1 - H1 * Q1) / self.V1
        dxA1dt = (self.F0 * xA1 + FR * 0 - Ff1 * xA1 - H1 * Q1 * xA1 - r1 * self.V1) / (self.V1 * H1)
        dxB1dt = (self.F0 * xB1 + FR * 0 - Ff1 * xB1 - H1 * Q1 * xB1 - r1 * self.V1) / (self.V1 * H1)
        dT1dt = (self.F0 * T1 + FR * 298.15 - Ff1 * T1 - H1 * Q1 * T1 + (-delta_H_r1) * r1 * self.V1) / (self.V1 * H1)

        # Reactor 2
        r2 = self.k2 * (xA2 * xB2)**0.5
        dH2dt = (H1 * Q1 - Ff2 - H2 * Q2) / self.V2
        dxA2dt = (H1 * Q1 * xA1 - Ff2 * xA2 - H2 * Q2 * xA2 - r2 * self.V2) / (self.V2 * H2)
        dxB2dt = (H1 * Q1 * xB1 - Ff2 * xB2 - H2 * Q2 * xB2 - r2 * self.V2) / (self.V2 * H2)
        dT2dt = (H1 * Q1 * T1 - Ff2 * T2 - H2 * Q2 * T2 + (-delta_H_r2) * r2 * self.V2) / (self.V2 * H2)

        # Reactor 3
        r3 = self.k3 * (xA3 * xB3)**0.5
        dH3dt = (H2 * Q2 - self.F0 - FR - H3 * Q3) / self.V3
        dxA3dt = (H2 * Q2 * xA2 - self.F0 * xA3 - FR * 0 - H3 * Q3 * xA3 - r3 * self.V3) / (self.V3 * H3)
        dxB3dt = (H2 * Q2 * xB2 - self.F0 * xB3 - FR * 0 - H3 * Q3 * xB3 - r3 * self.V3) / (self.V3 * H3)
        dT3dt = (H2 * Q2 * T2 - self.F0 * T3 - FR * 298.15 - H3 * Q3 * T3 + (-delta_H_r3) * r3 * self.V3) / (self.V3 * H3)
        
        # print(self.state,np.array([dH1dt, dxA1dt, dxB1dt, dT1dt, dH2dt, dxA2dt, dxB2dt, dT2dt, dH3dt, dxA3dt, dxB3dt, dT3dt]).reshape([1,-1]))
        # self.state += np.array([dH1dt, dxA1dt, dxB1dt, dT1dt, dH2dt, dxA2dt, dxB2dt, dT2dt, dH3dt, dxA3dt, dxB3dt, dT3dt]).reshape([1,-1])*self.dt
        return [dH1dt, dxA1dt, dxB1dt, dT1dt, dH2dt, dxA2dt, dxB2dt, dT2dt, dH3dt, dxA3dt, dxB3dt, dT3dt]
    def simulate(self, u):
        dt = self.dt
        # t = np.arange(t_span[0], t_span[1], dt)
        # n_steps = len(t)
        # x = np.zeros((len(x0), n_steps))
        # x[:, 0] = x0

        # for i in range(1, n_steps):
        # x[:, i] = x[:, i - 1] + np.array(self.dynamics(x[:, i - 1], u)) * dt
        self.state += np.array(self.dynamics(self.state[0], u[0])) * dt
        # return t, x
    # def simulate(self, u):#, t_span, x0
    #     t_span = [0, 0+self.dt]
    #     x0 = self.state[0].tolist()
    #     u = u[0].tolist()
    #     sol = solve_ivp(self.dynamics, t_span, x0, args=(u,))
    #     self.state = sol.y[0].reshape([1,-1])
    #     # return sol.t, sol.y
    
    def plot_results(self, t, x):
        fig, axes = plt.subplots(4, 3, figsize=(15, 10))
        state_names = ['H', 'xA', 'xB', 'T']
        reactor_numbers = ['1', '2', '3']

        for i in range(4):  # Iterate over states
            for j in range(3):  # Iterate over reactors
                state_idx = i * 3 + j
                axes[i, j].plot(t, x[state_idx], label=f'Reactor {reactor_numbers[j]}')
                axes[i, j].set_xlabel('Time')
                axes[i, j].set_ylabel(f'{state_names[i]}{reactor_numbers[j]}')
                axes[i, j].legend()

        plt.tight_layout()
        plt.show()


class Linear_MPC_Controller:
    def __init__(self):#, model
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        self.Rd = np.diag([0.01, 1.0])                 # input difference cost matrix
        self.Q = np.diag([1.0, 1.0])                   # state cost matrix
        self.Qf = self.Q                               # state final matrix
        self.dt=0.2
        self.L=4
        self.observer = LSTMWithAttention(input_size=18, hidden_size=16, num_layers=1, output_size=12).to(torch.float64).to(device)
        # self.observer = MLP_network(3, [16], 20).to(torch.float64).to(device)      
        self.optimizer = optim.Adam(params=self.observer.parameters(), lr=0.001)       
        self.buffer = Buffer(10000)
        self.batch_size_max = 128*1     
        self.repeat_num = 1#00
        self.history = {'loss':[]}
        self.flag = None


        # Example usage
        k1 = 0.1
        k2 = 0.2
        k3 = 0.3
        F0 = 10.0
        V1 = 100.0
        V2 = 150.0
        V3 = 200.0

        self.plant_2 = ReactorsInSeries(k1, k2, k3, F0, V1, V2, V3)

    def mpc_cost(self, u_k, state_hat, reference):
        
        u_k = u_k.reshape([self.horiz, -1]).T
        # cost = 0.0

        self.plant_2.state = state_hat
        new_state = []
        for i in range(self.horiz):
            old_state = np.array(self.plant_2.state)
            # self.plant_2.dynamics(u_k[:,i].reshape([1,-1]))
            self.plant_2.simulate(u_k[:,i].reshape([1,-1]))
            new_state.append(self.plant_2.state)

            self.seq_maker.push([np.concatenate((u_k[:,i].reshape([1,-1]), old_state.reshape([1,-1])), 1), np.array(new_state[i]).reshape([1,-1])])
        a, b = self.seq_maker.sample(self.seq_maker.__len__())
        self.buffer.push([a.sum(1),b.sum(1)])
        a = torch.tensor(a).squeeze(1).to(torch.float64).to(device)
        new_state = self.observer.forward(a.unsqueeze(0)).detach().cpu().numpy()
        cost = np.mean((np.array(new_state)-reference)**2)
        # print(cost)
        return cost

    def optimize(self, state_hat, points):
        self.horiz = points.shape[0]
        self.seq_len = self.horiz
        self.seq_maker = Buffer2(self.seq_len)
        bnd = [(0.001, 10),(0.001, 10),(0.001, 10),(0.001, 10),(0.001, 10),(0.001, 10)]*self.horiz
        # bnd = [(-10, 10),(-10, 10),(-10, 10),(-10, 10),(-10, 10),(-10, 10)]*self.horiz
        result = minimize(self.mpc_cost, args=(state_hat, points), x0 = np.zeros((6*self.horiz)), method='SLSQP', bounds = bnd)#

        if len(self.buffer) > 1:#batch_size:
            self.batch_size = min(len(self.buffer), self.batch_size_max)
            for j in range(self.repeat_num):
                a, b = self.buffer.sample(self.batch_size)
                a = torch.tensor(np.array(a)).to(torch.float64).to(device)
                b = torch.tensor(np.array(b)[:,-1,:]).to(torch.float64).to(device)
                output = self.observer(a)
                loss = F.mse_loss(output, b)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.history['loss'].append(loss.detach().cpu().numpy().reshape([1])[0])
                print(self.history['loss'][-1])
                if self.history['loss'][-1]<0.01:
                    self.repeat_num = 1
        return result.x.reshape([self.horiz,-1])[0].reshape([1,-1])