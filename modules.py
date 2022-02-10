import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPolicy(nn.Module):
    """The class to realize the Gaussian policy.
    The MIMO and CLQR have different bounds of action space. Thus some hyper-paras are different."""
    def __init__(self, state_dim, fc1_dim, fc2_dim, action_dim, device, T):
        super(GaussianPolicy, self).__init__()
        self.net = MLP_Gaussian(state_dim, fc1_dim, fc2_dim, action_dim, device)
        self.log_std = -0.5 * torch.ones(action_dim, dtype=torch.float, device=device)
        self.action_dim = action_dim
        self.T = T
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        raise NotImplementedError

    def evaluate_action(self, state_torch, action_torch):
        self.net.train()
        mu = self.net(state_torch)
        self.log_std.requires_grad = True
        self.std_eval = torch.exp(self.log_std)
        self.std_eval = self.std_eval.view(1, -1).repeat(self.T, 1)
        gaussian_ = torch.distributions.normal.Normal(mu, self.std_eval)
        log_prob_action = gaussian_.log_prob(action_torch).sum(dim=1)

        return log_prob_action

    def sample_action(self, state):
        self.net.eval()
        self.log_std.requires_grad = False
        state_torch = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            mu = self.net(state_torch)
            self.std_sample = torch.exp(self.log_std)
            gaussian_ = torch.distributions.normal.Normal(mu, self.std_sample)
            action = gaussian_.sample()

        return action.detach().cpu().numpy()


class MLP_Gaussian(nn.Module):
    """The neural network used to approximate the Gaussian policy"""
    def __init__(self, state_dim, fc1_dim, fc2_dim, action_dim, device):
        super(MLP_Gaussian, self).__init__()
        self.input_dim = state_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        nn.init.orthogonal_(self.fc1.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias.data, 0.0)

        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        nn.init.orthogonal_(self.fc2.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias.data, 0.0)

        self.fc3 = nn.Linear(self.fc2_dim, self.action_dim)
        nn.init.orthogonal_(self.fc3.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc3.bias.data, 0.0)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        mu = self.fc3(x)
        mu = 2.5 * torch.sigmoid(mu)

    # def forward(self, state):  # use this when simulating the CLQR.
    #     x = self.fc1(state)
    #     x = torch.tanh(x)
    #     x = self.fc2(x)
    #     x = torch.tanh(x)
    #     mu = self.fc3(x)

        return mu


class BetaPolicy(nn.Module):
    """The class to realize the Beta policy.
    The MIMO and CLQR have different bounds of action space. Thus some hyper-paras are different."""
    def __init__(self, state_dim, fc1_dim, fc2_dim, action_dim, device, T):
        super(BetaPolicy, self).__init__()
        self.net = MLP_Beta(state_dim, fc1_dim, fc2_dim, 2 * action_dim, device)
        self.action_dim = action_dim
        self.h = 2.5 * torch.ones(self.action_dim, dtype=torch.float, device=device)
        # self.h = 2 * torch.ones(self.action_dim, dtype=torch.float, device=device)  # use this when simulating CLQR.
        self.T = T
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        raise NotImplementedError

    def evaluate_action(self, state_torch, action_torch):
        self.net.train()
        a_b_values = self.net(state_torch)
        a = a_b_values[:, 0:self.action_dim]
        b = a_b_values[:, self.action_dim:]
        beta_ = torch.distributions.beta.Beta(a, b)
        h = self.h.view(1, -1).repeat(self.T, 1)
        log_prob_action = beta_.log_prob(action_torch / h).sum(dim=1)
        # log_prob_action = beta_.log_prob((action_torch + h) / (2 * h)).sum(dim=1)  # use this when simulating CLQR.

        return log_prob_action

    def sample_action(self, state):
        self.net.eval()
        state_torch = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            a_b_values = self.net(state_torch)
            a = a_b_values[0:self.action_dim]
            b = a_b_values[self.action_dim:]
            beta_ = torch.distributions.beta.Beta(a, b)
            action = beta_.sample() * self.h
            # action = beta_.sample() * (2 * self.h) - self.h  # use this when simulating CLQR.

        return action.detach().cpu().numpy()


class MLP_Beta(nn.Module):
    """The neural network used to approximate the Beta policy"""
    def __init__(self, state_dim, fc1_dim, fc2_dim, action_dim, device):
        super(MLP_Beta, self).__init__()
        self.input_dim = state_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        nn.init.orthogonal_(self.fc1.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias.data, 0.0)

        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        nn.init.orthogonal_(self.fc2.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias.data, 0.0)

        self.fc3 = nn.Linear(self.fc2_dim, self.action_dim)
        nn.init.orthogonal_(self.fc3.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc3.bias.data, 0.0)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        a_b_values = self.fc3(x)
        a_b_values = F.softplus(a_b_values)

    # def forward(self, state):  # use this when simulating CLQR.
    #     x = self.fc1(state)
    #     x = torch.tanh(x)
    #     x = self.fc2(x)
    #     x = torch.tanh(x)
    #     a_b_values = self.fc3(x)
    #     a_b_values = F.softplus(a_b_values) + 1

        return a_b_values


class DataStorage(object):
    """The class to realize the dynamic storage of data samples"""
    def __init__(self, T, num_new_data, state_dim, action_dim, constraint_dim):
        self.T = T
        self.num_new_data = num_new_data
        self.count = 0
        self.state_memory = np.zeros((2 * self.T, state_dim))
        self.action_memory = np.zeros((2 * self.T, action_dim))
        self.cost_memory = np.zeros((2 * self.T, 1+constraint_dim))
        self.n_entries = 0
        self.state_memory_tmp = np.zeros((num_new_data, state_dim))
        self.action_memory_tmp = np.zeros((num_new_data, action_dim))
        self.cost_memory_tmp = np.zeros((num_new_data, 1+constraint_dim))

    def store_experiences(self, state, action, costs):
        if self.count < 2 * self.T:
            self.state_memory[self.count] = state
            self.action_memory[self.count] = action
            self.cost_memory[self.count] = costs
            self.count += 1
        else:
            ind = self.count % self.num_new_data
            self.state_memory_tmp[ind] = state
            self.action_memory_tmp[ind] = action
            self.cost_memory_tmp[ind] = costs
            if ind == self.num_new_data-1:
                self.state_memory[0: 2 * self.T - self.num_new_data] = self.state_memory[self.num_new_data:]
                self.state_memory[2 * self.T - self.num_new_data:] = self.state_memory_tmp
                self.action_memory[0: 2 * self.T - self.num_new_data] = self.action_memory[self.num_new_data:]
                self.action_memory[2 * self.T - self.num_new_data:] = self.action_memory_tmp
                self.cost_memory[0: 2 * self.T - self.num_new_data] = self.cost_memory[self.num_new_data:]
                self.cost_memory[2 * self.T - self.num_new_data:] = self.cost_memory_tmp
            self.count += 1

        if self.n_entries < 2 * self.T:
            self.n_entries += 1

    def take_experiences(self):

        return self.state_memory, self.action_memory, self.cost_memory


class Environment_MIMO(object):
    """The environment class of the MIMO power allocation.
    For conciseness, we adopt the 'delay' Q/mu in the simulation."""
    def __init__(self, seed, Nt, UE_num):
        super(Environment_MIMO, self).__init__()
        self.seed = seed
        self.seed_step = seed
        self.Nt = Nt
        self.UE_num = UE_num
        self.user_per_group = 2
        self.group_num = int(UE_num / self.user_per_group)
        self.state_dim = 2 * UE_num * Nt + UE_num
        self.action_dim = UE_num + 1
        self.Np = 4

        np.random.seed(seed)
        PathGain_dB = np.random.uniform(-10, 10, self.group_num)
        self.PathGain = 10 ** (PathGain_dB / 10)
        alpha_power_group = np.zeros((self.group_num, self.Np))
        for group in range(self.group_num):
            tmp = np.random.exponential(scale=1, size=self.Np)
            alpha_power_group[group] = (tmp * self.PathGain[group]) / np.sum(tmp)
        self.alpha_power = np.tile(alpha_power_group, (self.user_per_group, 1))

        array_reponse_group = np.zeros((self.group_num * self.Nt, self.Np)) + \
                              1j * np.zeros((self.group_num * self.Nt, self.Np))
        for group in range(self.group_num):
            A_tmp = np.zeros((self.Nt, self.Np)) + 1j * np.zeros((self.Nt, self.Np))
            for i in range(self.Np):
                AoD = self.laprnd(mu=0, angular_spread=5)
                A_tmp[:, i] = np.exp(1j * np.pi * np.sin(AoD) * np.arange(0, self.Nt))
            array_reponse_group[group * self.Nt: (group+1) * self.Nt] = A_tmp
        self.array_response = np.tile(array_reponse_group, (self.user_per_group, 1))

        self.H_g = np.zeros((self.group_num, Nt)) + 1j * np.zeros((self.group_num, Nt))
        self.H = np.zeros((UE_num, Nt)) + 1j * np.zeros((UE_num, Nt))
        self.D = np.zeros(UE_num)
        self.state = np.zeros(self.state_dim)
        self.noise_power = 1e-6
        self.Dmax = 5

    def reset(self):
        # Reset the environment and return the initial state.
        np.random.seed(self.seed)
        for g in range(self.group_num):
            alpha_power_g = self.alpha_power[g]
            A_g = self.array_response[g * self.Nt: (g + 1) * self.Nt]
            alpha_g = np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np) + \
                      1j * np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np)
            self.H_g[g] = A_g @ alpha_g
        self.H = np.repeat(self.H_g, self.user_per_group, axis=0)
        self.D = np.zeros(self.UE_num)
        h_real = np.real(self.H)
        h_real = h_real.reshape(-1)
        h_imag = np.imag(self.H)
        h_imag = h_imag.reshape(-1)
        self.state = np.hstack((h_real, h_imag, self.D))

        return self.state

    def step(self, action):
        # action contains power allocation and regularization factor.
        # return the next_state, reward, done = False, info.
        np.random.seed(self.seed_step)
        self.seed_step += 1
        action = action.reshape(-1)
        action[action <= 0] = 1e-6
        power = action[0: self.UE_num]
        reg_factor = action[self.UE_num]

        reward = np.sum(power)
        costs = self.D
        info = {'cost_' + str(i): costs[i - 1] for i in range(1, self.UE_num + 1)}
        info['cost'] = np.sum(costs)

        try:
            V = self.H.conjugate().T @ np.linalg.inv(self.H @ self.H.conjugate().T + reg_factor * np.eye(self.UE_num))
        except:
            V = self.H.conjugate().T @ np.linalg.pinv(self.H @ self.H.conjugate().T + reg_factor * np.eye(self.UE_num))

        norm_vector = np.zeros(self.UE_num)
        for k in range(self.UE_num):
            norm_vector[k] = 1 / (np.linalg.norm(V[:, k]) + 1e-7)
        V_tilda = V @ np.diag(norm_vector)

        hv_tilda = self.H @ V_tilda
        r_d = np.zeros(self.UE_num)
        for k in range(self.UE_num):
            module_squ = np.abs(hv_tilda[k]) ** 2
            numerator = power[k] * module_squ[k]
            module_squ[k] = 0
            dominator = np.sum(power * module_squ) + self.noise_power
            r_d[k] = np.log2(1 + numerator / dominator)
        A_d = np.random.uniform(0, 2, self.UE_num)
        self.D = self.D + A_d - r_d
        self.D[self.D <= 0] = 0.0
        self.D[self.D >= self.Dmax] = self.Dmax
        for g in range(self.group_num):
            alpha_power_g = self.alpha_power[g]
            A_g = self.array_response[g * self.Nt: (g + 1) * self.Nt]
            alpha_g = np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np) + \
                      1j * np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np)
            self.H_g[g] = A_g @ alpha_g
        self.H = np.repeat(self.H_g, self.user_per_group, axis=0)
        h_real = np.real(self.H)
        h_real = h_real.reshape(-1)
        h_imag = np.imag(self.H)
        h_imag = h_imag.reshape(-1)
        self.state = np.hstack((h_real, h_imag, self.D))
        d = False

        return self.state, reward, d, info

    def laprnd(self, mu, angular_spread):
        # generate random number of Laplacian distribution.
        b = angular_spread / np.sqrt(2)
        a = np.random.rand(1) - 0.5
        x = mu - b * np.sign(a) * np.log(1 - 2 * np.abs(a))

        return x


class Environment_CLQR(object):
    """The environment class of the CLQR."""
    def __init__(self, seed, state_dim, action_dim):
        super(Environment_CLQR, self).__init__()
        self.seed = seed
        self.seed_step = seed
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.s = np.zeros(state_dim)
        self.A = np.zeros((state_dim, state_dim))
        self.B = np.zeros((state_dim, action_dim))
        self.Q1 = np.zeros((state_dim, state_dim))
        self.R1 = np.zeros((action_dim, action_dim))
        self.Q2 = np.zeros((state_dim, state_dim))
        self.R2 = np.zeros((action_dim, action_dim))
        self.noise_mu = 1
        self.noise_std = 0.9

    def reset(self):
        # Reset the environment and return the initial state.
        np.random.seed(self.seed)
        self.A = np.random.randn(self.state_dim, self.state_dim)
        self.A = (self.A + self.A.T) / 30
        self.B = np.random.randn(self.state_dim, self.action_dim) / 3
        eig_values = np.random.rand(self.state_dim)
        S = np.diag(eig_values)
        U = self.generate_ortho_mat(dim=self.state_dim)
        self.Q1 = U @ S @ (U.T)
        E1 = np.random.randn(self.action_dim, self.action_dim)
        self.R1 = E1 @ (E1.T)
        np.random.seed(self.seed + 1996)
        C2 = np.random.exponential(1/3, size=(self.state_dim, self.state_dim))
        self.Q2 = C2 @ (C2.T)
        eig_values = np.random.rand(self.action_dim)
        S = np.diag(eig_values)
        U = self.generate_ortho_mat(dim=self.action_dim)
        self.R2 = U @ S @ (U.T)
        self.R2 = self.R2 @ (self.R2.T)

        self.s = np.random.randn(self.state_dim)

        return self.s

    def step(self, a):
        # return the next_state, reward, done = False, info.
        np.random.seed(self.seed_step)
        self.seed_step += 1
        a = a.reshape(-1)
        r = self.s.T @ self.Q1 @ self.s + a.T @ self.R1 @ a
        c = self.s.T @ self.Q2 @ self.s + a.T @ self.R2 @ a
        d = False
        info = {'cost': c}
        self.s = self.A @ self.s + self.B @ a + (self.noise_mu + self.noise_std * np.random.randn(self.state_dim))

        return self.s, r, d, info

    def generate_ortho_mat(self, dim):
        # generate orthogonal matrix
        random_state = np.random
        H = np.eye(dim)
        D = np.ones((dim,))
        for n in range(1, dim):
            x = random_state.normal(size=(dim - n + 1,))
            D[n - 1] = np.sign(x[0])
            x[0] -= D[n - 1] * np.sqrt((x * x).sum())
            # Householder transformation
            Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
            mat = np.eye(dim)
            mat[n - 1:, n - 1:] = Hx
            H = np.dot(H, mat)
            # Fix the last sign such that the determinant is 1
        D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
        # Equivalent to np.dot(np.diag(D), H) but faster, apparently
        H = (D * H.T).T
        return H
