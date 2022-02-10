import torch
import numpy as np
import os
from modules import Environment_MIMO
from modules import Environment_CLQR
from modules import GaussianPolicy
from modules import BetaPolicy
from modules import DataStorage
from utility_functions import update_policy
import matplotlib.pyplot as plt


def main_func(example_name, T, num_new_data, tau, alpha_pow, beta_pow):
    """The main code of simulating the MIMO power allocation or the CLQR by using the SCAOPO algorithm
       with Gaussian policy or Beta policy.
       Please use example_name as 'MIMO_Gaussian' or 'MIMO_Beta' or 'CLQR_Gaussian' or 'CLQR_Beta'."""
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda:0"
    if 'MIMO' in example_name:
        Nt, UE_num = 8, 4  # The number of antennas and users.
        state_dim = 2 * UE_num * Nt + UE_num
        action_dim = UE_num + 1
        env = Environment_MIMO(seed=seed, Nt=Nt, UE_num=UE_num)
        constraint_dim = UE_num
        constr_lim = np.array([1.0, 1.4, 1.0, 1.4])
    else:
        state_dim, action_dim = 15, 4
        env = Environment_CLQR(seed=seed, state_dim=state_dim, action_dim=action_dim)
        constraint_dim = 1
        constr_lim = 380 * np.ones(constraint_dim)

    if 'Gaussian' in example_name:
        fc1_dim, fc2_dim = 128, 128  # hidden sizes of policy network.
        actor = GaussianPolicy(state_dim, fc1_dim, fc2_dim, action_dim, device, T)
    else:
        fc1_dim, fc2_dim = 128, 128  # hidden sizes of policy network.
        actor = BetaPolicy(state_dim, fc1_dim, fc2_dim, action_dim, device, T)

    tau_reward = tau
    tau_cost = tau
    interaction_steps = 0
    num_steps = int(1e6)
    chkpt_dir = example_name
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    chkpt_dir_data = chkpt_dir + '/saved_data'
    if not os.path.exists(chkpt_dir_data):
        os.makedirs(chkpt_dir_data)

    chkpt_dir_model = chkpt_dir + '/saved_model'
    if not os.path.exists(chkpt_dir_model):
        os.makedirs(chkpt_dir_model)

    # Initialization
    theta_dim = 0
    for para in actor.net.parameters():
        theta_dim += para.numel()
    real_theta_dim = theta_dim + action_dim  # the dimension of the policy parameter.
    # real_theta_dim = theta_dim  # use this when using the Beta policy
    paras_torch = torch.zeros((real_theta_dim,), dtype=torch.float, device=device)
    ind = 0
    for para in actor.net.parameters():
        tmp = para.numel()
        paras_torch[ind: ind + tmp] = para.data.view(-1)
        ind = ind + tmp
    paras_torch[ind:] = actor.log_std  # comment this when using the Beta policy
    func_value = np.zeros(constraint_dim + 1)
    grad = np.zeros((constraint_dim + 1, real_theta_dim))

    # Training
    buffer = DataStorage(T, num_new_data, state_dim, action_dim, constraint_dim)
    t_update = 0  # the number of updating policy
    model_saved_count = 0  # the number of saved model
    reward_rate_all = []  # all objective cost values
    cost_rate_all = []  # all constraint cost values
    reward_sum = 0
    cost_sum_total = 0
    observation = env.reset()
    for step in range(num_steps):
        # generate new data (sample one step of the env)
        state = observation
        action = actor.sample_action(state)
        observation, reward, done, info = env.step(action)  # reward is the objective cost in the paper.
        costs = np.zeros(constraint_dim + 1)
        costs[0] = reward
        for k in range(1, constraint_dim + 1):
            costs[k] = (info.get('cost_' + str(k), info.get('cost', 0)) - constr_lim[k - 1])
        buffer.store_experiences(state, action, costs)
        interaction_steps += 1
        reward_sum += reward
        cost_sum_total += info.get('cost', 0)

        # print results in the run
        if (step + 1) % 3000 == 0:
            reward_rate_all.append(reward_sum / interaction_steps)
            cost_rate_all.append(cost_sum_total / interaction_steps)
            print('step: %d, reward_rate = %.3f, avg_cost_rate = %.3f' % (step, reward_sum / interaction_steps
                                                                          ,
                                                                          cost_sum_total / interaction_steps / constraint_dim))
            np.save(chkpt_dir_data + '/reward_rate_all.npy', np.array(reward_rate_all))
            np.save(chkpt_dir_data + '/cost_rate_all.npy', np.array(cost_rate_all))

        # update the policy
        if (interaction_steps % num_new_data == 0) and (buffer.n_entries == 2 * T):
            # estimate the function value.
            t_update += 1
            alpha = 1 / (t_update ** alpha_pow)
            beta = 1 / (t_update ** beta_pow)
            state_batch, action_batch, costs_batch = buffer.take_experiences()
            func_value_tilda = np.mean(costs_batch, axis=0)
            func_value = (1 - alpha) * func_value + alpha * func_value_tilda

            # estimate the Q-value
            Q_hat = np.zeros((T, 1 + constraint_dim))
            for _ in range(1, T + 1):
                costs_tmp = costs_batch[_: _ + T]
                Q_hat[_ - 1] = np.sum(costs_tmp, axis=0) - T * func_value

            Q_hat[:, 0] = (Q_hat[:, 0] - np.mean(Q_hat[:, 0])) / (np.std(Q_hat[:, 0]) + 1e-6)
            for _ in range(1, 1 + constraint_dim):
                Q_hat[:, _] = Q_hat[:, _] - np.mean(Q_hat[:, _])

            Q_hat_torch = torch.tensor(Q_hat, dtype=torch.float, device=device)

            # estimate the gradient
            state_batch_torch = torch.tensor(state_batch[1:T + 1], dtype=torch.float, device=device)
            action_batch_torch = torch.tensor(action_batch[1:T + 1], dtype=torch.float, device=device)
            grad_tilda_torch = torch.zeros((1 + constraint_dim, real_theta_dim), dtype=torch.float,
                                           device=device)
            for _ in range(1 + constraint_dim):
                # calculate the gradient
                actor.zero_grad()
                log_prob = actor.evaluate_action(state_batch_torch, action_batch_torch)
                actor_loss = (Q_hat_torch[:, _] * log_prob).mean()
                actor_loss.backward()
                grad_tmp = torch.zeros(real_theta_dim, dtype=torch.float, device=device)
                ind = 0
                for para in actor.net.parameters():
                    tmp = para.numel()
                    grad_tmp[ind: ind + tmp] = para.grad.view(-1)
                    ind = ind + tmp
                grad_tmp[ind:] = actor.log_std.grad  # comment this when using the Beta policy
                grad_tilda_torch[_] = grad_tmp
            grad = (1 - alpha) * grad + alpha * grad_tilda_torch.detach().cpu().numpy()

            # update the policy parameter
            paras_bar = update_policy(func_value, grad, paras_torch.detach().cpu().numpy(),
                                      tau_reward=tau_reward, tau_cost=tau_cost)
            paras_bar_torch = torch.tensor(paras_bar, dtype=torch.float, device=device)
            paras_torch = (1 - beta) * paras_torch + beta * paras_bar_torch
            ind = 0
            for para in actor.net.parameters():
                tmp = para.numel()
                para.data = paras_torch[ind: ind + tmp].view(para.shape)
                ind = ind + tmp
            actor.log_std = paras_torch[ind:]  # comment this when using the Beta policy

        # save model
        if (step + 1) % 10000 == 0:
            checkpoint_file = os.path.join(chkpt_dir_model, 'Actor' + str(model_saved_count))
            torch.save(actor.net.state_dict(), checkpoint_file)
            model_saved_count += 1

    # plot results
    epoc = np.linspace(0, len(reward_rate_all) - 1, len(reward_rate_all))
    reward_rate_all = np.array(reward_rate_all)
    cost_limit = np.ones(epoc.shape[0]) * constr_lim.mean()
    cost_rate_all = np.array(cost_rate_all) / constraint_dim
    if 'MIMO' in example_name:
        name_split = example_name.split('_')
        plt.figure()
        plt.plot(epoc, reward_rate_all, label=r'$T = 1500, new = 1000$')
        plt.xlabel('Epoch', fontsize=12, fontweight='roman')
        plt.ylabel('Power consumption (W)', fontsize=12, fontweight='roman')
        plt.title('MIMO using the ' + name_split[1] + ' policy')
        plt.legend(loc=1)
        plt.show()

        plt.figure()
        plt.plot(epoc, cost_rate_all, label=r'$T = 1500, new = 1000$')
        plt.plot(epoc, cost_limit, 'k:', linewidth=1.5)
        plt.xlabel('Epoch', fontsize=12, fontweight='roman')
        plt.ylabel('Average delay per user (ms)', fontsize=12, fontweight='roman')
        plt.title('MIMO using the ' + name_split[1] + ' policy')
        plt.legend(loc=1)
        plt.show()
    else:
        name_split = example_name.split('_')
        plt.figure()
        plt.plot(epoc, reward_rate_all, label=r'$T = 1500, new = 1000$')
        plt.xlabel('Epoch', fontsize=12, fontweight='roman')
        plt.ylabel('Objective cost', fontsize=12, fontweight='roman')
        plt.title('CLQR using the ' + name_split[1] + ' policy')
        plt.legend(loc=1)
        plt.show()

        plt.figure()
        plt.plot(epoc, cost_rate_all, label=r'$T = 1500, new = 1000$')
        plt.plot(epoc, cost_limit, 'k:', linewidth=1.5)
        plt.xlabel('Epoch', fontsize=12, fontweight='roman')
        plt.ylabel('Constraint cost', fontsize=12, fontweight='roman')
        plt.title('CLQR using the ' + name_split[1] + ' policy')
        plt.legend(loc=1)
        plt.show()


if __name__ == "__main__":
    example_name = 'MIMO_Gaussian'
    T = 1500  # 2T is the number of stored experiences
    num_new_data = 1000  # the number of newly added experiences at each update.
    alpha_pow, beta_pow = 0.6, 0.8  # the powers of the decreasing sequences of step sizes alpha and beta.
    tau = 1.0  # the regularization constant in the surrogate function.
    main_func(example_name, T, num_new_data, tau, alpha_pow, beta_pow)

    # example_name = 'MIMO_Beta'
    # T = 1500
    # num_new_data = 1000
    # alpha_pow, beta_pow = 0.6, 0.9
    # tau = 0.35
    # main_func(example_name, T, num_new_data, tau, alpha_pow, beta_pow)

    # example_name = 'CLQR_Gaussian'
    # T = 1500
    # num_new_data = 1000
    # alpha_pow, beta_pow = 0.6, 0.9
    # tau = 10.0
    # main_func(example_name, T, num_new_data, tau, alpha_pow, beta_pow)

    # example_name = 'CLQR_Beta'
    # T = 1500
    # num_new_data = 1000
    # alpha_pow, beta_pow = 0.6, 0.8
    # tau = 10.0
    # main_func(example_name, T, num_new_data, tau, alpha_pow, beta_pow)










