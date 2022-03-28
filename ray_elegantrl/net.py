import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal, Categorical
from scipy.stats import beta as scipy_beta
from scipy.optimize import fmin

"""
Modify [ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)
by https://github.com/GyChou
"""

'''Q Network'''


class QNet(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))
        self.explore_rate = 1.

    def forward(self, state):
        return self.net(state)  # Q value


class QNetDuel(nn.Module):  # Dueling DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_val = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, 1))  # Q value
        self.net_adv = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, action_dim))  # advantage function value 1
        self.explore_rate = 1.

    def forward(self, state):
        t_tmp = self.net_state(state)
        q_val = self.net_val(t_tmp)
        q_adv = self.net_adv(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # dueling Q value


class QNetTwin(nn.Module):  # Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())  # state
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))  # q2 value
        self.softmax = torch.nn.Softmax(dim=1)
        self.explore_rate = 1.

    # policy
    def forward(self, state):
        tmp = self.net_state(state)
        q = self.net_q1(tmp)
        return self.softmax(q).argmax(dim=1)

    def get_a_prob(self, state):
        tmp = self.net_state(state)
        q = self.net_q1(tmp)
        return self.softmax(q)

    def get_q1(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp)

    def get_q1_q2(self, state):
        tmp = self.net_state(state)
        q1 = self.net_q1(tmp)
        q2 = self.net_q2(tmp)
        return q1, q2  # two Q values


class QNetTwinDuel(nn.Module):  # D3QN: Dueling Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_val1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1))  # q1 value
        self.net_val2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1))  # q2 value
        self.net_adv1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim))  # advantage function value 1
        self.net_adv2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim))  # advantage function value 1
        self.softmax = torch.nn.Softmax(dim=1)
        self.explore_rate = 1.

    # policy
    def forward(self, state):
        t_tmp = self.net_state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        q = q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # one dueling Q value
        return self.softmax(q).argmax(dim=1)

    def get_a_prob(self, state):
        t_tmp = self.net_state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        q = q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # one dueling Q value
        return self.softmax(q)

    def get_q1(self, state):
        t_tmp = self.net_state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # one dueling Q value

    def get_q1_q2(self, state):
        tmp = self.net_state(state)

        val1 = self.net_val1(tmp)
        adv1 = self.net_adv1(tmp)
        q1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        val2 = self.net_val2(tmp)
        adv2 = self.net_adv2(tmp)
        q2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)
        return q1, q2  # two dueling Q values


'''Policy Network (Actor)'''


class Actor(nn.Module):  # DPG: Deterministic Policy Gradient
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class RNNPPO(nn.Module):

    def __init__(self, mid_dim, state_dim, action_dim, hidden_state_dim, if_use_dn=False):
        super().__init__()
        self.rnn = nn.LSTMCell(state_dim, hidden_state_dim)
        self.hidden_state_dim = hidden_state_dim
        self.critic_net = nn.Sequential(nn.Linear(hidden_state_dim + state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        # nn.Linear(mid_dim, mid_dim), nn.ReLU(),  # nn.Hardswish(),
                                        nn.Linear(mid_dim, 1), )

        self.actor_net = nn.Sequential(nn.Linear(hidden_state_dim + state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                       # nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim), )

        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        lstm_layer_norm(self.rnn, self.hidden_state_dim)

        layer_norm(self.actor_net[-1], std=0.1)  # output layer for action
        layer_norm(self.critic_net[-1], std=0.5)  # output layer for Q value

    def critic_forward(self, state, hidden_state, cell_state):  # state: batch, dim
        rnn_embading, _ = self.lstm_infer(state, hidden_state, cell_state)
        return self.critic_net(torch.cat([state, rnn_embading], dim=1))  # V value

    def actor_forward(self, state, hidden_state, cell_state):  # state: batch, dim
        hidden_state_next, cell_state_next = self.lstm_infer(state, hidden_state, cell_state)
        return self.actor_net(torch.cat([state, hidden_state_next], dim=1)).tanh(), hidden_state_next, cell_state_next

    def forward(self, state, hidden_state, cell_state, len_sequence):  # state: timestep, batch, dim
        rnn_embading = self.lstm_forward(state, hidden_state, cell_state, len_sequence)
        state = state.permute(1, 0, 2)
        state = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(state)], dim=0)
        input = torch.cat([state, rnn_embading], dim=1)
        return self.actor_net(input).tanh(), self.critic_net(input)

    def lstm_forward(self, state, hidden_state, cell_state, len_sequence):  # state: timestep, batch, dim
        if not self.if_store_state:
            hidden_state = torch.zeros([state.shape[1], self.hidden_state_dim]).to(device=state.device)  # zero state
            cell_state = torch.zeros([state.shape[1], self.hidden_state_dim]).to(device=state.device)  # zero state
        rnn_ouput_list = []
        for i in range(len(state)):
            hidden_state, cell_state = self.rnn(state[i], (hidden_state, cell_state))
            rnn_ouput_list.append(hidden_state)
        tmp_output = torch.stack(rnn_ouput_list, dim=1)
        rnn_embading = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(tmp_output)], dim=0)
        return rnn_embading

    def lstm_infer(self, state, hidden_state, cell_state):  # state: batch, dim
        hidden_state, cell_state = self.rnn(state, (hidden_state, cell_state))
        return hidden_state, cell_state

    def get_action(self, state, hidden_state, cell_state):  # state: batch, dim
        hidden_state_next, cell_state_next = self.lstm_infer(state, hidden_state, cell_state)
        input = torch.cat([state, hidden_state_next], dim=1)
        pi = self.get_distribution(input)
        return pi.sample().tanh(), hidden_state_next, cell_state_next

    def get_distribution(self, state):
        a_avg = self.actor_net(state)
        a_std = self.a_std_log.clamp(-20, 2).exp()
        return Normal(loc=a_avg, scale=a_std)

    def compute_logprob_infer(self, state, action, hidden_state, cell_state):
        action = action.atanh()
        rnn_embading, _ = self.lstm_infer(state, hidden_state, cell_state)
        input = torch.cat([state, rnn_embading], dim=1)
        pi = self.get_distribution(input)
        return pi.log_prob(action).sum(dim=1)

    def compute_logprob(self, state, action, hidden_state, cell_state, len_sequence):
        action = action.atanh()
        rnn_embading = self.lstm_forward(state, hidden_state, cell_state, len_sequence)
        state = state.permute(1, 0, 2)
        state = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(state)], dim=0)
        input = torch.cat([state, rnn_embading], dim=1)
        pi = self.get_distribution(input)

        delta = ((pi.loc - action) / pi.scale).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
        log_prob1 = -(self.a_std_log.clamp(-20, 2) + self.sqrt_2pi_log + delta).sum(1)
        return log_prob1


class RNNEmbedding(nn.Module):
    def __init__(self, hidden_state_dim, state_dim, embedding_dim=64, if_store_state=True):
        super().__init__()
        self.ego_state = 12
        self.state_dim = state_dim
        self.if_store_state = if_store_state
        self.rnn = nn.LSTMCell(self.ego_state, hidden_state_dim)
        self.hidden_state_dim = hidden_state_dim
        self.embedding = nn.Linear(self.state_dim + self.hidden_state_dim, embedding_dim)  # the average of action

        lstm_layer_norm(self.rnn, self.hidden_state_dim)

    def forward(self, state, hidden_state, cell_state, len_sequence):  # state: timestep, batch, dim
        rnn_input = state[:, :, :self.ego_state]
        rnn_embading = self.sequence_infer(rnn_input, hidden_state, cell_state, len_sequence)
        state = state.permute(1, 0, 2)
        state = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(state)], dim=0)
        return self.embedding(torch.cat([state, rnn_embading], dim=1))

    def sequence_infer(self, state, hidden_state, cell_state, len_sequence):  # state: timestep, batch, dim
        if not self.if_store_state:
            hidden_state = torch.zeros([state.shape[1], self.hidden_state_dim]).to(device=state.device)  # zero state
            cell_state = torch.zeros([state.shape[1], self.hidden_state_dim]).to(device=state.device)  # zero state
        rnn_ouput_list = []
        for i in range(len(state)):
            hidden_state, cell_state = self.rnn(state[i], (hidden_state, cell_state))
            rnn_ouput_list.append(hidden_state)
        tmp_output = torch.stack(rnn_ouput_list, dim=0)
        output = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(tmp_output)], dim=0)
        return output

    def single_infer(self, state, hidden_state, cell_state):  # state: batch, dim
        input = state[:, :self.ego_state]
        hidden_state, cell_state = self.rnn(input, (hidden_state, cell_state))
        return hidden_state, cell_state

    def embedding_infer(self, state, hidden_state, cell_state):
        next_hidden_state, next_cell_state = self.single_infer(state, hidden_state, cell_state)
        embedding = self.embedding(torch.cat([state, next_hidden_state], dim=1))
        return embedding, next_hidden_state, next_cell_state


class CarlaRNNPPO(nn.Module):

    def __init__(self, mid_dim, state_dim, action_dim, hidden_state_dim, if_store_state=True, if_use_dn=False):
        super().__init__()
        self.ego_state = 12
        self.state_dim = state_dim
        self.if_store_state = if_store_state
        self.rnn = nn.LSTMCell(self.ego_state, hidden_state_dim)
        self.hidden_state_dim = hidden_state_dim
        self.critic_net = nn.Sequential(nn.Linear(hidden_state_dim + state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        # nn.Linear(mid_dim, mid_dim), nn.ReLU(),  # nn.Hardswish(),
                                        nn.Linear(mid_dim, 1), )

        self.actor_net = nn.Sequential(nn.Linear(hidden_state_dim + state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                       # nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim), )

        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        lstm_layer_norm(self.rnn, self.hidden_state_dim)

        layer_norm(self.actor_net[-1], std=0.1)  # output layer for action
        layer_norm(self.critic_net[-1], std=0.5)  # output layer for Q value

    def critic_forward(self, state, hidden_state, cell_state):  # state: batch, dim
        rnn_input = state[:, :self.ego_state]
        rnn_embading, _ = self.lstm_infer(rnn_input, hidden_state, cell_state)
        return self.critic_net(torch.cat([state, rnn_embading], dim=1))  # V value

    def actor_forward(self, state, hidden_state, cell_state):  # state: batch, dim
        rnn_input = state[:, :self.ego_state]
        hidden_state_next, cell_state_next = self.lstm_infer(rnn_input, hidden_state, cell_state)
        return self.actor_net(
            torch.cat([state, hidden_state_next], dim=1)).tanh(), hidden_state_next, cell_state_next

    def forward(self, state, hidden_state, cell_state, len_sequence):  # state: timestep, batch, dim
        rnn_input = state[:, :, :self.ego_state]
        rnn_embading = self.lstm_forward(rnn_input, hidden_state, cell_state, len_sequence)
        state = state.permute(1, 0, 2)
        state = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(state)], dim=0)
        input = torch.cat([state, rnn_embading], dim=1)
        return self.actor_net(input).tanh(), self.critic_net(input)

    def lstm_forward(self, state, hidden_state, cell_state, len_sequence):  # state: timestep, batch, dim
        if not self.if_store_state:
            hidden_state = torch.zeros([state.shape[1], self.hidden_state_dim]).to(device=state.device)  # zero state
            cell_state = torch.zeros([state.shape[1], self.hidden_state_dim]).to(device=state.device)  # zero state
        rnn_ouput_list = []
        for i in range(len(state)):
            hidden_state, cell_state = self.rnn(state[i], (hidden_state, cell_state))
            rnn_ouput_list.append(hidden_state)
        tmp_output = torch.stack(rnn_ouput_list, dim=1)
        rnn_embading = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(tmp_output)], dim=0)
        return rnn_embading

    def lstm_infer(self, state, hidden_state, cell_state):  # state: batch, dim
        hidden_state, cell_state = self.rnn(state, (hidden_state, cell_state))
        return hidden_state, cell_state

    def get_action(self, state, hidden_state, cell_state):  # state: batch, dim
        rnn_input = state[:, :self.ego_state]
        hidden_state_next, cell_state_next = self.lstm_infer(rnn_input, hidden_state, cell_state)
        input = torch.cat([state, hidden_state_next], dim=1)
        pi = self.get_distribution(input)
        return pi.sample().tanh(), hidden_state_next, cell_state_next

    def get_distribution(self, state):
        a_avg = self.actor_net(state)
        a_std = self.a_std_log.clamp(-20, 2).exp()
        # if state[:, :self.state_dim].isnan().any():
        #     print("state")
        #     print(state[:, :self.state_dim])
        # if state[:, self.state_dim:].isnan().any():
        #     print("hidden state")
        #     print(state[:, self.state_dim:])
        # if a_avg.isnan().any() or a_std.isnan().any() or a_avg.isinf().any() or a_std.isinf().any():
        #     print(a_avg)
        return Normal(loc=a_avg, scale=a_std)

    def compute_logprob_infer(self, state, action, hidden_state, cell_state):
        action = action.atanh()
        rnn_input = state[:, :self.ego_state]
        rnn_embading, _ = self.lstm_infer(rnn_input, hidden_state, cell_state)
        input = torch.cat([state, rnn_embading], dim=1)
        pi = self.get_distribution(input)
        return pi.log_prob(action).sum(dim=1)

    def compute_logprob(self, state, action, hidden_state, cell_state, len_sequence):
        action = action.atanh()
        rnn_input = state[:, :, :self.ego_state]
        rnn_embading = self.lstm_forward(rnn_input, hidden_state, cell_state, len_sequence)
        state = state.permute(1, 0, 2)
        state = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(state)], dim=0)
        input = torch.cat([state, rnn_embading], dim=1)
        pi = self.get_distribution(input)

        delta = ((pi.loc - action) / pi.scale).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
        log_prob1 = -(self.a_std_log.clamp(-20, 2) + self.sqrt_2pi_log + delta).sum(1)
        return log_prob1


class CarlaRNNPPOMG(nn.Module):

    def __init__(self, mid_dim, state_dim, action_dim, hidden_state_dim, if_store_state=True, if_use_dn=False):
        super().__init__()
        self.ego_state = 12
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_store_state = if_store_state
        self.rnn = nn.LSTMCell(self.ego_state, hidden_state_dim)
        self.hidden_state_dim = hidden_state_dim
        self.critic_net = nn.Sequential(nn.Linear(hidden_state_dim + state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        # nn.Linear(mid_dim, mid_dim), nn.ReLU(),  # nn.Hardswish(),
                                        nn.Linear(mid_dim, 1), )

        self.actor_net = nn.Sequential(nn.Linear(hidden_state_dim + state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.actor_net_a_loc = nn.Linear(mid_dim, action_dim)  # the average of action
        self.actor_net_a_cholesky = nn.Linear(mid_dim, (action_dim * (action_dim + 1)) // 2)
        self.softplus = nn.Softplus(threshold=18.)

        lstm_layer_norm(self.rnn, self.hidden_state_dim)
        layer_norm(self.actor_net_a_loc, std=0.1)  # output layer for action
        layer_norm(self.critic_net[-1], std=0.5)  # output layer for Q value

    def critic_forward(self, state, hidden_state, cell_state):  # state: batch, dim
        rnn_input = state[:, :self.ego_state]
        rnn_embading, _ = self.lstm_infer(rnn_input, hidden_state, cell_state)
        return self.critic_net(torch.cat([state, rnn_embading], dim=1))  # V value

    def actor_forward(self, state, hidden_state, cell_state):  # state: batch, dim
        rnn_input = state[:, :self.ego_state]
        hidden_state_next, cell_state_next = self.lstm_infer(rnn_input, hidden_state, cell_state)
        tmp = self.actor_net(torch.cat([state, hidden_state_next], dim=1))
        return self.actor_net_a_loc(tmp).tanh(), hidden_state_next, cell_state_next

    def forward(self, state, hidden_state, cell_state, len_sequence):  # state: timestep, batch, dim
        rnn_input = state[:, :, :self.ego_state]
        rnn_embading = self.lstm_forward(rnn_input, hidden_state, cell_state, len_sequence)
        state = state.permute(1, 0, 2)
        state = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(state)], dim=0)
        input = torch.cat([state, rnn_embading], dim=1)
        tmp = self.actor_net(input)
        return self.actor_net_a_loc(tmp).tanh(), self.critic_net(input)

    def lstm_forward(self, state, hidden_state, cell_state, len_sequence):  # state: timestep, batch, dim
        if not self.if_store_state:
            hidden_state = torch.zeros([state.shape[1], self.hidden_state_dim]).to(device=state.device)  # zero state
            cell_state = torch.zeros([state.shape[1], self.hidden_state_dim]).to(device=state.device)  # zero state
        rnn_ouput_list = []
        for i in range(len(state)):
            hidden_state, cell_state = self.rnn(state[i], (hidden_state, cell_state))
            rnn_ouput_list.append(hidden_state)
        tmp_output = torch.stack(rnn_ouput_list, dim=1)
        rnn_embading = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(tmp_output)], dim=0)
        return rnn_embading

    def lstm_infer(self, state, hidden_state, cell_state):  # state: batch, dim
        hidden_state, cell_state = self.rnn(state, (hidden_state, cell_state))
        return hidden_state, cell_state

    def get_action(self, state, hidden_state, cell_state):  # state: batch, dim
        rnn_input = state[:, :self.ego_state]
        hidden_state_next, cell_state_next = self.lstm_infer(rnn_input, hidden_state, cell_state)
        input = torch.cat([state, hidden_state_next], dim=1)
        pi = self.get_distribution(input)
        return pi.sample().tanh(), hidden_state_next, cell_state_next

    def get_loc_cholesky(self, state):
        t_tmp = self.actor_net(state)
        a_loc = self.actor_net_a_loc(t_tmp)  # NOTICE! it is a_loc without .tanh()
        # a_cholesky_vector = self.net_a_cholesky(t_tmp).relu()
        a_cholesky_vector = self.softplus(self.actor_net_a_cholesky(t_tmp))
        # cholesky_diag_index = torch.arange(self.action_dim, dtype=torch.long) + 1
        # cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        # a_cholesky_vector[:, cholesky_diag_index] = self.softplus(a_cholesky_vector[:, cholesky_diag_index])
        tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        a_cholesky = torch.zeros(size=(a_loc.shape[0], self.action_dim, self.action_dim), dtype=torch.float32,
                                 device=t_tmp.device)
        a_cholesky[:, tril_indices[0], tril_indices[1]] = a_cholesky_vector
        return a_loc, a_cholesky

    def get_distribution(self, state):
        loc, cholesky = self.get_loc_cholesky(state)
        return MultivariateNormal(loc=loc, scale_tril=cholesky)

    def compute_logprob_infer(self, state, action, hidden_state, cell_state):
        action = action.atanh()
        rnn_input = state[:, :self.ego_state]
        rnn_embading, _ = self.lstm_infer(rnn_input, hidden_state, cell_state)
        input = torch.cat([state, rnn_embading], dim=1)
        pi = self.get_distribution(input)
        return pi.log_prob(action)

    def compute_logprob(self, state, action, hidden_state, cell_state, len_sequence):
        action = action.atanh()
        rnn_input = state[:, :, :self.ego_state]
        rnn_embading = self.lstm_forward(rnn_input, hidden_state, cell_state, len_sequence)
        state = state.permute(1, 0, 2)
        state = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(state)], dim=0)
        input = torch.cat([state, rnn_embading], dim=1)
        pi = self.get_distribution(input)
        return pi.log_prob(action)


class CarlaRNNPPOSequence(nn.Module):

    def __init__(self, mid_dim, state_dim, action_dim, hidden_state_dim, if_store_state=True, if_use_dn=False):
        super().__init__()
        self.ego_state = 12
        self.state_dim = state_dim
        self.if_store_state = if_store_state
        self.rnn = nn.LSTMCell(self.ego_state, hidden_state_dim)
        self.hidden_state_dim = hidden_state_dim
        self.critic_net = nn.Sequential(nn.Linear(hidden_state_dim + state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        # nn.Linear(mid_dim, mid_dim), nn.ReLU(),  # nn.Hardswish(),
                                        nn.Linear(mid_dim, 1), )

        self.actor_net = nn.Sequential(nn.Linear(hidden_state_dim + state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                       # nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim), )

        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        lstm_layer_norm(self.rnn, self.hidden_state_dim)

        layer_norm(self.actor_net[-1], std=0.1)  # output layer for action
        layer_norm(self.critic_net[-1], std=0.5)  # output layer for Q value

    def critic_forward(self, state, hidden_state, cell_state):  # state: timestep, dim
        rnn_input = state[:, :self.ego_state].unsqueeze(dim=0)
        rnn_embading, _ = self.lstm_infer(rnn_input, hidden_state, cell_state)
        return self.critic_net(torch.cat([state, rnn_embading], dim=1))  # V value

    def actor_forward(self, state, hidden_state, cell_state):  # state: timestep, dim
        rnn_input = state[:, :self.ego_state].unsqueeze(dim=1)
        hidden_state_next, cell_state_next = self.lstm_infer(rnn_input, hidden_state, cell_state)
        return self.actor_net(
            torch.cat([state[-1].unsqueeze(dim=0), hidden_state_next],
                      dim=1)).tanh(), hidden_state_next, cell_state_next

    def forward(self, state, hidden_state, cell_state, len_sequence):  # state: timestep, batch, dim
        rnn_input = state[:, :, :self.ego_state]
        rnn_embading = self.lstm_forward(rnn_input, hidden_state, cell_state, len_sequence)
        state = state.permute(1, 0, 2)
        state = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(state)], dim=0)
        input = torch.cat([state, rnn_embading], dim=1)
        return self.actor_net(input).tanh(), self.critic_net(input)

    def lstm_forward(self, state, hidden_state, cell_state, len_sequence):  # state: timestep, batch, dim
        if not self.if_store_state:
            hidden_state = torch.zeros([state.shape[1], self.hidden_state_dim]).to(device=state.device)  # zero state
            cell_state = torch.zeros([state.shape[1], self.hidden_state_dim]).to(device=state.device)  # zero state
        rnn_ouput_list = []
        for i in range(len(state)):
            hidden_state, cell_state = self.rnn(state[i], (hidden_state, cell_state))
            rnn_ouput_list.append(hidden_state)
        tmp_output = torch.stack(rnn_ouput_list, dim=1)
        rnn_embading = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(tmp_output)], dim=0)
        return rnn_embading

    def lstm_infer(self, state, hidden_state, cell_state):  # state: timestep, 1, dim
        for i in range(state.shape[0]):
            hidden_state, cell_state = self.rnn(state[i], (hidden_state, cell_state))
        return hidden_state, cell_state

    def get_action(self, state, hidden_state, cell_state):  # state: timestep, dim
        rnn_input = state[:, :self.ego_state].unsqueeze(dim=1)  # rnn_input: timestep, 1, dim
        hidden_state_next, cell_state_next = self.lstm_infer(rnn_input, hidden_state, cell_state)
        input = torch.cat([state[-1].unsqueeze(dim=0), hidden_state_next], dim=1)
        pi = self.get_distribution(input)
        return pi.sample().tanh(), hidden_state_next, cell_state_next

    def get_distribution(self, state):
        a_avg = self.actor_net(state)
        a_std = self.a_std_log.clamp(-20, 2).exp()
        # if state[:, :self.state_dim].isnan().any():
        #     print("state")
        #     print(state[:, :self.state_dim])
        # if state[:, self.state_dim:].isnan().any():
        #     print("hidden state")
        #     print(state[:, self.state_dim:])
        # if a_avg.isnan().any() or a_std.isnan().any() or a_avg.isinf().any() or a_std.isinf().any():
        #     print(a_avg)
        return Normal(loc=a_avg, scale=a_std)

    def compute_logprob_infer(self, state, action, hidden_state, cell_state):
        action = action.atanh()
        rnn_input = state[:, :self.ego_state].unsqueeze(dim=0)
        rnn_embading, _ = self.lstm_infer(rnn_input, hidden_state, cell_state)
        input = torch.cat([state, rnn_embading], dim=1)
        pi = self.get_distribution(input)
        return pi.log_prob(action).sum(dim=1)

    def compute_logprob(self, state, action, hidden_state, cell_state, len_sequence):
        action = action.atanh()
        rnn_input = state[:, :, :self.ego_state]
        rnn_embading = self.lstm_forward(rnn_input, hidden_state, cell_state, len_sequence)
        state = state.permute(1, 0, 2)
        state = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(state)], dim=0)
        input = torch.cat([state, rnn_embading], dim=1)
        pi = self.get_distribution(input)

        delta = ((pi.loc - action) / pi.scale).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
        log_prob1 = -(self.a_std_log.clamp(-20, 2) + self.sqrt_2pi_log + delta).sum(1)
        return log_prob1


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        if isinstance(state_dim, int):
            if if_use_dn:
                nn_dense = DenseNet(mid_dim)
                inp_dim = nn_dense.inp_dim
                out_dim = nn_dense.out_dim

                self.net = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
                                         nn_dense,
                                         nn.Linear(out_dim, action_dim), )
            else:
                self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                         nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                         # nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                         nn.Linear(mid_dim, action_dim), )
        else:
            def set_dim(i):
                return int(12 * 1.5 ** i)

            self.net = nn.Sequential(NnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                                     nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                                     nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                                     NnReshape(-1),
                                     nn.Linear(set_dim(5), mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, action_dim), )

        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        layer_norm(self.net[-1], std=0.1)  # output layer for action

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get_action(self, state):
        pi = self.get_distribution(state)
        return pi.sample().tanh()

    def get_distribution(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.clamp(-20, 2).exp()
        if state.isnan().any() or state.isinf().any():
            print("state")
            print(state)
        if a_avg.isnan().any() or a_std.isnan().any() or a_avg.isinf().any() or a_std.isinf().any():
            print(a_avg)
        return Normal(loc=a_avg, scale=(a_std))

    def compute_logprob(self, state, action):
        action = action.atanh()
        pi = self.get_distribution(state)

        delta = ((pi.loc - action) / pi.scale).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
        log_prob1 = -(self.a_std_log.clamp(-20, 2) + self.sqrt_2pi_log + delta).sum(1)
        # log_prob2=pi.log_prob(action).sum(dim=1)
        return log_prob1


class ActorDiscretePPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.net_a_prob = nn.Linear(mid_dim, self.action_dim)
        layer_norm(self.net_a_prob, std=0.1)  # output layer for action

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, state):
        tmp = self.net(state)
        d_a_prob = self.softmax(self.net_a_prob(tmp))
        return d_a_prob.argmax(dim=1)  # action

    def get_action(self, state):
        pi = self.get_distribution(state)
        return pi.sample()

    def get_distribution(self, state):
        tmp = self.net(state)
        da_prob = self.softmax(self.net_a_prob(tmp))
        return Categorical(da_prob)

    def compute_logprob(self, state, action):
        action = action.squeeze()

        pi = self.get_distribution(state)

        d_log_prob = pi.log_prob(action)
        return d_log_prob


class ActorSADPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mid_dim = mid_dim
        self.sp_a_num = nn.Parameter(torch.Tensor([3 for _ in range(self.action_dim)]), requires_grad=False)
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.net_a_prob = nn.Sequential(*[nn.Linear(mid_dim, int(self.sp_a_num[i])) for i in range(self.action_dim)])
        for net_a_prob in self.net_a_prob:
            layer_norm(net_a_prob, std=0.1)  # output layer for action

        self.softmax = torch.nn.Softmax(dim=1)

    def set_sp_a_num(self, sp_a_num, device='cpu'):
        self.sp_a_num = nn.Parameter(torch.Tensor(sp_a_num).to(device=device), requires_grad=False)
        self.net_a_prob = nn.Sequential(
            *[nn.Linear(self.mid_dim, int(self.sp_a_num[i])) for i in range(self.action_dim)])

    def forward(self, state):
        tmp = self.net(state)
        d_a_idx = [self.softmax(net_a_prob(tmp)).argmax(dim=1) for net_a_prob in self.net_a_prob]
        d_a_idx = torch.cat(d_a_idx, dim=0).unsqueeze(dim=0)
        return ((d_a_idx / (self.sp_a_num - 1)) * 2 - 1)  # action

    def get_action(self, state):
        dists = self.get_distribution(state)
        sample_idx = torch.cat([dist.sample() for dist in dists], dim=0).unsqueeze(dim=0)
        return ((sample_idx / (self.sp_a_num - 1)) * 2 - 1)

    def get_distribution(self, state):
        tmp = self.net(state)
        das_prob = [self.softmax(net_a_prob(tmp)) for net_a_prob in self.net_a_prob]
        # da_prob = torch.cat(da_prob, dim=1)
        dist = [Categorical(da_prob) for da_prob in das_prob]
        return dist

    def compute_logprob(self, state, action):
        action = (((self.sp_a_num - 1) * (action + 1)) / 2).round().type(torch.long)
        dists = self.get_distribution(state)
        d_log_probs = [dist.log_prob(action[:, i]) for i, dist in enumerate(dists)]
        return sum(d_log_probs)

    def _discrete_to_continous(self, input):
        return

    def _continous_to_discrete(self, input):
        return


class ActorHybridPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_shared=False):
        super().__init__()
        self.state_dim = state_dim
        self.c_action_dim = action_dim[0]
        self.d_action_dim = action_dim[1]
        self.if_shared = if_shared
        if self.if_shared:
            self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
            self.net_ca_avg = nn.Linear(mid_dim, self.c_action_dim)
            self.net_da = nn.Linear(mid_dim, self.d_action_dim)
            layer_norm(self.net_ca_avg, std=0.1)  # output layer for action
        else:
            self.net_ca_avg = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                            nn.Linear(mid_dim, self.c_action_dim))
            self.net_da = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, self.d_action_dim))
            layer_norm(self.net_ca_avg[-1], std=0.1)  # output layer for action

        self.a_std_log = nn.Parameter(torch.zeros((1, self.c_action_dim)) - 0.5,
                                      requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, state):
        if self.if_shared:
            tmp = self.net(state)
            c_a = self.net_ca_avg(tmp).tanh()
            d_a_prob = self.softmax(self.net_da(tmp))
        else:
            c_a = self.net_ca_avg(state).tanh()
            d_a_prob = self.softmax(self.net_da(state))

        da = d_a_prob.argmax(dim=1).unsqueeze(dim=1)
        return torch.cat((c_a, da), 1)  # action

    def get_action(self, state):
        pi_c, pi_d = self.get_distribution(state)
        return torch.cat((pi_c.sample().tanh(), pi_d.sample().unsqueeze(dim=1)), 1)

    def get_distribution(self, state):
        if self.if_shared:
            tmp = self.net(state)
            ca_avg = self.net_ca_avg(tmp)
            da_prob = self.softmax(self.net_da(tmp))
        else:
            c_a = self.net_ca_avg(state).tanh()
            d_a_prob = self.softmax(self.net_da(state))
        ca_std = self.a_std_log.clamp(-20, 2).exp()
        pi_c = Normal(loc=ca_avg, scale=(ca_std))
        pi_d = Categorical(da_prob)
        return pi_c, pi_d

    def compute_logprob(self, state, action):
        c_action = action[:, :-1].atanh()
        d_action = action[:, -1]

        pi_c, pi_d = self.get_distribution(state)

        delta = ((pi_c.loc - c_action) / pi_c.scale).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
        c_log_prob = -(self.a_std_log.clamp(-20, 2) + self.sqrt_2pi_log + delta).sum(1)

        d_log_prob = pi_d.log_prob(d_action)
        return c_log_prob, d_log_prob


# class ActorPPOG(nn.Module):
#     def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
#         super().__init__()
#
#         self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
#                                  nn.Linear(mid_dim, mid_dim), nn.ReLU(),
#                                  # nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
#                                  )
#
#         self.net_a_avg = nn.Linear(mid_dim, action_dim)  # the average of action
#         self.net_a_logstd = nn.Linear(mid_dim, action_dim)  # the log_std of action
#
#         layer_norm(self.net_a_avg, std=0.1)  # output layer for action
#         layer_norm(self.net_a_logstd, std=0.1)
#
#     def forward(self, state):
#         return self.net_a_avg(self.net(state)).tanh()  # action
#
#     def get_action(self, state):
#         pi = self.get_distribution(state)
#         return pi.sample()
#
#     def get_distribution(self, state):
#         tmp = self.net(state)
#         a_avg = self.net_a_avg(tmp)
#         a_std = self.net_a_logstd(tmp).exp()
#         return Normal(loc=a_avg, scale=a_std)
#
#     def compute_logprob(self, state, action):
#         pi = self.get_distribution(state)
#         return pi.log_prob(action).sum(dim=1)


class ActorPPOBeta(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 # nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 )

        self.net_a_alpha = nn.Linear(mid_dim, action_dim)  # the average of action
        self.net_a_beta = nn.Linear(mid_dim, action_dim)  # the log_std of action

        self.softplus = nn.Softplus(threshold=18.)
        self.beta = torch.distributions.Beta

    def forward(self, state):
        tmp = self.net(state)
        alpha = (self.softplus(self.net_a_alpha(tmp)) + 1)
        beta = (self.softplus(self.net_a_beta(tmp)) + 1)
        # pi = self.beta(alpha, beta)
        # return pi.mean * 2. - 1
        return ((alpha - 1) / (alpha + beta - 2)) * 2. - 1.

    def get_action(self, state):
        pi = self.get_distribution(state)
        return pi.sample() * 2. - 1.

    def get_distribution(self, state):
        tmp = self.net(state)
        alpha = self.softplus(self.net_a_alpha(tmp)) + 1
        beta = self.softplus(self.net_a_beta(tmp)) + 1
        return self.beta(alpha, beta)

    def get_explore(self, state):
        pi = self.get_distribution(state)
        return (pi.concentration0 + pi.concentration1 - 2) / 2

    def compute_logprob(self, state, action):
        pi = self.get_distribution(state)
        return pi.log_prob((action + 1.) / 2.).sum(dim=1)


# class ActorPPOBeta2(nn.Module):
#     def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
#         super().__init__()
#
#         self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
#                                  nn.Linear(mid_dim, mid_dim), nn.ReLU(),
#                                  # nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
#                                  )
#
#         self.net_a_alpha = nn.Linear(mid_dim, action_dim)  # the average of action
#         self.net_a_beta = nn.Linear(mid_dim, action_dim)  # the log_std of action
#         self.net_a_explore = nn.Linear(mid_dim, action_dim)
#         # self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
#
#         self.softplus = nn.Softplus(threshold=18.)
#         self.sigmoid = torch.nn.Sigmoid()
#         self.softmax = torch.nn.Softmax(dim=2)
#         self.beta = torch.distributions.Beta
#
#         self.explore_upbound = float('inf')
#         self.threshold = self.beta(self.explore_upbound / 2 + 1, 1).mean
#
#     def forward(self, state):
#         tmp = self.net(state)
#         explore = (self.softplus(self.net_a_explore(tmp))).clamp_max(self.explore_upbound)
#         explore = explore.unsqueeze(dim=2).repeat(1, 1, 2)
#         alpha_beta = torch.cat((self.net_a_alpha(tmp).unsqueeze(dim=2),
#                                 self.net_a_beta(tmp).unsqueeze(dim=2)), dim=2)
#         alpha_beta = self.softmax(alpha_beta) * explore + 1
#         alpha = alpha_beta[:, :, 0]
#         beta = alpha_beta[:, :, 1]
#
#         # prob_alpha = self.sigmoid(self.net_a_alpha(tmp))
#         # prob_beta = 1 - prob_alpha
#         # alpha = (prob_alpha * explore) + 1
#         # beta = (prob_beta * explore) + 1
#
#         # alpha = (self.softplus(self.net_a_alpha(tmp)) + 1)[0].detach().flatten()
#         # beta = (self.softplus(self.net_a_beta(tmp)) + 1)[0].detach().flatten()
#         # def f(x, dim):
#         #     return -scipy_beta(alpha[dim], beta[dim]).pdf(x)
#         #
#         # max_prob_a = np.zeros((alpha.shape[0],))
#         # for i in range(alpha.shape[0]):
#         #     max_prob_a[i] = fmin(func=f, x0=np.array([0.]), args=(i,), disp=0)
#         # return torch.Tensor(max_prob_a).unsqueeze(dim=0) * 2. - 1.
#         # return (self.beta(prob_alpha * explore + 1, prob_beta * explore + 1).mean * 2. - 1.). \
#         #            clamp(-self.threshold, self.threshold) / self.threshold
#
#         return ((alpha - 1) / (alpha + beta - 2)) * 2. - 1.
#
#     def get_action(self, state):
#         pi = self.get_distribution(state)
#         return pi.sample() * 2. - 1.
#         # return (pi.sample() * 2. - 1.).clamp(-self.threshold, self.threshold) / self.threshold
#
#     def get_distribution(self, state):
#         tmp = self.net(state)
#         explore = (self.softplus(self.net_a_explore(tmp))).clamp_max(self.explore_upbound)
#         explore = explore.unsqueeze(dim=2).repeat(1, 1, 2)
#         alpha_beta = torch.cat((self.net_a_alpha(tmp).unsqueeze(dim=2),
#                                 self.net_a_beta(tmp).unsqueeze(dim=2)),
#                                dim=2)  # [batch, action_dim, 2: (alpha, beta)]
#         alpha_beta = self.softmax(alpha_beta) * explore + 1
#         alpha = alpha_beta[:, :, 0]
#         beta = alpha_beta[:, :, 1]
#
#         # prob_alpha = self.sigmoid(self.net_a_alpha(tmp))
#         # prob_beta = 1 - prob_alpha
#         # alpha = prob_alpha * explore + 1
#         # beta = prob_beta * explore + 1
#         return self.beta(alpha, beta)
#
#     def get_explore(self, state):
#         tmp = self.net(state)
#         return (self.softplus(self.net_a_explore(tmp))).clamp_max(self.explore_upbound)
#
#     def compute_logprob(self, state, action):
#         tmp = self.net(state)
#         explore = (self.softplus(self.net_a_explore(tmp))).clamp_max(self.explore_upbound)
#         explore = explore.unsqueeze(dim=2).repeat(1, 1, 2)
#         alpha_beta = torch.cat((self.net_a_alpha(tmp).unsqueeze(dim=2),
#                                 self.net_a_beta(tmp).unsqueeze(dim=2)),
#                                dim=2)  # [batch, action_dim, 2: (alpha, beta)]
#         alpha_beta = self.softmax(alpha_beta) * explore + 1
#         alpha = alpha_beta[:, :, 0]
#         beta = alpha_beta[:, :, 1]
#         pi = self.beta(alpha, beta)
#
#         # log_weights = ((explore + 1).log() - pi.log_prob(
#         #     (pi.concentration0 - 1) / (pi.concentration0 + pi.concentration1 - 2))).detach_()
#         # return (log_weights + pi.log_prob((action + 1.) / 2.)).sum(dim=1)
#         return pi.log_prob((action + 1.) / 2.).sum(dim=1)
#
#     def get_entropy(self, state, action, target_entropy=None):
#         tmp = self.net(state)
#         explore = (self.softplus(self.net_a_explore(tmp))).clamp_max(self.explore_upbound)
#         explore = explore.unsqueeze(dim=2).repeat(1, 1, 2)
#         alpha_beta = torch.cat((self.net_a_alpha(tmp).unsqueeze(dim=2),
#                                 self.net_a_beta(tmp).unsqueeze(dim=2)),
#                                dim=2)  # [batch, action_dim, 2: (alpha, beta)]
#         alpha_beta = self.softmax(alpha_beta).detach_() * explore + 1
#         alpha = alpha_beta[:, :, 0]
#         beta = alpha_beta[:, :, 1]
#         tmp_pi = self.beta(alpha, beta)
#         log_prob = tmp_pi.log_prob((action + 1.) / 2.).sum(dim=1)
#         if target_entropy is not None:
#             return (log_prob.exp() * log_prob - target_entropy).clamp_min(0).mean()
#         else:
#             return (log_prob.exp() * log_prob).mean()
#
#
class ActorPPOBeta2(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 # nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 )

        self.net_a_alpha = nn.Linear(mid_dim, action_dim)  # the average of action
        self.net_a_beta = nn.Linear(mid_dim, action_dim)  # the log_std of action
        self.a_explore = nn.Parameter(torch.zeros((1, action_dim)) + 1., requires_grad=True)
        # self.a_explore_log = nn.Parameter(torch.zeros((1, action_dim)) + 1., requires_grad=True)

        self.softplus = nn.Softplus(threshold=18.)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=2)
        self.beta = torch.distributions.Beta

    def forward(self, state):
        tmp = self.net(state)
        # explore = self.a_explore_log.clamp(-16,2).exp()
        explore = self.softplus(self.a_explore)
        explore = explore.unsqueeze(dim=2).repeat(1, 1, 2)
        alpha_beta = torch.cat((self.net_a_alpha(tmp).unsqueeze(dim=2),
                                self.net_a_beta(tmp).unsqueeze(dim=2)), dim=2)
        alpha_beta = self.softmax(alpha_beta) * explore + 1
        alpha = alpha_beta[:, :, 0]
        beta = alpha_beta[:, :, 1]
        max_prob = ((alpha - 1) / (alpha + beta - 2))
        return max_prob * 2. - 1.

    def get_action(self, state):
        pi = self.get_distribution(state)
        return pi.sample() * 2. - 1.
        # return (pi.sample() * 2. - 1.).clamp(-self.threshold, self.threshold) / self.threshold

    def get_distribution(self, state):
        tmp = self.net(state)
        # explore = self.a_explore_log.clamp(-16,2).exp()
        explore = self.softplus(self.a_explore)
        explore = explore.unsqueeze(dim=2).repeat(1, 1, 2)
        alpha_beta = torch.cat((self.net_a_alpha(tmp).unsqueeze(dim=2),
                                self.net_a_beta(tmp).unsqueeze(dim=2)), dim=2)  # [batch, action_dim, 2: (alpha, beta)]
        alpha_beta = self.softmax(alpha_beta) * explore + 1
        alpha = alpha_beta[:, :, 0]
        beta = alpha_beta[:, :, 1]
        return self.beta(alpha, beta)

    def get_explore(self, state):
        # return self.a_explore_log.clamp(-16,2).exp()
        return self.softplus(self.a_explore)

    def compute_logprob(self, state, action):
        tmp = self.net(state)
        # explore = self.a_explore_log.clamp(-16,2).exp()
        explore = self.softplus(self.a_explore)
        explore = explore.unsqueeze(dim=2).repeat(1, 1, 2)
        alpha_beta = torch.cat((self.net_a_alpha(tmp).unsqueeze(dim=2),
                                self.net_a_beta(tmp).unsqueeze(dim=2)),
                               dim=2)  # [batch, action_dim, 2: (alpha, beta)]
        alpha_beta = self.softmax(alpha_beta) * explore + 1
        alpha = alpha_beta[:, :, 0]
        beta = alpha_beta[:, :, 1]
        pi = self.beta(alpha, beta)
        return pi.log_prob((action + 1.) / 2.).sum(dim=1)


class ActorPPOMG(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(), )

        self.net_a_loc = nn.Linear(mid_dim, action_dim)  # the average of action
        self.net_a_cholesky = nn.Linear(mid_dim, (action_dim * (action_dim + 1)) // 2)
        self.softplus = nn.Softplus(threshold=18.)
        layer_norm(self.net_a_loc, std=0.1)  # output layer for action, it is no necessary.
        # layer_norm(self.net_a_cholesky, std=0.01)

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_a_loc(tmp).tanh()  # action

    def get_distribution(self, state):
        loc, cholesky = self.get_loc_cholesky(state)
        return MultivariateNormal(loc=loc, scale_tril=cholesky)

    def get_loc_cholesky(self, state):
        t_tmp = self.net_state(state)
        a_loc = self.net_a_loc(t_tmp)  # NOTICE! it is a_loc without .tanh()
        # a_cholesky_vector = self.net_a_cholesky(t_tmp).relu()
        a_cholesky_vector = self.softplus(self.net_a_cholesky(t_tmp))
        # cholesky_diag_index = torch.arange(self.action_dim, dtype=torch.long) + 1
        # cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        # a_cholesky_vector[:, cholesky_diag_index] = self.softplus(a_cholesky_vector[:, cholesky_diag_index])
        tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        a_cholesky = torch.zeros(size=(a_loc.shape[0], self.action_dim, self.action_dim), dtype=torch.float32,
                                 device=t_tmp.device)
        a_cholesky[:, tril_indices[0], tril_indices[1]] = a_cholesky_vector
        return a_loc, a_cholesky

    def get_action(self, state):
        pi = self.get_distribution(state)
        return pi.sample().tanh()  # re-parameterize

    def compute_logprob(self, state, action):
        action = action.atanh()
        pi = self.get_distribution(state)
        return pi.log_prob(action)


class ActorSAC(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        if if_use_dn:
            nn_middle = DenseNet(mid_dim // 2)
            inp_dim = nn_middle.inp_dim
            out_dim = nn_middle.out_dim
        else:
            nn_middle = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
            inp_dim = mid_dim
            out_dim = mid_dim

        self.net_state = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
                                       nn_middle, )
        self.net_a_avg = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the average of action
        self.net_a_std = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the log_std of action
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

        self.softplus = torch.nn.Softplus()
        # layer_norm(self.net_a_avg, std=0.01)  # output layer for action, it is no necessary.

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        dist = self.get_distribution(state)
        return dist.sample().tanh()  # re-parameterize

    def get_distribution(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()  # todo
        return Normal(a_avg, a_std)

    def get_action_logprob(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)  # todo (-20, 2)
        a_std = a_std_log.exp()

        """add noise to action in stochastic policy"""
        noise = torch.randn_like(a_avg, requires_grad=True)
        a_noise = a_avg + a_std * noise
        a_tan = a_noise.tanh()  # action.tanh()
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_avg, a_std, requires_grad=True)

        ''' compute logprob according to mean and std of action (stochastic policy) '''
        # # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        # logprob = a_std_log + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        # different from above (gradient)
        logprob = a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)
        # same as below:
        # from torch.distributions.normal import Normal
        # logprob_noise = Normal(a_avg, a_std).logprob(a_noise)
        # logprob = logprob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()
        # same as below:
        # a_delta = (a_avg - a_noise).pow(2) /(2*a_std.pow(2))
        # logprob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        # logprob = logprob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()

        # logprob = logprob + (-a_tan.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        logprob = logprob + 2. * (np.log(2.) - a_noise - self.softplus(-2. * a_noise))
        # same as below:
        # epsilon = 1e-6
        # logprob = logprob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log()
        return a_tan, logprob.sum(1, keepdim=True)


# class ActorSACBeta(nn.Module):
#     def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
#         super().__init__()
#         if if_use_dn:  # use a DenseNet (DenseNet has both shallow and deep linear layer)
#             nn_dense = DenseNet(mid_dim)
#             inp_dim = nn_dense.inp_dim
#             out_dim = nn_dense.out_dim
#
#             self.net_state = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
#                                            nn_dense, )
#         else:  # use a simple network. Deeper network does not mean better performance in RL.
#             self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
#                                            nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
#                                            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), )
#             out_dim = mid_dim
#
#         self.net_a_alpha = nn.Linear(out_dim, action_dim)  # the average of action
#         self.net_a_beta = nn.Linear(out_dim, action_dim)  # the log_std of action
#
#         self.softplus = nn.Softplus(threshold=18.)
#         self.beta = torch.distributions.Beta
#         # layer_norm(self.net_a_avg, std=0.01)  # output layer for action, it is no necessary.
#
#     def forward(self, state):
#         tmp = self.net_state(state)
#         alpha = self.softplus(self.net_a_alpha(tmp)) + 1
#         beta = self.softplus(self.net_a_beta(tmp)) + 1
#         if alpha.any() > 1000 or beta.any() > 1000:
#             print()
#         return self.beta(alpha, beta).mean * 2. - 1.
#
#     def get_action(self, state):
#         pi = self.get_distribution(state)
#         return pi.sample() * 2. - 1.
#
#     def get_distribution(self, state):
#         tmp = self.net_state(state)
#         alpha = self.softplus(self.net_a_alpha(tmp)) + 1
#         beta = self.softplus(self.net_a_beta(tmp)) + 1
#         return self.beta(alpha, beta)
#
#     def get_action_logprob(self, state):
#         pi = self.get_distribution(state)
#         origin_action = pi.rsample()
#         log_prob = pi.log_prob(origin_action) + (-origin_action.pow(2) + 1.000001).log()
#         return origin_action * 2. - 1., log_prob.sum(1, keepdim=True)


# class ActorSACBeta2(nn.Module):
#     def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
#         super().__init__()
#
#         self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
#                                  # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
#                                  nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
#                                  nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
#                                  )
#
#         self.net_a_alpha = nn.Linear(mid_dim, action_dim)  # the average of action
#         self.net_a_beta = nn.Linear(mid_dim, action_dim)  # the log_std of action
#         self.net_a_explore = nn.Linear(mid_dim, action_dim)
#
#         self.softplus = nn.Softplus(threshold=18.)
#         self.sigmoid = torch.nn.Sigmoid()
#         self.softmax = torch.nn.Softmax(dim=2)
#         self.beta = torch.distributions.Beta
#         self.threshold = 0.95
#         self.explore_upbound = 30.
#
#     def forward(self, state):
#         tmp = self.net(state)
#         explore = self.softplus(self.net_a_explore(tmp)).clamp_max(self.explore_upbound) + 1
#         # explore = explore.unsqueeze(dim=2).repeat(1, 1, 2)[0, 1, :]
#         # alpha_beta = torch.cat((self.net_a_alpha(tmp).unsqueeze(dim=2),
#         #                         self.net_a_beta(tmp).unsqueeze(dim=2)), dim=2)
#         # alpha_beta = self.softmax(alpha_beta) * explore + 1
#         # return self.beta(alpha_beta[:, :, 0], alpha_beta[:, :, 1]).mean * 2. - 1.
#         prob_alpha = self.sigmoid(self.net_a_alpha(tmp))
#         prob_beta = 1 - prob_alpha
#         return (self.beta(prob_alpha * explore + 1, prob_beta * explore + 1).mean * 2. - 1.). \
#                    clamp(-self.threshold, self.threshold) / self.threshold
#
#     def get_action(self, state):
#         pi = self.get_distribution(state)
#         return (pi.sample() * 2. - 1.).clamp(-self.threshold, self.threshold) / self.threshold
#
#     def get_distribution(self, state):
#         tmp = self.net(state)
#         explore = self.softplus(self.net_a_explore(tmp)).clamp_max(self.explore_upbound) + 1
#         # explore = explore.unsqueeze(dim=2).repeat(1, 1, 2)[0, 1, :]
#         # alpha_beta = torch.cat((self.net_a_alpha(tmp).unsqueeze(dim=2),
#         #                         self.net_a_beta(tmp).unsqueeze(dim=2)),
#         #                        dim=2)  # [batch, action_dim, 2: (alpha, beta)]
#         # alpha_beta = self.softmax(alpha_beta) * explore + 1
#         # return self.beta(alpha_beta[:, :, 0], alpha_beta[:, :, 1])
#         prob_alpha = self.sigmoid(self.net_a_alpha(tmp))
#         prob_beta = 1 - prob_alpha
#         return self.beta(prob_alpha * explore + 1, prob_beta * explore + 1)
#
#     def get_explore(self, state):
#         tmp = self.net(state)
#         return self.softplus(self.net_a_explore(tmp)).clamp_max(self.explore_upbound) + 2
#
#     def compute_logprob(self, state, action):
#         pi = self.get_distribution(state)
#         return pi.log_prob((action * self.threshold + 1.) / 2.).sum(dim=1)
#
#     def get_action_logprob(self, state):
#         pi = self.get_distribution(state)
#         origin_action = pi.rsample()
#         log_prob = pi.log_prob(origin_action * self.threshold) + (-origin_action.pow(2) + 1.000001).log()
#         return origin_action * 2. - 1., log_prob.sum(1, keepdim=True)


# class ActorHybridSAC(nn.Module):
#     def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
#         super().__init__()
#         self.action_dim = action_dim
#         self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
#                                        nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
#                                        nn.Linear(mid_dim, mid_dim), nn.Hardswish())
#         out_dim = mid_dim
#
#         self.net_ca_avg = nn.Linear(out_dim, action_dim[0])  # the average of action
#         self.net_ca_std = nn.Linear(out_dim, action_dim[0])  # the log_std of action
#
#         self.net_da = nn.Linear(out_dim, action_dim[1])
#         self.soft_max = nn.Softmax(dim=-1)
#         self.Categorical = torch.distributions.Categorical
#
#         self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
#         layer_norm(self.net_ca_avg, std=0.01)  # output layer for action, it is no necessary.
#
#     def forward(self, state):
#         tmp = self.net_state(state)
#         da_prob = self.soft_max(self.net_da(tmp))
#         da = da_prob.argmax(dim=1).unsqueeze(dim=1)
#         return torch.cat((self.net_ca_avg(tmp).tanh(), da), 1)  # action
#
#     def get_action(self, state):
#         t_tmp = self.net_state(state)
#         a_avg = self.net_ca_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
#         a_std = self.net_ca_std(t_tmp).clamp(-20, 2).exp()  # todo
#         da_prob = self.soft_max(self.net_da(t_tmp))
#         da_sample = torch.multinomial(da_prob, num_samples=1, replacement=True).squeeze(dim=1)
#         return torch.cat((torch.normal(a_avg, a_std).tanh()[0], da_sample), 0)  # re-parameterize
#
#     def get_action_logprob(self, state):
#         t_tmp = self.net_state(state)
#         a_avg = self.net_ca_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
#         a_std_log = self.net_ca_std(t_tmp).clamp(-20, 2)  # todo (-20, 2)
#         a_std = a_std_log.exp()
#         da_prob = self.soft_max(self.net_da(t_tmp))
#
#         """add noise to action in stochastic policy"""
#         noise = torch.randn_like(a_avg, requires_grad=True)
#         action = a_avg + a_std * noise
#         a_tan = action.tanh()  # action.tanh()
#         '''compute logprob according to mean and std of action (stochastic policy)'''
#         delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)
#         logprob = a_std_log + self.sqrt_2pi_log + delta
#         logprob = logprob + (-a_tan.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
#
#         # da = torch.multinomial(da_prob, num_samples=1, replacement=True).squeeze(dim=1)
#         da = da_prob.argmax(dim=1)
#         da_onehot = torch.nn.functional.one_hot(da, self.action_dim[1])
#         return torch.cat((torch.normal(a_avg, a_std).tanh(), da_onehot), 1), \
#                logprob.sum(1, keepdim=True) + da_prob.max(dim=1)[0].unsqueeze(dim=1).log()
#         # return torch.cat((torch.normal(a_avg, a_std).tanh(), da_prob), 1), \
#         #        logprob.sum(1, keepdim=True) * da_prob.max(dim=1)[0].unsqueeze(dim=1)


# class ActorSACMG(nn.Module):
#     def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
#         super().__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
#                                        nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
#                                        nn.Linear(mid_dim, mid_dim), nn.Hardswish())
#
#         self.net_a_loc = nn.Linear(mid_dim, action_dim)  # the average of action
#         self.net_a_cholesky = nn.Linear(mid_dim, (action_dim * (action_dim + 1)) // 2)
#         self.softplus = nn.Softplus(threshold=18.)
#         layer_norm(self.net_a_loc, std=0.01)  # output layer for action, it is no necessary.
#         # layer_norm(self.net_a_cholesky, std=0.01)
#
#     def forward(self, state):
#         return self.net_a_loc(self.net_state(state)).tanh()  # action
#
#     def get_distribution(self, state):
#         loc, cholesky = self.get_loc_cholesky(state)
#         return MultivariateNormal(loc=loc, scale_tril=cholesky)
#
#     def get_loc_cholesky(self, state):
#         t_tmp = self.net_state(state)
#         a_loc = self.net_a_loc(t_tmp)  # NOTICE! it is a_loc without .tanh()
#         # a_cholesky_vector = self.softplus(self.net_a_cholesky(t_tmp).relu())
#         a_cholesky_vector = self.softplus(self.net_a_cholesky(t_tmp))
#         # a_cholesky_vector = self.net_a_cholesky(t_tmp).clamp(-20, 2).exp()
#         # cholesky_diag_index = torch.arange(self.action_dim, dtype=torch.long) + 1
#         # cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
#         # a_cholesky_vector[:, cholesky_diag_index] = self.softplus(a_cholesky_vector[:, cholesky_diag_index])
#         tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
#         a_cholesky = torch.zeros(size=(a_loc.shape[0], self.action_dim, self.action_dim), dtype=torch.float32,
#                                  device=t_tmp.device)
#         a_cholesky[:, tril_indices[0], tril_indices[1]] = a_cholesky_vector
#         return a_loc, a_cholesky
#
#     def get_action(self, state):
#         pi = self.get_distribution(state)
#         return pi.sample().tanh()  # re-parameterize
#
#     def get_action_logprob(self, state):
#         a_loc, a_cholesky = self.get_loc_cholesky(state)
#         pi = MultivariateNormal(loc=a_loc, scale_tril=a_cholesky)
#         action = pi.rsample()
#         a_tan = action.tanh()
#         log_prob = pi.log_prob(action).unsqueeze(im=1) + (-a_tan.pow(2) + 1.000001).log()
#         return a_tan, log_prob.mean(dim=1, keepdim=True)


# class ActorMPO(nn.Module):
#     def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
#         super().__init__()
#         self.action_dim = action_dim
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if if_use_dn:  # use a DenseNet (DenseNet has both shallow and deep linear layer)
#             nn_dense_net = DenseNet(mid_dim)
#             self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
#                                            nn_dense_net, )
#             lay_dim = nn_dense_net.out_dim
#         else:  # use a simple network. Deeper network does not mean better performance in RL.
#             lay_dim = mid_dim
#             self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
#                                            nn.Linear(mid_dim, lay_dim), nn.ReLU(),
#                                            nn.Linear(mid_dim, lay_dim), nn.ReLU())
#             # nn.Linear(mid_dim, lay_dim), nn.Hardswish())
#         self.net_a_avg = nn.Linear(lay_dim, action_dim)  # the average of action
#         self.cholesky_layer = nn.Linear(lay_dim, (action_dim * (action_dim + 1)) // 2)
#
#         layer_norm(self.net_a_avg, std=0.01)  # output layer for action, it is no necessary.
#         layer_norm(self.cholesky_layer, std=0.01)
#
#     def forward(self, state):
#         return self.net_a_avg(self.net_state(state)).tanh()  # action
#
#     def get_distribution(self, state):
#         a_avg, cholesky = self.get_loc_cholesky(state)
#         return MultivariateNormal(loc=a_avg, scale_tril=cholesky)
#
#     def get_loc_cholesky(self, state):
#         t_tmp = self.net_state(state)
#         a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
#         cholesky_vector = self.cholesky_layer(t_tmp)
#         cholesky_diag_index = torch.arange(self.action_dim, dtype=torch.long) + 1
#         cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
#         cholesky_vector[:, cholesky_diag_index] = F.softplus(cholesky_vector[:, cholesky_diag_index])
#         tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
#         cholesky = torch.zeros(size=(a_avg.shape[0], self.action_dim, self.action_dim), dtype=torch.float32,
#                                device=t_tmp.device)
#         cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
#         return a_avg, cholesky
#
#     def get_action(self, state):
#         pi_action = self.get_distribution(state)
#         return pi_action.sample().tanh()  # re-parameterize
#
#     def get_actions(self, state, sampled_actions_num):
#         pi_action = self.get_distribution(state)
#         return pi_action.sample((sampled_actions_num,)).tanh()


'''Value Network (Critic)'''


class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # Q value


class CriticAdv(nn.Module):
    def __init__(self, state_dim, mid_dim, if_use_dn=False):
        super().__init__()
        if isinstance(state_dim, int):
            if if_use_dn:
                nn_dense = DenseNet(mid_dim)
                inp_dim = nn_dense.inp_dim
                out_dim = nn_dense.out_dim

                self.net = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
                                         nn_dense,
                                         nn.Linear(out_dim, 1), )
            else:
                self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                         nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                         # nn.Linear(mid_dim, mid_dim), nn.ReLU(),  # nn.Hardswish(),
                                         nn.Linear(mid_dim, 1), )

        else:
            def set_dim(i):
                return int(12 * 1.5 ** i)

            self.net = nn.Sequential(NnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                                     nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                                     nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                                     NnReshape(-1),
                                     nn.Linear(set_dim(5), mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, 1))

        layer_norm(self.net[-1], std=0.5)  # output layer for Q value

    def forward(self, state):
        return self.net(state)  # Q value


class CriticAdv_Multi(nn.Module):
    def __init__(self, state_dim, mid_dim, reward_dim, if_shared=False):
        super().__init__()
        self.if_shared = if_shared
        if self.if_shared:
            self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                     # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim), nn.ReLU(), )

            self.tails = nn.ModuleList([nn.Linear(mid_dim, 1) for _ in range(reward_dim)])
            [layer_norm(self.tails[i], std=0.5) for i in range(reward_dim)]  # output layer for Q value
        else:
            self.tails = nn.ModuleList([nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                                      nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                                      nn.Linear(mid_dim, 1)) for _ in range(reward_dim)])
            [layer_norm(self.tails[i][-1], std=0.5) for i in range(reward_dim)]

    def forward(self, state):
        if self.if_shared:
            tmp = self.net(state)
            return torch.cat([self.tails[i](tmp) for i in range(len(self.tails))], dim=1)  # Multi Q
        else:
            return torch.cat([self.tails[i](state) for i in range(len(self.tails))], dim=1)  # Multi Q value


class CriticTwin(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()

        if if_use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            nn_middle = DenseNet(mid_dim)
            inp_dim = nn_middle.inp_dim
            out_dim = nn_middle.out_dim
        else:  # use a simple network for actor. Deeper network does not mean better performance in RL.
            nn_middle = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU())
            inp_dim = mid_dim
            out_dim = mid_dim

        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, inp_dim), nn.ReLU(),
                                    nn_middle, )  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q2 value

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


class HybridCriticTwin(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()

        if if_use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            nn_dense = DenseNet(mid_dim)
            inp_dim = nn_dense.inp_dim
            out_dim = nn_dense.out_dim

            self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim[0] + action_dim[1], inp_dim), nn.ReLU(),
                                        nn_dense, )  # state-action value function
        else:  # use a simple network for actor. Deeper network does not mean better performance in RL.
            self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim[0] + action_dim[1], mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU())
            out_dim = mid_dim

        self.net_q1 = nn.Linear(out_dim, 1)
        self.net_q2 = nn.Linear(out_dim, 1)
        layer_norm(self.net_q1, std=0.1)
        layer_norm(self.net_q2, std=0.1)

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


"""utils"""


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, mid_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(mid_dim // 2, mid_dim // 2), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(mid_dim * 1, mid_dim * 1), nn.Hardswish())
        self.inp_dim = mid_dim // 2
        self.out_dim = mid_dim * 2

    def forward(self, x1):  # x1.shape == (-1, mid_dim // 2)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3  # x3.shape == (-1, mid_dim * 2)


class ConcatNet(nn.Module):  # concatenate
    def __init__(self, mid_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.dense2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.dense3 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.dense4 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.out_dim = mid_dim * 4

    def forward(self, x0):
        x1 = self.dense1(x0)
        x2 = self.dense2(x0)
        x3 = self.dense3(x0)
        x4 = self.dense4(x0)

        return torch.cat((x1, x2, x3, x4), dim=1)


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


def lstm_layer_norm(layer, hidden_size):
    # std = 0.1
    # torch.nn.init.orthogonal(layer.weight_ih, std)
    # torch.nn.init.orthogonal(layer.weight_hh, std)
    # from torch.nn import Parameter
    # layer.bias_ih = Parameter(torch.zeros_like(layer.bias_ih))
    # layer.bias_hh = Parameter(torch.zeros_like(layer.bias_hh))

    k = np.sqrt(1 / hidden_size)
    torch.nn.init.uniform_(layer.weight_ih, -k, k)
    torch.nn.init.uniform_(layer.weight_hh, -k, k)
    torch.nn.init.uniform_(layer.bias_ih, -k, k)
    torch.nn.init.uniform_(layer.bias_hh, -k, k)
