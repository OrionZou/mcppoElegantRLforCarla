import os
import torch
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import numpy.random as rd
import random
from copy import deepcopy
from ray_elegantrl.net import *
from ray_elegantrl.intrinsic_reward import IReward
from scipy.optimize import minimize

# from IPython import embed

"""
Modify [ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)
by https://github.com/GyChou
"""


class AgentBase:
    def __init__(self, args=None):
        self.learning_rate = 1e-4 if args is None else args['learning_rate']
        self.soft_update_tau = 2 ** -8 if args is None else args['soft_update_tau']  # 5e-3 ~= 2 ** -8
        self.state = None  # set for self.update_buffer(), initialize before training
        self.device = None

        self.act = self.act_target = None
        self.cri = self.cri_target = None
        self.act_optimizer = None
        self.cri_optimizer = None
        self.criterion = None
        self.get_obj_critic = None
        self.train_record = {}

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        `int net_dim` the dimension of networks (the width of neural networks)
        `int state_dim` the dimension of state (the number of state vector)
        `int action_dim` the dimension of action (the number of discrete action)
        `bool if_per` Prioritized Experience Replay for sparse reward
        """

    @property
    def policy(self):
        return self.act

    @staticmethod
    def select_action(state, policy, explore_rate=1.) -> np.ndarray:
        """Select actions for exploration; run on cpu

        :array state: state.shape==(state_dim, )
        :return array action: action.shape==(action_dim, ), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor((state,), dtype=torch.float32).detach_()
        if rd.rand() < explore_rate:  # epsilon-greedy
            action = policy.get_action(states)[0]
        else:
            action = policy(states)[0]
        return action.cpu().numpy()

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """update the neural network by sampling batch data from ReplayBuffer

        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning

        `buffer` Experience replay buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explore target_step number of step in env
        `int batch_size` sample batch_size of data for Stochastic Gradient Descent
        :float repeat_times: the times of sample batch = int(target_step * repeat_times) in off-policy
        :return float obj_a: the objective value of actor
        :return float obj_c: the objective value of critic
        """

    def save_model(self, cwd):
        act_save_path = f'{cwd}/actor.pth'
        cri_save_path = f'{cwd}/critic.pth'
        self.to_cpu()
        if self.act is not None:
            torch.save(self.act.state_dict(), act_save_path)
        if self.cri is not None:
            torch.save(self.cri.state_dict(), cri_save_path)

    def load_model(self, cwd):
        act_save_path = f'{cwd}/actor.pth'
        cri_save_path = f'{cwd}/critic.pth'

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)

        if (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        self.to_device()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network

        :nn.Module target_net: target network update via a current network, it is more stable
        :nn.Module current_net: current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))

    def update_record(self, **kwargs):
        """update the self.train_record for recording the metrics in training process
        :**kwargs :named arguments is the metrics name, arguments value is the metrics value.
        both of them will be prined and showed in tensorboard
        """
        self.train_record.update(kwargs)

    def to_cpu(self):
        device = torch.device('cpu')
        if next(self.act.parameters()).is_cuda:
            self.act.to(device)
        if next(self.cri.parameters()).is_cuda:
            self.cri.to(device)

    def to_device(self):
        if not next(self.act.parameters()).is_cuda:
            self.act.to(self.device)
        if not next(self.cri.parameters()).is_cuda:
            self.cri.to(self.device)


'''Value-based Methods (DQN variances)'''


class AgentDQN(AgentBase):
    def __init__(self, args=None):
        super().__init__(args)
        # the probability of choosing action randomly in epsilon-greedy
        self.explore_rate = 0.1 if args is None else args['explore_rate']
        self.action_dim = None  # chose discrete action randomly in epsilon-greedy

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNet(net_dim, state_dim, action_dim).to(self.device)
        self.cri.explore_rate = self.explore_rate
        self.cri_target = deepcopy(self.cri)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        self.act = self.cri  # to keep the same from Actor-Critic framework

        self.criterion = torch.nn.MSELoss(reduction='none' if if_per else 'mean')
        if if_per:
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.get_obj_critic = self.get_obj_critic_raw

    @staticmethod
    def select_action(state, policy, explore_rate=1.) -> int:  # for discrete action space
        states = torch.as_tensor((state,), dtype=torch.float32).detach_()
        a_prob = policy.get_a_prob(states)[0]
        explore_rate = policy.explore_rate if explore_rate > policy.explore_rate else explore_rate
        if rd.rand() < explore_rate:  # epsilon-greedy
            a_prob = a_prob.detach().numpy()  # choose action according to Q value
            a_int = rd.choice(a_prob.shape[0], p=a_prob)
        else:
            a_int = a_prob.argmax(dim=0).detach().numpy()
        return a_int

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        q_value = obj_critic = None
        for _ in range(int(target_step * repeat_times)):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        self.update_record(obj_a=q_value.mean().item(), obj_c=obj_critic.item())
        return self.train_record

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q

        q_value = self.cri.get_q1(state).gather(1, action.type(torch.long))
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q

        q_value = self.cri.get_q1(state).gather(1, action.type(torch.long))
        obj_critic = (self.criterion(q_value, q_label) * is_weights).mean()
        return obj_critic, q_value


class AgentDuelingDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNetDuel(net_dim, state_dim, action_dim).to(self.device)
        self.cri.explore_rate = self.explore_rate
        self.cri_target = deepcopy(self.cri)
        self.act = self.cri

        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss(reduction='none' if if_per else 'mean')
        self.get_obj_critic = self.get_obj_critic_per if if_per else self.get_obj_critic_raw


class AgentDoubleDQN(AgentDQN):
    def __init__(self, args=None):
        super().__init__(args)
        # the probability of choosing action randomly in epsilon-greedy
        self.explore_rate = 0.25 if args is None else args['explore_rate']
        self.softmax = torch.nn.Softmax(dim=1)

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNetTwin(net_dim, state_dim, action_dim).to(self.device)
        self.cri.explore_rate = self.explore_rate
        self.cri_target = deepcopy(self.cri)
        self.cri_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act = self.cri
        self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per else 'mean')
        self.get_obj_critic = self.get_obj_critic_per if if_per else self.get_obj_critic_raw

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s))
            next_q = next_q.max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q
        act_int = action.type(torch.long)
        q1, q2 = [qs.gather(1, act_int) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s))
            next_q = next_q.max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q
        act_int = action.type(torch.long)
        q1, q2 = [qs.gather(1, act_int) for qs in self.act.get_q1_q2(state)]
        obj_critic = ((self.criterion(q1, q_label) + self.criterion(q2, q_label)) * is_weights).mean()
        return obj_critic, q1


class AgentD3QN(AgentDoubleDQN):  # D3QN: Dueling Double DQN
    def __init__(self, args=None):
        super().__init__(args)

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNetTwinDuel(net_dim, state_dim, action_dim).to(self.device)
        self.cri.explore_rate = self.explore_rate
        self.cri_target = deepcopy(self.cri)
        self.act = self.cri

        self.criterion = torch.nn.SmoothL1Loss(reduction='none') if if_per else torch.nn.SmoothL1Loss()
        self.cri_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per else 'mean')
        self.get_obj_critic = self.get_obj_critic_per if if_per else self.get_obj_critic_raw


'''Actor-Critic Methods (Policy Gradient)'''


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.ou_explore_noise = 0.3  # explore noise of action
        self.ou_noise = None

    def init(self, net_dim, state_dim, action_dim, if_per=False):
        self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.ou_explore_noise)
        # I don't recommend to use OU-Noise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = Critic(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.act = Actor(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per else 'mean')
        if if_per:
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.get_obj_critic = self.get_obj_critic_raw

    @staticmethod
    def select_action(state, policy, explore_rate=1.) -> np.ndarray:
        ou_explore_noise = 0.3
        states = torch.as_tensor((state,), dtype=torch.float32).detach_()
        action = policy.get_action(states)[0]
        if rd.rand() < explore_rate:  # epsilon-greedy
            ou_noise = OrnsteinUhlenbeckNoise(size=action.shape[0], sigma=ou_explore_noise)
            action = (action + ou_noise())
        action = action.clamp(-1, 1)
        return action.detach().numpy()

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        obj_critic = obj_actor = None  # just for print return
        for _ in range(int(target_step * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            q_value_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, q_value_pg).mean()  # obj_actor
            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()

            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        self.update_record(obj_a=obj_actor.item(), obj_c=obj_critic.item())
        return self.train_record

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = (self.criterion(q_value, q_label) * is_weights).mean()

        td_error = (q_label - q_value.detach()).abs()
        buffer.td_error_update(td_error)
        return obj_critic, state


class AgentTD3(AgentDDPG):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1  # standard deviation of explore noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency, for soft target update

    def init(self, net_dim, state_dim, action_dim, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = CriticTwin(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.act = Actor(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per else 'mean')
        if if_per:
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.get_obj_critic = self.get_obj_critic_raw

    @staticmethod
    def select_action(state, policy, explore_rate=1.) -> np.ndarray:
        explore_noise = 0.1
        states = torch.as_tensor((state,), dtype=torch.float32).detach_()
        action = policy(states)[0]
        if rd.rand() < explore_rate:  # epsilon-greedy
            action = (action + torch.randn_like(action) * explore_noise)
        action = action.clamp(-1, 1)
        return action.detach().numpy()

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        obj_critic = obj_actor = None
        for i in range(int(target_step * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            if i % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            q_value_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, q_value_pg).mean()  # obj_actor
            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()
            if i % self.update_freq == 0:  # delay update
                self.soft_update(self.act_target, self.act, self.soft_update_tau)

        self.update_record(obj_a=obj_actor.item(), obj_c=obj_critic.item() / 2)
        return self.train_record

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        """Prioritized Experience Replay

        Contributor: Github GyChou
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = ((self.criterion(q1, q_label) + self.criterion(q2, q_label)) * is_weights).mean()

        td_error = (q_label - torch.min(q1, q1).detach()).abs()
        buffer.td_error_update(td_error)
        return obj_critic, state


class AgentSAC(AgentBase):
    def __init__(self, args=None):
        super().__init__(args)
        self.alpha_log = None
        self.alpha_optimizer = None
        # * np.log(action_dim)
        self.target_entropy = 1.0

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_entropy *= np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), self.learning_rate)

        self.cri = CriticTwin(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.act = ActorSAC(net_dim, state_dim, action_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per else 'mean')
        if if_per:
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.get_obj_critic = self.get_obj_critic_raw

    @staticmethod
    def select_action(state, policy, explore_rate=1.) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32).detach_()
        if rd.rand() < explore_rate:  # epsilon-greedy
            action = policy.get_action(states)[0]
        else:
            action = policy(states)[0]
        return action.detach().numpy()

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        alpha = self.alpha_log.exp().detach()
        obj_critic = None
        for _ in range(int(target_step * repeat_times)):
            '''objective of critic'''
            obj_critic, state, action = self.get_obj_critic(buffer, batch_size, alpha)
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient

            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            obj_alpha.backward()
            self.alpha_optimizer.step()

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            obj_actor = -(torch.min(*self.cri_target.get_q1_q2(state, action_pg)) + logprob * alpha).mean()

            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()

        self.update_record(obj_a=obj_actor,
                           obj_c=obj_critic,
                           alpha=alpha.item(),
                           a0_avg=action[:, 0].mean().item(),
                           a1_avg=action[:, 1].mean().item(),
                           a0_std=action[:, 0].std().item(),
                           a1_std=action[:, 1].std().item(),
                           )
        return self.train_record

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a, next_logprob = self.act.get_action_logprob(next_s)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
            q_label = reward + mask * (next_q + next_logprob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)  # twin critics
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, state, action

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_a, next_logprob = self.act.get_action_logprob(next_s)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
            q_label = reward + mask * (next_q + next_logprob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)  # twin critics
        obj_critic = ((self.criterion(q1, q_label) + self.criterion(q2, q_label)) * is_weights).mean()

        td_error = (q_label - torch.min(q1, q1).detach()).abs()
        buffer.td_error_update(td_error)
        return obj_critic, state, action


class AgentModSAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
    def __init__(self, args=None):
        super().__init__(args)
        self.if_use_dn = True if args is None else args['if_use_dn']
        self.policy_type = None if args is None else args['policy_type']
        self.objective_type = None if args is None else args['objective_type']
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_entropy *= np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), self.learning_rate)

        self.cri = CriticTwin(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate)

        if self.policy_type == 'mg':
            self.act = ActorSACMG(net_dim, state_dim, action_dim).to(self.device)
        elif self.policy_type == 'beta':
            self.act = ActorSACBeta(net_dim, state_dim, action_dim).to(self.device)
        elif self.policy_type == 'beta2':
            self.act = ActorSACBeta2(net_dim, state_dim, action_dim).to(self.device)
        else:
            self.act = ActorSAC(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        # self.act = ActorSACMG(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)

        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per else 'mean')
        if if_per:
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.get_obj_critic = self.get_obj_critic_raw

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()
        train_steps = int(buffer.now_len * repeat_times / batch_size)
        # train_steps = int(target_step) # bad
        tar_act = deepcopy(self.act)
        obj_actor = None
        update_a = 0
        alpha = None
        for update_c in range(1, train_steps):
            alpha = self.alpha_log.exp()

            '''objective of critic (loss function of critic)'''
            obj_critic, state, action = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_c = 0.995 * self.obj_c + 0.0025 * obj_critic.item()  # for reliable_lambda
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.cri.parameters(), 4.0)
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient

            '''objective of alpha (temperature parameter automatic adjustment)'''
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            obj_alpha.backward()
            torch.nn.utils.clip_grad_norm_(self.alpha_log, 4.0)
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2).detach()

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = torch.min(*self.cri.get_q1_q2(state, a_noise_pg))
                obj_actor = -(q_value_pg + logprob * alpha).mean()

                self.act_optimizer.zero_grad()
                obj_actor.backward()
                torch.nn.utils.clip_grad_norm_(self.act.parameters(), 4.0)
                self.act_optimizer.step()
                self.soft_update(self.act_target, self.act, self.soft_update_tau)

        with torch.no_grad():
            _, _, _, latest_s = buffer.sample_latest(target_step)
            kl = torch.distributions.kl_divergence(tar_act.get_distribution(latest_s),
                                                   self.act.get_distribution(latest_s)).mean()

        if self.policy_type == 'beta2':
            action_explore = self.act.get_explore(state)
            self.update_record(obj_a=obj_actor,
                               obj_c=obj_critic,
                               alpha=alpha.item(),
                               _lambda=reliable_lambda,
                               a0_exp=action_explore[:, 0].mean().item(),
                               a1_exp=action_explore[:, 1].mean().item(),
                               a0_avg=action[:, 0].mean().item(),
                               a1_avg=action[:, 1].mean().item(),
                               a0_std=action[:, 0].std().item(),
                               a1_std=action[:, 1].std().item(),
                               )
        else:
            self.update_record(obj_a=obj_actor,
                               obj_c=self.obj_c,
                               alpha=alpha.item(),
                               _lambda=reliable_lambda,
                               mean_kl=100 * kl.mean().item(),
                               a0_avg=action[:, 0].mean().item(),
                               a1_avg=action[:, 1].mean().item(),
                               a0_std=action[:, 0].std().item(),
                               a1_std=action[:, 1].std().item(),
                               )
        return self.train_record

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a, next_logprob = self.act_target.get_action_logprob(next_s)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
            q_label = reward + mask * (next_q + next_logprob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)  # twin critics
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, state, action

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_a, next_logprob = self.act_target.get_action_logprob(next_s)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
            q_label = reward + mask * (next_q + next_logprob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)  # twin critics
        obj_critic = ((self.criterion(q1, q_label) + self.criterion(q2, q_label)) * is_weights).mean()

        td_error = (q_label - torch.min(q1, q1).detach()).abs()
        buffer.td_error_update(td_error)
        return obj_critic, state, action


# class AgentSACRS(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
#     def __init__(self, args=None):
#         super().__init__(args)
#         self.if_use_dn = True if args is None else args['if_use_dn']
#         self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda
#
#     def init(self, net_dim, state_dim, action_dim, reward_dim=2, if_per=False):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.target_entropy *= np.log(action_dim)
#         self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
#                                       requires_grad=True, device=self.device)  # trainable parameter
#         self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), self.learning_rate)
#
#         self.cri = CriticTwin(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
#         self.cri_target = deepcopy(self.cri)
#         self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate)
#
#         self.cri_a = CriticTwin(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
#         self.cri_a_target = deepcopy(self.cri)
#         self.cri_a_optimizer = torch.optim.Adam(self.cri_a.parameters(), self.learning_rate)
#
#         self.act = ActorSAC(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
#         # self.act = ActorSACMG(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
#         self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
#
#         self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per else 'mean')
#         if if_per:
#             self.get_obj_critic = self.get_obj_critic_per
#         else:
#             self.get_obj_critic = self.get_obj_critic_raw
#
#     def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
#         buffer.update_now_len_before_sample()
#
#         k = 1.0 + buffer.now_len / buffer.max_len
#         batch_size_ = int(batch_size * k)
#         train_steps = int(target_step * k * repeat_times)
#
#         alpha = self.alpha_log.exp().detach()
#         update_a = 0
#         for update_c in range(1, train_steps):
#             '''objective of critic (loss function of critic)'''
#             obj_critic, obj_critic_a, state = self.get_obj_critic(buffer, batch_size_, alpha)
#             self.obj_c = 0.995 * self.obj_c + 0.0025 * obj_critic.item()  # for reliable_lambda
#             self.cri_optimizer.zero_grad()
#             obj_critic.backward()
#             self.cri_optimizer.step()
#             self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
#
#             self.cri_a_optimizer.zero_grad()
#             obj_critic_a.backward()
#             self.cri_a_optimizer.step()
#             self.soft_update(self.cri_a_target, self.cri_a, self.soft_update_tau)
#
#             '''objective of alpha (temperature parameter automatic adjustment)'''
#             action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
#
#             obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
#             self.alpha_optimizer.zero_grad()
#             obj_alpha.backward()
#             self.alpha_optimizer.step()
#
#             with torch.no_grad():
#                 self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
#             alpha = self.alpha_log.exp().detach()
#
#             '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
#             reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
#             if_update_a = (update_a / update_c) < (1 / (2 - reliable_lambda))
#             if if_update_a:  # auto TTUR
#                 update_a += 1
#
#                 q_value_pg = torch.min(*self.cri_target.get_q1_q2(state, action_pg))  # ceta3
#                 obj_actor = -(q_value_pg + logprob * alpha.detach()).mean()
#                 obj_actor = obj_actor * reliable_lambda  # max(0.01, reliable_lambda)
#
#                 self.act_optimizer.zero_grad()
#                 obj_actor.backward()
#                 self.act_optimizer.step()
#
#         self.update_record(obj_a=obj_actor,
#                            obj_c=self.obj_c,
#                            obj_ca=obj_critic_a,
#                            alpha=alpha.item(),
#                            _lambda=reliable_lambda)
#         return self.train_record
#
#     def get_obj_critic_raw(self, buffer, batch_size, alpha):
#         with torch.no_grad():
#             reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
#             next_a, next_logprob = self.act.get_action_logprob(next_s)
#             next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
#             next_q_a = torch.min(*self.cri_a_target.get_q1_q2(next_s, next_a))
#             f = mask[:, 0].unsqueeze(dim=1) * self.cri_a(next_s, next_a) - self.cri_a(state, action)
#             q_label = reward[:, 0].unsqueeze(dim=1) + mask[:, 0].unsqueeze(dim=1) * (next_q + next_logprob * alpha)
#             # q_label = reward[:, 0].unsqueeze(dim=1) + f + mask[:, 0].unsqueeze(dim=1) * (next_q + next_logprob * alpha)
#             qa_label = reward[:, 1].unsqueeze(dim=1) + mask[:, 1].unsqueeze(dim=1) * next_q_a
#
#         q1, q2 = self.cri.get_q1_q2(state, action)  # twin critics
#         qa1, qa2 = self.cri_a.get_q1_q2(state, action)  # twin critics
#         obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
#         obj_critic_a = self.criterion(qa1, qa_label) + self.criterion(qa2, qa_label)
#         return obj_critic, obj_critic_a, state


# class AgentHybridSAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
#     def __init__(self, args=None):
#         super().__init__(args)
#         self.if_use_dn = True if args is None else args['if_use_dn']
#         self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda
#
#     def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.action_dim = action_dim
#         self.target_entropy *= np.log(action_dim[0] + 1)
#         self.alpha_log = torch.tensor((-np.log(action_dim[0] + 1) * np.e,), dtype=torch.float32,
#                                       requires_grad=True, device=self.device)  # trainable parameter
#         self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), self.learning_rate)
#
#         self.cri = HybridCriticTwin(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
#         self.cri_target = deepcopy(self.cri)
#         self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate)
#
#         self.act = ActorHybridSAC(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
#         # self.act = ActorSACMG(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
#         self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
#
#         self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per else 'mean')
#         if if_per:
#             self.get_obj_critic = self.get_obj_critic_per
#         else:
#             self.get_obj_critic = self.get_obj_critic_raw
#
#     @staticmethod
#     def select_action(state, policy):
#         states = torch.as_tensor((state,), dtype=torch.float32).detach_()
#         return policy.get_action(states).detach().numpy()
#
#     def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
#         buffer.update_now_len_before_sample()
#
#         k = 1.0 + buffer.now_len / buffer.max_len
#         batch_size_ = int(batch_size * k)
#         train_steps = int(target_step * k * repeat_times)
#
#         alpha = self.alpha_log.exp().detach()
#         update_a = 0
#         for update_c in range(1, train_steps):
#             '''objective of critic (loss function of critic)'''
#             obj_critic, state = self.get_obj_critic(buffer, batch_size_, alpha)
#             self.obj_c = 0.995 * self.obj_c + 0.0025 * obj_critic.item()  # for reliable_lambda
#             self.cri_optimizer.zero_grad()
#             obj_critic.backward()
#             self.cri_optimizer.step()
#             self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
#
#             '''objective of alpha (temperature parameter automatic adjustment)'''
#             action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
#
#             obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
#             self.alpha_optimizer.zero_grad()
#             obj_alpha.backward()
#             self.alpha_optimizer.step()
#
#             with torch.no_grad():
#                 self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
#             alpha = self.alpha_log.exp().detach()
#
#             '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
#             reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
#             if_update_a = (update_a / update_c) < (1 / (2 - reliable_lambda))
#             if if_update_a:  # auto TTUR
#                 update_a += 1
#
#                 q_value_pg = torch.min(*self.cri_target.get_q1_q2(state, action_pg))  # ceta3
#                 obj_actor = -(q_value_pg + logprob * alpha.detach()).mean()
#                 obj_actor = obj_actor * reliable_lambda  # max(0.01, reliable_lambda)
#
#                 self.act_optimizer.zero_grad()
#                 obj_actor.backward()
#                 self.act_optimizer.step()
#
#         self.update_record(obj_a=obj_actor, obj_c=self.obj_c, alpha=alpha.item(), _lambda=reliable_lambda)
#         return self.train_record
#
#     def get_obj_critic_raw(self, buffer, batch_size, alpha):
#         with torch.no_grad():
#             reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
#             next_a, next_logprob = self.act.get_action_logprob(next_s)
#             next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
#             q_label = reward + mask * (next_q + next_logprob * alpha)
#         # q1, q2 = self.cri.get_q1_q2(state, action)  # twin critics
#         action = torch.cat((action[:, 0].unsqueeze(dim=1),
#                             torch.nn.functional.one_hot(action[:, -1].type(torch.long), self.action_dim[1])), 1)
#         q1, q2 = self.cri.get_q1_q2(state, action)
#         obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
#         return obj_critic, state


# class AgentHybrid2SAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
#     def __init__(self, args=None):
#         super().__init__(args)
#         self.if_use_dn = True if args is None else args['if_use_dn']
#         self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda
#
#     def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.action_dim = action_dim
#         self.target_entropy *= np.log(action_dim[0] + 1)
#         self.alpha_log = torch.tensor((-np.log(action_dim[0] + 1) * np.e,), dtype=torch.float32,
#                                       requires_grad=True, device=self.device)  # trainable parameter
#         self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), self.learning_rate)
#
#         self.cri = QNetTwinDuel(net_dim, state_dim + action_dim[0], action_dim[1]).to(self.device)
#         self.cri_target = deepcopy(self.cri)
#         self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate)
#
#         self.act_c = ActorSAC(net_dim, state_dim, action_dim[0], self.if_use_dn).to(self.device)
#         self.act_d = self.cri
#         self.act = [self.act_c, self.act_d]
#         self.act_optimizer = torch.optim.Adam(self.act_c.parameters(), self.learning_rate)
#         self.soft_max = torch.nn.Softmax(dim=1)
#         self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per else 'mean')
#         if if_per:
#             self.get_obj_critic = self.get_obj_critic_per
#         else:
#             self.get_obj_critic = self.get_obj_critic_raw
#
#     @staticmethod
#     def select_action(state, policy):
#         states = torch.as_tensor((state,), dtype=torch.float32).detach_()
#         action_c = policy[0].get_action(states)
#         action_d_prob = policy[1].get_a_prob(torch.cat((states, action_c), dim=1))[0].detach().numpy()
#         action_d_int = rd.choice(action_d_prob.shape[0], p=action_d_prob)
#         return np.hstack((action_c[0].detach().numpy(), action_d_int))
#
#     def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
#         buffer.update_now_len_before_sample()
#
#         k = 1.0 + buffer.now_len / buffer.max_len
#         batch_size_ = int(batch_size * k)
#         train_steps = int(target_step * k * repeat_times)
#
#         alpha = self.alpha_log.exp().detach()
#         update_a = 0
#         for update_c in range(1, train_steps):
#             '''objective of critic (loss function of critic)'''
#             obj_critic, state = self.get_obj_critic(buffer, batch_size_, alpha)
#             self.obj_c = 0.995 * self.obj_c + 0.0025 * obj_critic.item()  # for reliable_lambda
#             self.cri_optimizer.zero_grad()
#             obj_critic.backward()
#             self.cri_optimizer.step()
#             self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
#
#             '''objective of alpha (temperature parameter automatic adjustment)'''
#             action_c_pg, logprob_c = self.act_c.get_action_logprob(state)  # policy gradient
#             logprob = logprob_c + self.soft_max(self.act_d.get_a_prob(torch.cat((state, action_c_pg), dim=1))) \
#                 .max(dim=1, keepdim=True).values.log()
#
#             obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
#             self.alpha_optimizer.zero_grad()
#             obj_alpha.backward()
#             self.alpha_optimizer.step()
#
#             with torch.no_grad():
#                 self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
#             alpha = self.alpha_log.exp().detach()
#
#             '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
#             reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
#             if_update_a = (update_a / update_c) < (1 / (2 - reliable_lambda))
#             if if_update_a:  # auto TTUR
#                 update_a += 1
#
#                 q_value_pg = torch.min(*self.cri_target.get_q1_q2(torch.cat((state, action_c_pg), dim=1)))  # ceta3
#                 obj_actor = -(q_value_pg + logprob * alpha.detach()).mean()
#                 obj_actor = obj_actor * reliable_lambda  # max(0.01, reliable_lambda)
#
#                 self.act_optimizer.zero_grad()
#                 obj_actor.backward()
#                 self.act_optimizer.step()
#
#         self.update_record(obj_a=obj_actor, obj_c=self.obj_c, alpha=alpha.item(), _lambda=reliable_lambda)
#         return self.train_record
#
#     def get_obj_critic_raw(self, buffer, batch_size, alpha):
#         with torch.no_grad():
#             reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
#             next_a_c, next_logprob_c = self.act_c.get_action_logprob(next_s)
#             next_q = torch.min(*self.cri_target.get_q1_q2(torch.cat((next_s, next_a_c), dim=1)))
#             next_logprob = next_logprob_c + self.soft_max(next_q).max(dim=1, keepdim=True).values
#             next_q = next_q.max(dim=1, keepdim=True).values
#             q_label = reward + mask * (next_q + next_logprob * alpha)
#         # q1, q2 = self.cri.get_q1_q2(state, action)  # twin critics
#         q1, q2 = [qs.gather(1, action[:, -1].unsqueeze(dim=1).type(torch.long)) for qs in
#                   self.cri.get_q1_q2(torch.cat((state, action[:, :-1]), dim=1))]
#         obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
#         return obj_critic, state
#
#     def to_cpu(self):
#         device = torch.device('cpu')
#         if next(self.act[0].parameters()).is_cuda:
#             self.act[0].to(device)
#         if next(self.cri.parameters()).is_cuda:
#             self.cri.to(device)
#
#     def to_device(self):
#         if not next(self.act[0].parameters()).is_cuda:
#             self.act[0].to(self.device)
#         if not next(self.cri.parameters()).is_cuda:
#             self.cri.to(self.device)


class AgentPPO(AgentBase):
    def __init__(self, args=None):
        super().__init__(args)
        # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip),
        self.ratio_clip = 0.3 if args is None else args['ratio_clip']
        # could be 0.01 ~ 0.05
        self.lambda_entropy = 0.05 if args is None else args['lambda_entropy']
        # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.lambda_gae_adv = 0.97 if args is None else args['lambda_gae_adv']
        # if use Generalized Advantage Estimation
        self.if_use_gae = True if args is None else args['if_use_gae']
        # AgentPPO is an on policy DRL algorithm
        self.if_on_policy = True
        self.if_use_dn = False if args is None else args['if_use_dn']
        self.total_iterations = 1000 if args is None else args['total_iterations']
        self.loss_coeff_cri = 0.5
        self.objective_type = 'clip' if args is None else args['objective_type']
        self.beta = None if args is None else args['beta']
        self.policy_type = None if args is None else args['policy_type']

        self.optimizer = None
        self.compute_reward = None  # attribution

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        self.cri = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        if self.policy_type == 'mg':
            self.act = ActorPPOMG(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        else:
            self.act = ActorPPO(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])
        self.criterion = torch.nn.SmoothL1Loss()
        self.iter_index = 0
        assert if_per is False  # on-policy don't need PER

    @staticmethod
    def select_action(state, policy, explore_rate=1.) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32).detach_()
        if rd.rand() < explore_rate:  # epsilon-greedy
            action = policy.get_action(states)[0]
        else:
            action = policy(states)[0]
        return action.detach().numpy()

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = self.act.compute_logprob(buf_state, buf_action).unsqueeze(dim=1)
            buf_r_ret, buf_adv = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
            tar_act = deepcopy(self.act)
            del buf_reward, buf_mask

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        for _ in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_ret = buf_r_ret[indices]
            logprob = buf_logprob[indices]
            adv = buf_adv[indices]

            value = self.cri(state)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_ret)

            new_logprob = self.act.compute_logprob(state, action).unsqueeze(dim=1)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = adv * ratio
            if self.objective_type == 'clip':
                obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            elif self.objective_type == 'kl':
                mean_kl = torch.distributions.kl_divergence(tar_act.get_distribution(state),
                                                            self.act.get_distribution(state)).mean()
                obj_surrogate = -obj_surrogate1.mean() + self.beta * mean_kl
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            obj_united = obj_actor + obj_critic / (r_ret.std() + 1e-5) * self.loss_coeff_cri
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

        # self.iter_index += 1
        # ep_ratio = 1. - (self.iter_index / self.total_iterations)
        # self.ratio_clip = self.ratio_clip * ep_ratio
        kl = torch.distributions.kl_divergence(tar_act.get_distribution(state), self.act.get_distribution(state)).mean()
        self.update_record(obj_a=obj_surrogate.item(),
                           obj_c=obj_critic.item(),
                           mean_kl=100 * kl.mean().item(),
                           # a_std=self.act.a_std_log.exp().mean().item(),
                           entropy=(-obj_entropy.item()))

        return self.train_record

    def compute_reward_adv(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_ret = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_ret = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                                device=self.device)  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]
        buf_adv = buf_r_ret - (buf_mask * buf_value)
        buf_adv = (buf_adv - buf_adv.mean(dim=0)) / (buf_adv.std(dim=0) + 1e-5)
        return buf_r_ret, buf_adv

    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_ret = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_ret = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                                device=self.device)  # reward sum of previous step
        pre_adv = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                              device=self.device)  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]

            buf_adv[i] = buf_reward[i] + buf_mask[i] * pre_adv - buf_value[i]
            pre_adv = buf_value[i] + buf_adv[i] * self.lambda_gae_adv

        buf_adv = (buf_adv - buf_adv.mean(dim=0)) / (buf_adv.std(dim=0) + 1e-5)
        return buf_r_ret, buf_adv


class AgentPPO2(AgentPPO):
    def __init__(self, args=None):
        super().__init__(args)
        # AgentPPO is an on policy DRL algorithm
        self.if_on_policy = True
        # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip)
        if args is None:
            pass
        else:
            self.ratio_clip = args['ratio_clip'] if 'ratio_clip' in args.keys() else 0.3
            # could be 0.01 ~ 0.05
            self.lambda_entropy = args['lambda_entropy'] if 'lambda_entropy' in args.keys() else 0.05
            # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
            self.lambda_gae_adv = args['lambda_gae_adv'] if 'lambda_gae_adv' in args.keys() else 0.97
            # if use Generalized Advantage Estimation
            self.if_use_gae = args['if_use_gae'] if 'if_use_gae' in args.keys() else True
            self.if_use_dn = args['if_use_dn'] if 'if_use_dn' in args.keys() else False
            self.total_iterations = args['total_iterations'] if 'total_iterations' in args.keys() else 1000
            self.loss_coeff_cri = args['loss_coeff_cri'] if 'loss_coeff_cri' in args.keys() else 0.5
            self.objective_type = args['objective_type'] if 'objective_type' in args.keys() else 'clip'
            self.c_dclip = args['c_dclip'] if 'objective_type' in args.keys() and \
                                              args['objective_type'] == 'double_clip' else 3.
            self.if_ir = args['if_ir'] if 'if_ir' in args.keys() else None
            self.beta = args['beta'] if 'beta' in args.keys() else None
            self.policy_type = args['policy_type'] if 'policy_type' in args.keys() else None
            self.sp_a_num = args[
                'sp_a_num'] if self.policy_type == 'discrete_action_dim' and 'sp_a_num' in args.keys() else None

        self.target_entropy = None
        self.cri_optimizer = None
        self.act_optimizer = None
        self.compute_reward = None  # attribution

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv
        # self.target_entropy = np.log(action_dim)
        # self.target_entropy = -action_dim
        self.cri = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        if self.if_ir:
            self.irward_module = IReward(state_dim, action_dim=action_dim)
            # cwd="/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s45_a2_r1_tr700_ms1000/AgentPPO2_None_clip/exp_2021-10-09-10-35-21_cuda:0_rnd_nouseIR"
            # self.irward_module.save_load_model(cwd,False)
        if self.policy_type == 'discrete':
            self.act = ActorDiscretePPO(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        elif self.policy_type == 'discrete_action_dim':
            self.act = ActorSADPPO(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
            if self.act.sp_a_num is not None:
                self.act.set_sp_a_num(self.sp_a_num, device=self.device)
        elif self.policy_type == 'mg':
            self.act = ActorPPOMG(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        # elif self.policy_type == 'g':
        #     self.act = ActorPPOG(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        elif self.policy_type == 'beta':
            self.act = ActorPPOBeta(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        elif self.policy_type == 'beta2':
            self.act = ActorPPOBeta2(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        # elif self.policy_type == 'beta3':
        #     self.act = ActorPPOBeta3(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        else:
            self.act = ActorPPO(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)

        self.cri_optimizer = torch.optim.Adam(params=self.cri.parameters(), lr=self.learning_rate)
        self.act_optimizer = torch.optim.Adam(params=self.act.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()
        self.iter_index = 0
        assert if_per is False  # on-policy don't need PER

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_state = buffer.sample_all()
            if self.policy_type in ['discrete', 'beta', 'beta2']:
                buf_action = buf_action
            else:
                buf_action = buf_action.clamp(-1 + 5e-8, 1 - 5e-8)
            if self.if_ir:
                buf_reward, buf_ir = self.irward_module.calc_rnd(buf_state, buf_reward, buf_mask)
            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = self.act.compute_logprob(buf_state, buf_action).unsqueeze(dim=1)
            buf_r_ret, buf_adv = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
            tar_act = deepcopy(self.act)
            tar_cri = deepcopy(self.cri)
            del buf_reward, buf_mask

        '''update intrinsic reward'''
        if self.if_ir:
            ir_loss = self.irward_module.update_rnd(buf_state, buf_len)
        else:
            ir_loss = torch.zeros((1))

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        for _ in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len - 1, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            next_state = buf_state[indices + 1]
            action = buf_action[indices]
            r_ret = buf_r_ret[indices]
            logprob = buf_logprob[indices]
            adv = buf_adv[indices]

            value = self.cri(state)  # critic network predicts the reward_sum (Q value) of state

            if self.objective_type in ['tc_clip']:
                obj_critic = self.criterion(value, r_ret) + self.criterion(self.cri(next_state), tar_cri(next_state))
            elif self.objective_type in ['tc_clip2']:
                obj_critic = self.criterion(value, r_ret) + self.criterion(self.cri(buf_state[indices + 1]),
                                                                           buf_r_ret[indices + 1])
            else:
                obj_critic = self.criterion(value, r_ret)
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.cri.parameters(), 4.)
            self.cri_optimizer.step()

            new_logprob = self.act.compute_logprob(state, action).unsqueeze(dim=1)  # it is obj_actor
            ratio = (new_logprob - logprob).clamp(-20, 2).exp()
            obj_surrogate1 = adv * ratio
            if self.objective_type in ['clip', 'tc_clip', 'tc_clip2']:
                obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            elif self.objective_type == 'double_clip':
                obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.max(torch.min(obj_surrogate1, obj_surrogate2), self.c_dclip * adv).mean()
            elif self.objective_type == 'kl':
                mean_kl = torch.distributions.kl_divergence(tar_act.get_distribution(state),
                                                            self.act.get_distribution(state)).mean()
                obj_surrogate = -obj_surrogate1.mean() + self.beta * mean_kl
            elif self.objective_type == 'auto_kl':
                self.min_beta = 0.5
                self.max_beta = 5.
                self.kl_target = 0.5
                self.kl_alpha = 1.1
                mean_kl = torch.distributions.kl_divergence(tar_act.get_distribution(state),
                                                            self.act.get_distribution(state)).mean()
                if mean_kl > self.max_beta * self.kl_target:
                    self.beta = self.kl_alpha * self.beta
                elif mean_kl < self.min_beta * self.kl_target:
                    self.beta = self.beta / self.kl_alpha
                obj_surrogate = -obj_surrogate1.mean() + self.beta * mean_kl

            if self.target_entropy is not None:
                obj_entropy = (new_logprob.exp() * new_logprob - self.target_entropy).clamp_min(0).mean()
            else:
                obj_entropy = (new_logprob.exp() * new_logprob).mean()
            # action_explore = self.act.get_explore(buf_state)
            # action_explore = action_explore[:, 0] + action_explore[:, 1]
            # obj_entropy = (action_explore.log() * action_explore).mean()

            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            self.act_optimizer.zero_grad()
            obj_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.act.parameters(), 4.)
            self.act_optimizer.step()

        # self.iter_index += 1
        # ep_ratio = 1. - (self.iter_index / self.total_iterations)
        # self.ratio_clip = self.ratio_clip * ep_ratio
        tar_dist = tar_act.get_distribution(state)
        dist = self.act.get_distribution(state)
        if isinstance(tar_dist, list):
            kl = 0
            for i, tar_d in enumerate(tar_dist):
                kl += torch.distributions.kl_divergence(tar_d, dist[i]).mean()
        else:
            kl = torch.distributions.kl_divergence(tar_dist, dist).mean()

        if self.policy_type in ['beta2', 'beta', 'beta3']:
            with torch.no_grad():
                action_explore = self.act.get_explore(buf_state)
            self.update_record(obj_a=obj_surrogate.item(),
                               obj_c=obj_critic.item(),
                               mean_kl=100 * kl.mean().item(),
                               # a_std=self.act.a_std_log.exp().mean().item(),
                               entropy=(-obj_entropy.item()),
                               a0_exp=action_explore[:, 0].mean().item(),
                               a1_exp=action_explore[:, 1].mean().item(),
                               a0_avg=buf_action[:, 0].mean().item(),
                               a1_avg=buf_action[:, 1].mean().item(),
                               a0_std=buf_action[:, 0].std().item(),
                               a1_std=buf_action[:, 1].std().item(),
                               )
        elif self.if_ir:
            self.update_record(obj_a=obj_surrogate.item(),
                               obj_c=obj_critic.item(),
                               l_rnd=ir_loss.item(),
                               mean_kl=100 * kl.mean().item(),
                               # a_std=self.act.a_std_log.exp().mean().item(),
                               entropy=(-obj_entropy.item()),
                               a0_avg=buf_action[:, 0].mean().item(),
                               a1_avg=buf_action[:, 1].mean().item(),
                               a0_std=buf_action[:, 0].std().item(),
                               a1_std=buf_action[:, 1].std().item(),
                               m_ri=buf_ir.mean().item(),
                               )
        elif self.policy_type in ['discrete']:
            self.update_record(obj_a=obj_surrogate.item(),
                               obj_c=obj_critic.item(),
                               mean_kl=100 * kl.mean().item(),
                               entropy=(-obj_entropy.item()),
                               )
        else:
            self.update_record(obj_a=obj_surrogate.item(),
                               obj_c=obj_critic.item(),
                               mean_kl=100 * kl.mean().item(),
                               # a_std=self.act.a_std_log.exp().mean().item(),
                               entropy=(-obj_entropy.item()),
                               a0_avg=buf_action[:, 0].mean().item(),
                               a1_avg=buf_action[:, 1].mean().item(),
                               a0_std=buf_action[:, 0].std().item(),
                               a1_std=buf_action[:, 1].std().item(),
                               )

        return self.train_record

class AgentConstriantPPO2(AgentPPO2):  # add RNN version
    def __init__(self, args=None):
        super().__init__(args)
        # AgentPPO is an on policy DRL algorithm
        self.if_on_policy = True
        # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip)
        if args is None:
            pass
        else:
            self.ratio_clip = args['ratio_clip'] if 'ratio_clip' in args.keys() else 0.3
            self.beta = args['beta'] if 'beta' in args.keys() else 1.
            self.kl_target = args['kl_target'] if 'kl_target' in args.keys() else 0.01
            # could be 0.01 ~ 0.05
            self.lambda_entropy = args['lambda_entropy'] if 'lambda_entropy' in args.keys() else 0.05
            # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
            self.lambda_gae_adv = args['lambda_gae_adv'] if 'lambda_gae_adv' in args.keys() else 0.97
            # if use Generalized Advantage Estimation
            self.if_use_gae = args['if_use_gae'] if 'if_use_gae' in args.keys() else True
            self.if_use_dn = args['if_use_dn'] if 'if_use_dn' in args.keys() else False
            self.total_iterations = args['total_iterations'] if 'total_iterations' in args.keys() else 1000
            self.loss_coeff_cri = args['loss_coeff_cri'] if 'loss_coeff_cri' in args.keys() else 0.5
            self.objective_type = args['objective_type'] if 'objective_type' in args.keys() else 'clip'
            self.policy_type = args['policy_type'] if 'policy_type' in args.keys() else None

            # RNN
            self.if_rnn = args['if_rnn'] if 'if_rnn' in args.keys() else False
            if self.if_rnn:
                self.if_store_state = args['if_store_state'] if 'if_store_state' in args.keys() else True
                self.hidden_state_dim = args['hidden_state_dim'] if 'hidden_state_dim' in args.keys() else 128
                self.rnn_timestep = args['rnn_timestep'] if 'rnn_timestep' in args.keys() else 16
                self.embedding_dim = args['embedding_dim'] if 'embedding_dim' in args.keys() else 64
            # Multi Objective:
            self.if_critic_shared = args['if_critic_shared'] if 'if_critic_shared' in args.keys() else True
            self.if_auto_weights = args['if_auto_weights'] if 'if_auto_weights' in args.keys() else True
            self.weights = args['weights'] if 'weights' in args.keys() else None
            self.cost_threshold = args['cost_threshold'] if 'cost_threshold' in args.keys() else None
            if self.if_auto_weights:
                self.pid_Ki = args['pid_Ki'] if 'pid_Ki' in args.keys() else 0.01
                self.pid_Kp = args['pid_Kp'] if 'pid_Kp' in args.keys() else 0.25
                self.pid_Kd = args['pid_Kd'] if 'pid_Kd' in args.keys() else 4

        self.target_entropy = None
        self.cri_optimizer = None
        self.act_optimizer = None
        self.compute_reward = None  # attribution

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        self.state_dim = state_dim
        self.weights = np.ones(reward_dim) if self.weights is None else np.array(self.weights)
        self.weights = torch.as_tensor(self.weights, dtype=torch.float32, device=self.device)
        if (self.if_auto_weights) and (reward_dim > 1):
            self.cost_threshold = np.ones(reward_dim - 1) * 1e-6 if self.cost_threshold is None else np.array(
                self.cost_threshold)
            self.cost_threshold = torch.as_tensor(self.cost_threshold, dtype=torch.float32, device=self.device)
            self.pid_Ki = (torch.ones((reward_dim - 1)) * torch.tensor(self.pid_Ki)).to(
                dtype=torch.float32, device=self.device)
            self.pid_Kp = (torch.ones((reward_dim - 1)) * torch.tensor(self.pid_Kp)).to(
                dtype=torch.float32, device=self.device)
            self.pid_Kd = (torch.ones((reward_dim - 1)) * torch.tensor(self.pid_Kd)).to(
                dtype=torch.float32, device=self.device)
            self.pid_delta_p_ema_alpha = torch.ones((reward_dim - 1), dtype=torch.float32, device=self.device) * 0.95
            self.pid_delta_d_ema_alpha = torch.ones((reward_dim - 1), dtype=torch.float32, device=self.device) * 0.95
            self.pid_i = torch.zeros((reward_dim - 1), dtype=torch.float32, device=self.device)
            self._delta_p = torch.zeros((reward_dim - 1), dtype=torch.float32, device=self.device)
            self._cost_d = torch.zeros((reward_dim - 1), dtype=torch.float32, device=self.device)
            self._cost_d_pre = torch.zeros((reward_dim - 1), dtype=torch.float32, device=self.device)

        if self.if_rnn:
            self.rnn_embedding = RNNEmbedding(self.hidden_state_dim, state_dim, self.embedding_dim, self.if_store_state)
            state_dim = self.embedding_dim

        self.cri = CriticAdv_Multi(state_dim, net_dim, reward_dim, self.if_critic_shared).to(self.device)

        if self.policy_type == 'discrete':
            self.act = ActorDiscretePPO(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        elif self.policy_type == 'discrete_action_dim':
            self.act = ActorSADPPO(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
            if self.act.sp_a_num is not None:
                self.act.set_sp_a_num(self.sp_a_num, device=self.device)
        elif self.policy_type == 'mg':
            self.act = ActorPPOMG(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        elif self.policy_type == 'beta':
            self.act = ActorPPOBeta(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        elif self.policy_type == 'beta2':
            self.act = ActorPPOBeta2(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        else:
            self.act = ActorPPO(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        if self.if_rnn:
            self.total_optimizer = torch.optim.Adam([{"params": self.rnn_embedding.parameters()},
                                                     {"params": self.act.parameters()},
                                                     {"params": self.cri.parameters()}],
                                                    lr=self.learning_rate)
        else:
            self.cri_optimizer = torch.optim.Adam(params=self.cri.parameters(), lr=self.learning_rate)
            self.act_optimizer = torch.optim.Adam(params=self.act.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()
        self.iter_index = 0
        assert if_per is False  # on-policy don't need PER

    @property
    def policy(self):
        if self.if_rnn:
            return [self.rnn_embedding, self.act]
        else:
            return self.act

    @staticmethod
    def select_action(state, policy, hidden_state=None, cell_state=None, explore_rate=1.):
        states = torch.as_tensor((state,), dtype=torch.float32).detach_()
        if_rnn = 1 if isinstance(policy, list) else 0
        if if_rnn:
            if hidden_state is None or cell_state is None:
                hidden_state_dim = policy[0].hidden_state_dim
                hidden_state = torch.zeros([1, hidden_state_dim], dtype=torch.float32)
                cell_state = torch.zeros([1, hidden_state_dim], dtype=torch.float32)
            else:
                hidden_state = torch.as_tensor((hidden_state,), dtype=torch.float32).detach_()
                cell_state = torch.as_tensor((cell_state,), dtype=torch.float32).detach_()
            if rd.rand() < explore_rate:  # epsilon-greedy
                embedding, hidden_state_next, cell_state_next = policy[0].embedding_infer(states, hidden_state,
                                                                                          cell_state)
                action = policy[1].get_action(embedding)
            else:
                embedding, hidden_state_next, cell_state_next = policy[0].embedding_infer(states, hidden_state,
                                                                                          cell_state)
                action = policy[1](embedding)
            return action[0].detach().numpy(), \
                   hidden_state_next[0].detach().numpy(), \
                   cell_state_next[0].detach().numpy()
        else:
            if rd.rand() < explore_rate:  # epsilon-greedy
                action = policy.get_action(states)
            else:
                action = policy(states)
            return action[0].detach().numpy()

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            if self.if_rnn:
                buf_reward, buf_mask, buf_action, buf_state, buf_hidden_state, buf_cell_state = buffer.sample_all()
            else:
                buf_reward, buf_mask, buf_action, buf_state = buffer.sample_all()
            buf_reward[:, 1:] = -buf_reward[:, 1:]
            if self.policy_type in ['discrete', 'beta', 'beta2']:
                buf_action = buf_action
            else:
                buf_action = buf_action.clamp(-1 + 5e-8, 1 - 5e-8)

            if self.if_rnn:
                split_list = []
                idx = 0
                for i in range(buf_len):
                    if idx == self.rnn_timestep:
                        split_list.append([i - idx, i])
                        idx = 1
                    else:
                        if buf_mask[i].any() == 0:
                            split_list.append([i - idx, i])  # episode end <16 序列
                            idx = 1
                        else:
                            idx += 1
            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            if self.if_rnn:
                buf_embedding = torch.cat([self.rnn_embedding.embedding_infer(buf_state[i:i + bs],
                                                                              buf_hidden_state[i:i + bs],
                                                                              buf_cell_state[i:i + bs])[0] for i in
                                           range(0, buf_state.size(0), bs)], dim=0)
                buf_value = torch.cat([self.cri(buf_embedding[i:i + bs]) for i in range(0, buf_state.size(0), bs)],
                                      dim=0)
            else:
                buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            if self.if_rnn:
                buf_logprob = self.act.compute_logprob(buf_embedding, buf_action).unsqueeze(dim=1)
            else:
                buf_logprob = self.act.compute_logprob(buf_state, buf_action).unsqueeze(dim=1)
            buf_r_ret, buf_adv = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)

            if self.if_auto_weights:
                # self.auto_weights(buf_r_ret)
                self.auto_weights(buf_reward)
            tar_act = deepcopy(self.act)
            tar_cri = deepcopy(self.cri)
            # del buf_reward, buf_mask

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        if self.if_rnn:
            rnn_buf_len = len(split_list)
            batch_size = rnn_buf_len if rnn_buf_len < batch_size else batch_size
        else:
            target_idx = 200
            begin_batch_size = 16
            curr_batch_size = int(min(1, self.iter_index / target_idx) * batch_size)
            batch_size = curr_batch_size if curr_batch_size > begin_batch_size else begin_batch_size
        for _ in range(int(repeat_times * buf_len / batch_size)):
            if self.if_rnn:
                indices_list = random.sample(split_list, batch_size)
                indices = torch.cat([torch.arange(split[0], split[1], 1) for split in indices_list], dim=0).to(
                    self.device)
                state = torch.zeros((self.rnn_timestep, batch_size, self.state_dim), dtype=torch.float32,
                                    device=self.device)
                len_sequence = [split[1] - split[0] for split in indices_list]
                for i, split in enumerate(indices_list):
                    state[0:split[1] - split[0], i, :] = buf_state[split[0]:split[1], :]
                hidden_state = torch.cat([buf_hidden_state[split[0], :].unsqueeze(dim=0) for split in indices_list],
                                         dim=0).to(self.device)
                cell_state = torch.cat([buf_cell_state[split[0], :].unsqueeze(dim=0) for split in indices_list],
                                       dim=0).to(
                    self.device)
                state = self.rnn_embedding(state, hidden_state, cell_state, len_sequence)
            else:
                indices = torch.randint(buf_len - 1, size=(batch_size,), requires_grad=False, device=self.device)
                state = buf_state[indices]

            action = buf_action[indices]
            r_ret = buf_r_ret[indices]
            logprob = buf_logprob[indices]
            adv = buf_adv[indices]

            values = self.cri(state)  # critic network predicts the reward_sum (Q value) of state
            obj_critics = [self.criterion(values[:, i], r_ret[:, i]) for i in range(values.shape[1])]
            obj_critic = obj_critics[0]
            for obj_c in obj_critics[1:]:
                obj_critic += obj_c

            if obj_critic.isnan().any() or obj_critic.isinf().any():
                print('!!!')
            if not self.if_rnn:
                self.cri_optimizer.zero_grad()
                obj_critic.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.cri.parameters(), 4.)
                self.cri_optimizer.step()

            new_logprob = self.act.compute_logprob(state, action).unsqueeze(dim=1)  # it is obj_actor
            ratio = (new_logprob - logprob).clamp(-20, 2).exp()
            obj_surrogate1 = adv * ratio
            if self.objective_type in ['clip']:
                obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate_r = -torch.min(obj_surrogate1[:, 0], obj_surrogate2[:, 0]).mean(dim=0)
                obj_surrogate_c = torch.max(obj_surrogate1[:, 1:], obj_surrogate2[:, 1:]).mean(dim=0)
                obj_surrogate = torch.cat((obj_surrogate_r.unsqueeze(dim=0), obj_surrogate_c), dim=0)
            elif self.objective_type == 'kl':
                mean_kl = torch.distributions.kl_divergence(tar_act.get_distribution(state),
                                                            self.act.get_distribution(state)).mean()
                obj_surrogate_r = -obj_surrogate1[:, 0].mean(dim=0) + self.beta * mean_kl
                obj_surrogate_c = obj_surrogate1[:, 1:].mean(dim=0) + self.beta * mean_kl
                obj_surrogate = torch.cat((obj_surrogate_r.unsqueeze(dim=0), obj_surrogate_c), dim=0)
            elif self.objective_type == 'auto_kl':
                self.min_beta = 0.5
                self.max_beta = 5.
                self.kl_target = 0.5
                mean_kl = torch.distributions.kl_divergence(tar_act.get_distribution(state),
                                                            self.act.get_distribution(state)).mean()
                if mean_kl > 1.5 * self.kl_target:
                    self.beta = 2 * self.beta
                elif mean_kl < self.kl_target / 1.5:
                    self.beta = self.beta / 2
                obj_surrogate_r = -obj_surrogate1[:, 0].mean(dim=0) + self.beta * mean_kl
                obj_surrogate_c = obj_surrogate1[:, 1:].mean(dim=0) + self.beta * mean_kl
                obj_surrogate = torch.cat((obj_surrogate_r.unsqueeze(dim=0), obj_surrogate_c), dim=0)

            obj_surrogate = (obj_surrogate * (self.weights / self.weights.sum())).sum()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()
            # action_explore = self.act.get_explore(buf_state)
            # action_explore = action_explore[:, 0] + action_explore[:, 1]
            # obj_entropy = (action_explore.log() * action_explore).mean()

            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            if obj_actor.isnan().any() or obj_actor.isinf().any():
                print('!!!')
            if not self.if_rnn:
                self.act_optimizer.zero_grad()
                obj_actor.backward()
                torch.nn.utils.clip_grad_norm_(self.act.parameters(), 8.)
                self.act_optimizer.step()
            if self.if_rnn:
                L_total = obj_actor + obj_critic / (r_ret.std() + 1e-5) * self.loss_coeff_cri
                self.total_optimizer.zero_grad()
                L_total.backward()
                torch.nn.utils.clip_grad_norm_(self.act.parameters(), 4.)
                self.total_optimizer.step()

        self.iter_index += 1
        # ep_ratio = 1. - (self.iter_index / self.total_iterations)
        # self.ratio_clip = self.ratio_clip * ep_ratio
        if self.if_rnn:
            tar_dist = tar_act.get_distribution(buf_embedding)
            dist = self.act.get_distribution(buf_embedding)
        else:
            tar_dist = tar_act.get_distribution(buf_state)
            dist = self.act.get_distribution(buf_state)
        if isinstance(tar_dist, list):
            kl = 0
            for i, tar_d in enumerate(tar_dist):
                kl += torch.distributions.kl_divergence(tar_d, dist[i]).mean()
        else:
            kl = torch.distributions.kl_divergence(tar_dist, dist).mean()

        self.train_record['obj_a'] = obj_surrogate.item()
        for i, obj_critic in enumerate(obj_critics):
            self.train_record[f'obj_c{i}'] = obj_critic.item()
        for i in range(self.weights.shape[0]):
            self.train_record[f'w{i}'] = self.weights[i].item()
        self.train_record['kl100'] = 100 * kl.mean().item()
        self.train_record['entropy'] = -obj_entropy.item()
        if self.objective_type in ['auto_kl']:
            self.train_record['beta'] = self.beta
        for i in range(buf_action.shape[1]):
            self.train_record[f'a{i}_avg'] = buf_action[:, i].mean().item()
            self.train_record[f'a{i}_std'] = buf_action[:, i].std().item()

        return self.train_record

    def compute_reward_adv(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_ret = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_ret = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                                device=self.device)  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]
        buf_adv = buf_r_ret - (buf_mask * buf_value)
        buf_adv = (buf_adv - buf_adv.mean(dim=0)) / (buf_adv.std(dim=0) + 1e-5)
        return buf_r_ret, buf_adv

    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_ret = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_ret = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                                device=self.device)  # reward sum of previous step
        pre_adv = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                              device=self.device)  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]

            buf_adv[i] = buf_reward[i] + buf_mask[i] * pre_adv - buf_value[i]
            pre_adv = buf_value[i] + buf_adv[i] * self.lambda_gae_adv

        buf_adv = (buf_adv - buf_adv.mean(dim=0)) / (buf_adv.std(dim=0) + 1e-5)
        return buf_r_ret, buf_adv

    def auto_weights(self, buf_cost):
        # PID update here: CRL :cost instead nagetive_r
        cost_mean = buf_cost.mean(dim=0)[1:]
        delta = cost_mean - self.cost_threshold  # ep_cost_avg: tensor
        zeros = torch.zeros(cost_mean.shape, dtype=torch.float32, device=self.device)
        self.pid_i = torch.max(zeros, self.pid_i + delta * self.pid_Ki)
        a_p = self.pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self.pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * cost_mean
        pid_d = torch.max(zeros, self._cost_d - self._cost_d_pre)
        pid_o = (self.pid_Kp * self._delta_p + self.pid_i +
                 self.pid_Kd * pid_d)
        self.weights[1:] = torch.max(zeros, pid_o)
        self._cost_d_pre = self._cost_d

        self.weights[0] = max(self.weights[0], 2 * self.weights[1:].sum())

    def to_cpu(self):
        device = torch.device('cpu')
        if isinstance(self.policy, list):
            for net in self.policy:
                if next(net.parameters()).is_cuda:
                    net.to(device)
        else:
            if next(self.policy.parameters()).is_cuda:
                self.policy.to(device)
        if next(self.cri.parameters()).is_cuda:
            self.cri.to(device)

    def to_device(self):
        if isinstance(self.policy, list):
            for net in self.policy:
                if not next(net.parameters()).is_cuda:
                    net.to(self.device)
        else:
            if not next(self.policy.parameters()).is_cuda:
                self.policy.to(self.device)
        if not next(self.cri.parameters()).is_cuda:
            self.cri.to(self.device)


class AgentHybridPPO2(AgentPPO2):
    def __init__(self, args=None):
        super().__init__(args)
        # AgentPPO is an on policy DRL algorithm
        self.if_on_policy = True
        # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip)
        if args is None:
            pass
        else:
            self.ratio_clip = args['ratio_clip'] if 'ratio_clip' in args.keys() else 0.3
            # could be 0.01 ~ 0.05
            self.lambda_entropy = args['lambda_entropy'] if 'lambda_entropy' in args.keys() else 0.05
            # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
            self.lambda_gae_adv = args['lambda_gae_adv'] if 'lambda_gae_adv' in args.keys() else 0.97
            # if use Generalized Advantage Estimation
            self.if_use_gae = args['if_use_gae'] if 'if_use_gae' in args.keys() else True
            self.if_use_dn = args['if_use_dn'] if 'if_use_dn' in args.keys() else False
            self.total_iterations = args['total_iterations'] if 'total_iterations' in args.keys() else 1000
            self.loss_coeff_cri = args['loss_coeff_cri'] if 'loss_coeff_cri' in args.keys() else 0.5
            self.objective_type = args['objective_type'] if 'objective_type' in args.keys() else 'clip'

            self.if_ir = args['if_ir'] if 'if_ir' in args.keys() else None
            self.beta = args['beta'] if 'beta' in args.keys() else None
            self.policy_type = args['policy_type'] if 'policy_type' in args.keys() else None
            self.discrete_degree = args['discrete_degree'] if 'discrete_degree' in args.keys() else 3
            self.if_sp_action_loss = args['if_sp_action_loss'] if 'if_sp_action_loss' in args.keys() else False
            self.if_shared = args['if_shared'] if 'if_shared' in args.keys() else True

        self.target_entropy = None
        self.cri_optimizer = None
        self.act_optimizer = None
        self.compute_reward = None  # attribution

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv
        # self.target_entropy = np.log(action_dim)
        # self.target_entropy = -action_dim
        self.cri = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        self.act = ActorHybridPPO(net_dim, state_dim, [action_dim - 1, self.discrete_degree ** (action_dim - 1)],
                                  if_shared=self.if_shared).to(self.device)
        self.cri_optimizer = torch.optim.Adam(params=self.cri.parameters(), lr=self.learning_rate)
        self.act_optimizer = torch.optim.Adam(params=self.act.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()
        self.iter_index = 0
        assert if_per is False  # on-policy don't need PER

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_state = buffer.sample_all()
            buf_action[:, :-1] = buf_action[:, :-1].clamp(-1 + 5e-8, 1 - 5e-8)

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_c_logprob, buf_d_logprob = self.act.compute_logprob(buf_state, buf_action)
            buf_c_logprob = buf_c_logprob.unsqueeze(dim=1)
            buf_d_logprob = buf_d_logprob.unsqueeze(dim=1)
            if not self.if_sp_action_loss:
                buf_logprob = buf_c_logprob + buf_d_logprob
            buf_r_ret, buf_adv = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
            tar_act = deepcopy(self.act)
            tar_cri = deepcopy(self.cri)
            del buf_reward, buf_mask

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        for _ in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len - 1, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            next_state = buf_state[indices + 1]
            action = buf_action[indices]
            r_ret = buf_r_ret[indices]
            adv = buf_adv[indices]
            value = self.cri(state)  # critic network predicts the reward_sum (Q value) of state

            obj_critic = self.criterion(value, r_ret)
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.cri.parameters(), 4.)
            self.cri_optimizer.step()

            new_c_logprob, new_d_logprob = self.act.compute_logprob(state, action)  # it is obj_actor
            new_c_logprob = new_c_logprob.unsqueeze(dim=1)
            new_d_logprob = new_d_logprob.unsqueeze(dim=1)
            if self.if_sp_action_loss:
                c_logprob = buf_c_logprob[indices]
                d_logprob = buf_d_logprob[indices]
                ratio_c = (new_c_logprob - c_logprob).clamp(-20, 2).exp()
                ratio_d = (new_d_logprob - d_logprob).clamp(-20, 2).exp()
                c_obj_surrogate1 = adv * ratio_c
                d_obj_surrogate1 = adv * ratio_d

                c_obj_surrogate2 = adv * ratio_c.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                c_obj_surrogate = -torch.min(c_obj_surrogate1, c_obj_surrogate2).mean()
                d_obj_surrogate2 = adv * ratio_d.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                d_obj_surrogate = -torch.min(d_obj_surrogate1, d_obj_surrogate2).mean()

                obj_entropy_c = (new_c_logprob.exp() * new_c_logprob).mean()
                obj_entropy_d = (new_d_logprob.exp() * new_d_logprob).mean()

                # action_explore = self.act.get_explore(buf_state)
                # action_explore = action_explore[:, 0] + action_explore[:, 1]
                # obj_entropy = (action_explore.log() * action_explore).mean()
                obj_surrogate = c_obj_surrogate + d_obj_surrogate
                obj_entropy = obj_entropy_c + obj_entropy_d
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            else:
                logprob = buf_logprob[indices]
                new_logprob = new_c_logprob + new_d_logprob
                ratio = (new_logprob - logprob).clamp(-20, 2).exp()
                obj_surrogate1 = adv * ratio
                if self.objective_type in ['clip']:
                    obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                    obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
                elif self.objective_type == 'kl':
                    mean_kl = torch.distributions.kl_divergence(tar_act.get_distribution(state),
                                                                self.act.get_distribution(state)).mean()
                    obj_surrogate = -obj_surrogate1.mean() + self.beta * mean_kl

                if self.target_entropy is not None:
                    obj_entropy = (new_logprob.exp() * new_logprob - self.target_entropy).clamp_min(0).mean()
                else:
                    obj_entropy = (new_logprob.exp() * new_logprob).mean()
                # action_explore = self.act.get_explore(buf_state)
                # action_explore = action_explore[:, 0] + action_explore[:, 1]
                # obj_entropy = (action_explore.log() * action_explore).mean()

                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            self.act_optimizer.zero_grad()
            obj_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.act.parameters(), 4.)
            self.act_optimizer.step()
            if self.act.a_std_log.isnan().any() or self.act.a_std_log.isinf().any():
                print("test")

        # self.iter_index += 1
        # ep_ratio = 1. - (self.iter_index / self.total_iterations)
        # self.ratio_clip = self.ratio_clip * ep_ratio
        tar_c_dist, tar_d_dist = tar_act.get_distribution(state)
        c_dist, d_dist = self.act.get_distribution(state)

        kl_c = torch.distributions.kl_divergence(tar_c_dist, c_dist).mean()
        kl_d = torch.distributions.kl_divergence(tar_d_dist, d_dist).mean()
        if self.if_sp_action_loss:
            self.update_record(obj_ca=c_obj_surrogate.item(),
                               obj_da=d_obj_surrogate.item(),
                               obj_c=obj_critic.item(),
                               m_kl_c=100 * kl_c.mean().item(),
                               m_kl_d=100 * kl_d.mean().item(),
                               # a_std=self.act.a_std_log.exp().mean().item(),
                               ca_entropy=(-obj_entropy_c.item()),
                               da_entropy=(-obj_entropy_d.item()),
                               a0_avg=buf_action[:, 0].mean().item(),
                               a1_avg=buf_action[:, 1].mean().item(),
                               a0_std=buf_action[:, 0].std().item(),
                               a1_std=buf_action[:, 1].std().item(),
                               )
        else:
            self.update_record(obj_a=obj_surrogate.item(),
                               obj_c=obj_critic.item(),
                               m_kl_c=100 * kl_c.mean().item(),
                               m_kl_d=100 * kl_d.mean().item(),
                               # a_std=self.act.a_std_log.exp().mean().item(),
                               entropy=(-obj_entropy.item()),
                               a0_avg=buf_action[:, 0].mean().item(),
                               a1_avg=buf_action[:, 1].mean().item(),
                               a0_std=buf_action[:, 0].std().item(),
                               a1_std=buf_action[:, 1].std().item(),
                               )

        return self.train_record


class AgentHierarchicalPPO2(AgentPPO2):
    def __init__(self, args=None):
        super().__init__(args)
        # AgentPPO is an on policy DRL algorithm
        self.if_on_policy = True
        # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip)
        if args is None:
            pass
        else:
            self.ratio_clip = args['ratio_clip'] if 'ratio_clip' in args.keys() else 0.3
            # could be 0.01 ~ 0.05
            self.lambda_entropy = args['lambda_entropy'] if 'lambda_entropy' in args.keys() else 0.05
            # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
            self.lambda_gae_adv = args['lambda_gae_adv'] if 'lambda_gae_adv' in args.keys() else 0.97
            # if use Generalized Advantage Estimation
            self.if_use_gae = args['if_use_gae'] if 'if_use_gae' in args.keys() else True
            self.if_use_dn = args['if_use_dn'] if 'if_use_dn' in args.keys() else False
            self.total_iterations = args['total_iterations'] if 'total_iterations' in args.keys() else 1000
            self.loss_coeff_cri = args['loss_coeff_cri'] if 'loss_coeff_cri' in args.keys() else 0.5
            self.objective_type = args['objective_type'] if 'objective_type' in args.keys() else 'clip'

            self.beta = args['beta'] if 'beta' in args.keys() else None
            self.policy_type = args['policy_type'] if 'policy_type' in args.keys() else None
            self.discrete_degree = args['discrete_degree'] if 'discrete_degree' in args.keys() else 3
            self.train_model = args['train_model'] if 'train_model' in args.keys() else "mix"  # mix discrete continues
            self.save_path = args['save_path'] if 'save_path' in args.keys() else None

        self.target_entropy = None
        self.cri_optimizer = None
        self.act_optimizer = None
        self.compute_reward = None  # attribution

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv
        # self.target_entropy = np.log(action_dim)
        # self.target_entropy = -action_dim
        self.cri_d = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        self.cri_c = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        self.act_d = ActorDiscretePPO(net_dim, state_dim, self.discrete_degree ** (action_dim - 1)).to(self.device)
        self.act_c = ActorPPO(net_dim, state_dim, action_dim - 1).to(self.device)
        if self.save_path is not None:
            self.load_model(self.save_path)
        self.act = [self.act_c, self.act_d]
        self.cri_c_optimizer = torch.optim.Adam(params=self.cri_d.parameters(), lr=self.learning_rate)
        self.cri_d_optimizer = torch.optim.Adam(params=self.cri_c.parameters(), lr=self.learning_rate)
        self.act_d_optimizer = torch.optim.Adam(params=self.act_d.parameters(), lr=self.learning_rate)
        self.act_c_optimizer = torch.optim.Adam(params=self.act_c.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()
        self.iter_index = 0
        assert if_per is False  # on-policy don't need PER

    @staticmethod
    def select_action(state, policy, explore_rate=1.) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32).detach_()
        if rd.rand() < explore_rate:  # epsilon-greedy
            d_action = policy[1].get_action(states)
            c_action = policy[0].get_action(states)
            action = torch.cat((c_action, d_action.unsqueeze(dim=1)), 1)
        else:
            d_action = policy[1](states)
            c_action = policy[0](states)
            action = torch.cat((c_action, d_action.unsqueeze(dim=1)), 1)
        if action.isnan().any():
            print(action)
        return action[0].detach().numpy()

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_state = buffer.sample_all()
            buf_action[:, :-1] = buf_action[:, :-1].clamp(-1 + 5e-8, 1 - 5e-8)

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            if self.train_model in ['discrete']:
                buf_value = torch.cat([self.cri_d(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
                buf_r_ret, buf_adv = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
                buf_d_action = buf_action[:, -1].unsqueeze(dim=1)
                buf_d_logprob = self.act_d.compute_logprob(buf_state, buf_d_action).unsqueeze(dim=1)
            elif self.train_model in ['continues']:
                buf_value = torch.cat([self.cri_c(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
                buf_r_ret, buf_adv = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
                buf_c_action = buf_action[:, :-1]
                buf_c_logprob = self.act_c.compute_logprob(buf_state, buf_c_action).unsqueeze(dim=1)
            tar_act_d = deepcopy(self.act_d)
            tar_cri_d = deepcopy(self.cri_d)
            tar_act_c = deepcopy(self.act_c)
            tar_cri_c = deepcopy(self.cri_c)
            del buf_reward, buf_mask

        '''PPO: Surrogate objective of Trust Region'''
        # mix discrete continues
        if self.train_model in ['mix', 'discrete']:
            obj_critic = None
            for _ in range(int(repeat_times * buf_len / batch_size)):
                indices = torch.randint(buf_len - 1, size=(batch_size,), requires_grad=False, device=self.device)
                state = buf_state[indices]
                d_action = buf_d_action[indices]
                logprob = buf_d_logprob[indices]
                r_ret = buf_r_ret[indices]
                adv = buf_adv[indices]

                value = self.cri_d(state)  # critic network predicts the reward_sum (Q value) of state

                obj_critic = self.criterion(value, r_ret)
                self.cri_d_optimizer.zero_grad()
                obj_critic.backward()
                torch.nn.utils.clip_grad_norm_(self.cri_d.parameters(), 4.)
                self.cri_d_optimizer.step()

                new_logprob = self.act_d.compute_logprob(state, d_action).unsqueeze(dim=1)  # it is obj_actor
                ratio = (new_logprob - logprob).clamp(-20, 2).exp()
                obj_surrogate1 = adv * ratio
                if self.objective_type in ['clip']:
                    obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                    obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
                elif self.objective_type == 'kl':
                    mean_kl = torch.distributions.kl_divergence(tar_act_d.get_distribution(state),
                                                                self.act_d.get_distribution(state)).mean()
                    obj_surrogate = -obj_surrogate1.mean() + self.beta * mean_kl

                obj_entropy = (new_logprob.exp() * new_logprob).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
                self.act_d_optimizer.zero_grad()
                obj_actor.backward()
                torch.nn.utils.clip_grad_norm_(self.act_d.parameters(), 4.)
                self.act_d_optimizer.step()
            tar_dist = tar_act_d.get_distribution(state)
            dist = self.act_d.get_distribution(state)
            kl = torch.distributions.kl_divergence(tar_dist, dist).mean()

            self.train_record['obj_da'] = obj_surrogate.item()
            self.train_record['obj_dc'] = obj_critic.item()
            self.train_record['m_kl_d'] = 100 * kl.mean().item()
            self.train_record['da_entropy'] = -obj_entropy.item()
        if self.train_model in ['mix', 'continues']:
            obj_critic = None
            for _ in range(int(repeat_times * buf_len / batch_size)):
                indices = torch.randint(buf_len - 1, size=(batch_size,), requires_grad=False, device=self.device)
                state = buf_state[indices]
                c_action = buf_c_action[indices]
                logprob = buf_c_logprob[indices]
                r_ret = buf_r_ret[indices]
                adv = buf_adv[indices]

                value = self.cri_c(state)  # critic network predicts the reward_sum (Q value) of state

                obj_critic = self.criterion(value, r_ret)
                self.cri_c_optimizer.zero_grad()
                obj_critic.backward()
                torch.nn.utils.clip_grad_norm_(self.cri_c.parameters(), 4.)
                self.cri_c_optimizer.step()

                new_logprob = self.act_c.compute_logprob(state, c_action).unsqueeze(dim=1)  # it is obj_actor
                ratio = (new_logprob - logprob).clamp(-20, 2).exp()
                obj_surrogate1 = adv * ratio
                if self.objective_type in ['clip']:
                    obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                    obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
                elif self.objective_type == 'kl':
                    mean_kl = torch.distributions.kl_divergence(tar_act_c.get_distribution(state),
                                                                self.act_c.get_distribution(state)).mean()
                    obj_surrogate = -obj_surrogate1.mean() + self.beta * mean_kl

                obj_entropy = (new_logprob.exp() * new_logprob).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
                self.act_d_optimizer.zero_grad()
                obj_actor.backward()
                torch.nn.utils.clip_grad_norm_(self.act_d.parameters(), 4.)
                self.act_d_optimizer.step()
            tar_dist = tar_act_c.get_distribution(state)
            dist = self.act_c.get_distribution(state)
            kl = torch.distributions.kl_divergence(tar_dist, dist).mean()

            self.train_record['obj_ca'] = obj_surrogate.item()
            self.train_record['obj_cc'] = obj_critic.item()
            self.train_record['m_kl_c'] = 100 * kl.mean().item()
            self.train_record['ca_entropy'] = -obj_entropy.item()
            self.train_record['a0_avg'] = buf_action[:, 0].item()
            self.train_record['a1_avg'] = buf_action[:, 1].item()
            self.train_record['a0_std'] = buf_action[:, 0].item()
            self.train_record['a1_std'] = buf_action[:, 1].item()

        return self.train_record

    def save_model(self, cwd):
        act_c_save_path = f'{cwd}/actor.pth'
        cri_c_save_path = f'{cwd}/critic.pth'
        act_d_save_path = f'{cwd}/actor_d.pth'
        cri_d_save_path = f'{cwd}/critic_d.pth'
        self.to_cpu()
        if self.act_c is not None:
            torch.save(self.act_c.state_dict(), act_c_save_path)
        if self.cri_c is not None:
            torch.save(self.cri_c.state_dict(), cri_c_save_path)
        if self.act_d is not None:
            torch.save(self.act_d.state_dict(), act_d_save_path)
        if self.cri_d is not None:
            torch.save(self.cri_d.state_dict(), cri_d_save_path)

    def load_model(self, cwd):
        act_c_save_path = f'{cwd}/actor.pth'
        cri_c_save_path = f'{cwd}/critic.pth'
        act_d_save_path = f'{cwd}/actor_d.pth'
        cri_d_save_path = f'{cwd}/critic_d.pth'

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if self.train_model in ['mix', 'continues']:
            if (self.act_d is not None) and os.path.exists(act_d_save_path):
                load_torch_file(self.act_d, act_d_save_path)
                print("Loaded act_d:", cwd)
            if (self.cri_d is not None) and os.path.exists(cri_d_save_path):
                load_torch_file(self.cri_d, cri_d_save_path)
                print("Loaded cri_d:", cwd)
        if self.train_model in ['mix', 'discrete']:
            if (self.act_c is not None) and os.path.exists(act_c_save_path):
                load_torch_file(self.act_c, act_c_save_path)
                print("Loaded act_c:", cwd)
            if (self.cri_c is not None) and os.path.exists(cri_c_save_path):
                load_torch_file(self.cri_c, cri_c_save_path)
                print("Loaded cri_c:", cwd)

        self.to_device()

    def to_cpu(self):
        device = torch.device('cpu')
        if next(self.act_d.parameters()).is_cuda:
            self.act_d.to(device)
        if next(self.cri_d.parameters()).is_cuda:
            self.cri_d.to(device)
        if next(self.act_c.parameters()).is_cuda:
            self.act_c.to(device)
        if next(self.cri_c.parameters()).is_cuda:
            self.cri_c.to(device)

    def to_device(self):
        if not next(self.act_d.parameters()).is_cuda:
            self.act_d.to(self.device)
        if not next(self.cri_d.parameters()).is_cuda:
            self.cri_d.to(self.device)
        if not next(self.act_c.parameters()).is_cuda:
            self.act_c.to(self.device)
        if not next(self.cri_c.parameters()).is_cuda:
            self.cri_c.to(self.device)


class AgentRNNPPO2(AgentPPO2):
    def __init__(self, args=None):
        super().__init__(args)
        # AgentPPO is an on policy DRL algorithm
        self.if_on_policy = True
        # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip)
        if args is None:
            pass
        else:
            self.ratio_clip = args['ratio_clip'] if 'ratio_clip' in args.keys() else 0.3
            # could be 0.01 ~ 0.05
            self.lambda_entropy = args['lambda_entropy'] if 'lambda_entropy' in args.keys() else 0.05
            # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
            self.lambda_gae_adv = args['lambda_gae_adv'] if 'lambda_gae_adv' in args.keys() else 0.97
            # if use Generalized Advantage Estimation
            self.if_use_gae = args['if_use_gae'] if 'if_use_gae' in args.keys() else True
            self.if_use_dn = args['if_use_dn'] if 'if_use_dn' in args.keys() else False
            self.total_iterations = args['total_iterations'] if 'total_iterations' in args.keys() else 1000
            self.loss_coeff_cri = args['loss_coeff_cri'] if 'loss_coeff_cri' in args.keys() else 0.5
            self.objective_type = args['objective_type'] if 'objective_type' in args.keys() else 'clip'
            self.beta = args['beta'] if 'beta' in args.keys() else None
            self.policy_type = args['policy_type'] if 'policy_type' in args.keys() else None
            self.if_store_state = args['if_store_state'] if 'if_store_state' in args.keys() else True
            self.hidden_state_dim = args['hidden_state_dim'] if 'hidden_state_dim' in args.keys() else 128
            self.rnn_timestep = args['rnn_timestep'] if 'rnn_timestep' in args.keys() else 16
            self.infer_by_sequence = args['infer_by_sequence'] if 'infer_by_sequence' in args.keys() else False

        self.target_entropy = None
        self.cri_optimizer = None
        self.act_optimizer = None
        self.compute_reward = None  # attribution

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv
        # self.target_entropy = np.log(action_dim)
        # self.target_entropy = -action_dim
        if self.infer_by_sequence:
            self.cri = CarlaRNNPPOSequence(net_dim, state_dim, action_dim,
                                           hidden_state_dim=self.hidden_state_dim,
                                           if_store_state=self.if_store_state)
        else:
            if self.policy_type in ['mg']:
                self.cri = CarlaRNNPPOMG(net_dim, state_dim, action_dim,
                                         hidden_state_dim=self.hidden_state_dim,
                                         if_store_state=self.if_store_state)
            else:
                self.cri = CarlaRNNPPO(net_dim, state_dim, action_dim,
                                       hidden_state_dim=self.hidden_state_dim,
                                       if_store_state=self.if_store_state)
        self.act = self.cri
        self.optimizer = torch.optim.Adam(params=self.cri.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()
        self.iter_index = 0
        assert if_per is False  # on-policy don't need PER

    @staticmethod
    def select_action(state, policy, hidden_state, cell_state, explore_rate=1., infer_by_sequence=False):
        hidden_state_dim = policy.hidden_state_dim
        if infer_by_sequence:
            states = torch.as_tensor(state, dtype=torch.float32).detach_()
        else:
            states = torch.as_tensor((state,), dtype=torch.float32).detach_()
        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros([1, hidden_state_dim], dtype=torch.float32)
            cell_state = torch.zeros([1, hidden_state_dim], dtype=torch.float32)
        else:
            hidden_state = torch.as_tensor((hidden_state,), dtype=torch.float32).detach_()
            cell_state = torch.as_tensor((cell_state,), dtype=torch.float32).detach_()
        if rd.rand() < explore_rate:  # epsilon-greedy
            action, hidden_state_next, cell_state_next = policy.get_action(states, hidden_state, cell_state)
        else:
            action, hidden_state_next, cell_state_next = policy.actor_forward(states, hidden_state, cell_state)
        return action[0].detach().numpy(), \
               hidden_state_next[0].detach().numpy(), \
               cell_state_next[0].detach().numpy()

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_state, buf_hidden_state, buf_cell_state = buffer.sample_all()
            buf_action = buf_action.clamp(-1 + 5e-8, 1 - 5e-8)

            split_list = []
            idx = 0
            for i in range(buf_len):
                if idx == self.rnn_timestep:
                    split_list.append([i - idx, i])
                    idx = 1
                else:
                    if buf_mask[i] == 0:
                        split_list.append([i - idx, i])  # episode end <16 序列
                        idx = 1
                    else:
                        idx += 1

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat(
                [self.cri.critic_forward(buf_state[i:i + bs], buf_hidden_state[i:i + bs], buf_cell_state[i:i + bs]) for
                 i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = self.cri.compute_logprob_infer(buf_state, buf_action, buf_hidden_state,
                                                         buf_cell_state).unsqueeze(dim=1)
            buf_r_ret, buf_adv = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)

            tar_model = deepcopy(self.cri)
            del buf_reward, buf_mask

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        rnn_buf_len = len(split_list)
        batch_size = rnn_buf_len if rnn_buf_len < batch_size else batch_size
        for _ in range(int(repeat_times * rnn_buf_len / batch_size)):
            indices_list = random.sample(split_list, batch_size)
            indices = torch.cat([torch.arange(split[0], split[1], 1) for split in indices_list], dim=0).to(self.device)
            state = torch.zeros((self.rnn_timestep, batch_size, self.state_dim), dtype=torch.float32,
                                device=self.device)
            len_sequence = [split[1] - split[0] for split in indices_list]
            for i, split in enumerate(indices_list):
                state[0:split[1] - split[0], i, :] = buf_state[split[0]:split[1], :]
            hidden_state = torch.cat([buf_hidden_state[split[0], :].unsqueeze(dim=0) for split in indices_list],
                                     dim=0).to(self.device)
            cell_state = torch.cat([buf_cell_state[split[0], :].unsqueeze(dim=0) for split in indices_list], dim=0).to(
                self.device)
            action = buf_action[indices]
            r_ret = buf_r_ret[indices]
            logprob = buf_logprob[indices]
            adv = buf_adv[indices]

            _, value = self.cri(state, hidden_state,
                                cell_state, len_sequence)  # critic network predicts the reward_sum (Q value) of state

            obj_critic = self.criterion(value, r_ret)

            new_logprob = self.cri.compute_logprob(state, action, hidden_state, cell_state, len_sequence).unsqueeze(
                dim=1)  # it is obj_actor
            ratio = (new_logprob - logprob).clamp(-20, 2).exp()
            obj_surrogate1 = adv * ratio
            if self.objective_type in ['clip', 'tc_clip', 'tc_clip2']:
                obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            elif self.objective_type == 'kl':
                tar_rnn_embading = tar_model.lstm_forward(state, hidden_state, cell_state)
                rnn_embading = self.cri.lstm_forward(state, hidden_state, cell_state)
                mean_kl = torch.distributions.kl_divergence(tar_model.get_distribution(tar_rnn_embading),
                                                            self.cri.get_distribution(rnn_embading)).mean()
                obj_surrogate = -obj_surrogate1.mean() + self.beta * mean_kl
            elif self.objective_type == 'auto_kl':
                self.min_beta = 0.5
                self.max_beta = 5.
                self.kl_target = 0.5
                self.kl_alpha = 1.1
                tar_rnn_embading = tar_model.lstm_forward(state, hidden_state, cell_state)
                rnn_embading = self.cri.lstm_forward(state, hidden_state, cell_state)
                mean_kl = torch.distributions.kl_divergence(tar_model.get_distribution(tar_rnn_embading),
                                                            self.cri.get_distribution(rnn_embading)).mean()
                if mean_kl > self.max_beta * self.kl_target:
                    self.beta = self.kl_alpha * self.beta
                elif mean_kl < self.min_beta * self.kl_target:
                    self.beta = self.beta / self.kl_alpha
                obj_surrogate = -obj_surrogate1.mean() + self.beta * mean_kl

            if self.target_entropy is not None:
                obj_entropy = (new_logprob.exp() * new_logprob - self.target_entropy).clamp_min(0).mean()
            else:
                obj_entropy = (new_logprob.exp() * new_logprob).mean()
            # action_explore = self.act.get_explore(buf_state)
            # action_explore = action_explore[:, 0] + action_explore[:, 1]
            # obj_entropy = (action_explore.log() * action_explore).mean()

            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            if obj_actor.isnan().any() or obj_critic.isnan().any() or obj_actor.isinf().any() or obj_critic.isinf().any():
                print("test")
            loss = obj_actor + obj_critic / (r_ret.std() + 1e-5) * self.loss_coeff_cri
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cri.parameters(), 4.0)
            self.optimizer.step()

        # self.iter_index += 1
        # ep_ratio = 1. - (self.iter_index / self.total_iterations)
        # self.ratio_clip = self.ratio_clip * ep_ratio
        if self.cri.__class__.__name__ in ["CarlaRNNPPO", "CarlaRNNPPOMG", "CarlaRNNPPOSequence"]:
            rnn_input = state[:, :, :self.cri.ego_state]
            tar_rnn_embading = tar_model.lstm_forward(rnn_input, hidden_state, cell_state, len_sequence)
            rnn_embading = self.cri.lstm_forward(rnn_input, hidden_state, cell_state, len_sequence)
        else:
            tar_rnn_embading = tar_model.lstm_forward(state, hidden_state, cell_state, len_sequence)
            rnn_embading = self.cri.lstm_forward(state, hidden_state, cell_state, len_sequence)
        if self.cri.__class__.__name__ in ["RNNPPO"]:
            state = state.permute(1, 0, 2)
            state = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(state)], dim=0)
            kl = torch.distributions.kl_divergence(
                tar_model.get_distribution(torch.cat([state, tar_rnn_embading], dim=1)),
                self.act.get_distribution(torch.cat([state, rnn_embading], dim=1))).mean()
        elif self.cri.__class__.__name__ in ["CarlaRNNPPO", "CarlaRNNPPOMG", "CarlaRNNPPOSequence"]:
            state = state.permute(1, 0, 2)
            state = torch.cat([batch[:len_sequence[i]] for i, batch in enumerate(state)], dim=0)
            kl = torch.distributions.kl_divergence(
                tar_model.get_distribution(torch.cat([state, tar_rnn_embading], dim=1)),
                self.act.get_distribution(torch.cat([state, rnn_embading], dim=1))).mean()

        self.train_record['obj_a'] = obj_surrogate.item()
        self.train_record['obj_c'] = obj_critic.item()
        self.train_record['mean_kl'] = 100 * kl.mean().item()
        self.train_record['entropy'] = -obj_entropy.item()
        self.train_record['a0_avg'] = buf_action[:, 0].mean().item()
        self.train_record['a1_avg'] = buf_action[:, 1].mean().item()
        self.train_record['a0_std'] = buf_action[:, 0].std().item()
        self.train_record['a1_std'] = buf_action[:, 1].std().item()
        if self.policy_type in ['beta2', 'beta']:
            with torch.no_grad():
                action_explore = self.act.get_explore(buf_state)
            self.train_record['a0_exp'] = action_explore[:, 0].mean().item()
            self.train_record['a1_exp'] = action_explore[:, 1].mean().item()
        return self.train_record

    def to_cpu(self):
        device = torch.device('cpu')
        if next(self.cri.parameters()).is_cuda:
            self.cri.to(device)

    def to_device(self):
        if not next(self.cri.parameters()).is_cuda:
            self.cri.to(self.device)


class AgentPPO2CMAES(AgentPPO2):
    def __init__(self, args=None):
        super().__init__(args)
        # AgentPPO is an on policy DRL algorithm
        self.if_on_policy = True
        # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip)
        if args is None:
            pass
        else:
            self.ratio_clip = args['ratio_clip'] if 'ratio_clip' in args.keys() else 0.2
            # could be 0.01 ~ 0.05
            self.lambda_entropy = args['lambda_entropy'] if 'lambda_entropy' in args.keys() else 0.05
            # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
            self.lambda_gae_adv = args['lambda_gae_adv'] if 'lambda_gae_adv' in args.keys() else 0.97
            # if use Generalized Advantage Estimation
            self.K = 10
            self.if_use_gae = args['if_use_gae'] if 'if_use_gae' in args.keys() else True
            self.if_use_dn = args['if_use_dn'] if 'if_use_dn' in args.keys() else False
            self.total_iterations = args['total_iterations'] if 'total_iterations' in args.keys() else 1000
            self.loss_coeff_cri = args['loss_coeff_cri'] if 'loss_coeff_cri' in args.keys() else 0.5
            self.objective_type = args['objective_type'] if 'objective_type' in args.keys() else 'clip'
            self.c_dclip = args['c_dclip'] if 'objective_type' in args.keys() and \
                                              args['objective_type'] == 'double_clip' else 3.

        self.target_entropy = None
        self.cri_optimizer = None
        self.act_optimizer = None
        self.compute_reward = None  # attribution

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        self.cri = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        self.act = ActorPPOMG(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)

        self.cri_optimizer = torch.optim.Adam(params=self.cri.parameters(), lr=self.learning_rate)
        self.act_optimizer = torch.optim.Adam(params=self.act.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()
        self.iter_index = 0
        assert if_per is False  # on-policy don't need PER

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_state = buffer.sample_all()
            buf_action = buf_action.clamp(-1 + 5e-8, 1 - 5e-8)

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = self.act.compute_logprob(buf_state, buf_action).unsqueeze(dim=1)
            buf_r_ret, buf_adv = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
            tar_act = deepcopy(self.act)
            tar_cri = deepcopy(self.cri)
            del buf_reward, buf_mask

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        for i in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len - 1, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_ret = buf_r_ret[indices]
            logprob = buf_logprob[indices]
            adv = buf_adv[indices]

            value = self.cri(state)  # critic network predicts the reward_sum (Q value) of state

            obj_critic = self.criterion(value, r_ret)
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.cri.parameters(), 4.)
            self.cri_optimizer.step()

            if (i % (2 * self.K)) < self.K:  # fixed_loc to update stddev
                loc, cholesky = self.act.get_loc_cholesky(state)
                dist = MultivariateNormal(loc.detach_(), scale_tril=cholesky)
            else:  # fixed_stddev to update loc
                loc, cholesky = self.act.get_loc_cholesky(state)
                dist = MultivariateNormal(loc, scale_tril=cholesky.detach_())
            atanh_action = action.atanh()
            new_logprob = dist.log_prob(atanh_action).unsqueeze(dim=1)  # it is obj_actor
            ratio = (new_logprob - logprob).clamp(-20, 2).exp()
            obj_surrogate1 = adv * ratio
            if self.objective_type in ['clip']:
                obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            elif self.objective_type == 'double_clip':
                obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.max(torch.min(obj_surrogate1, obj_surrogate2), self.c_dclip * adv).mean()
            elif self.objective_type == 'kl':
                mean_kl = torch.distributions.kl_divergence(tar_act.get_distribution(state),
                                                            self.act.get_distribution(state)).mean()
                obj_surrogate = -obj_surrogate1.mean() + self.beta * mean_kl

            if self.target_entropy is not None:
                obj_entropy = (new_logprob.exp() * new_logprob - self.target_entropy).clamp_min(0).mean()
            else:
                obj_entropy = (new_logprob.exp() * new_logprob).mean()

            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            self.act_optimizer.zero_grad()
            obj_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.act.parameters(), 4.)
            self.act_optimizer.step()

        # self.iter_index += 1
        # ep_ratio = 1. - (self.iter_index / self.total_iterations)
        # self.ratio_clip = self.ratio_clip * ep_ratio
        kl = torch.distributions.kl_divergence(tar_act.get_distribution(state), self.act.get_distribution(state)).mean()

        self.update_record(obj_a=obj_surrogate.item(),
                           obj_c=obj_critic.item(),
                           mean_kl=100 * kl.mean().item(),
                           entropy=(-obj_entropy.item()),
                           a0_avg=buf_action[:, 0].mean().item(),
                           a1_avg=buf_action[:, 1].mean().item(),
                           a0_std=buf_action[:, 0].std().item(),
                           a1_std=buf_action[:, 1].std().item(), )

        return self.train_record


class AgentPPOMO2(AgentBase):
    def __init__(self, args=None):
        super().__init__(args)
        # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip),
        self.ratio_clip = 0.3 if args is None else args['ratio_clip']
        # could be 0.01 ~ 0.05
        self.lambda_entropy = 0.05 if args is None else args['lambda_entropy']
        # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.lambda_gae_adv = 0.97 if args is None else args['lambda_gae_adv']
        # if use Generalized Advantage Estimation
        self.if_use_gae = True if args is None else args['if_use_gae']
        # AgentPPO is an on policy DRL algorithm
        self.if_on_policy = True
        self.if_use_dn = False if args is None else args['if_use_dn']
        # Multi Objective:
        if args is not None:
            self.if_auto_weights = args['if_auto_weights'] if 'if_auto_weights' in args.keys() else False
            if 'weights' in args.keys():
                self.weights = args['weights']
            else:
                self.weights = None
            self.pid_Ki = args['pid_Ki'] if 'pid_Ki' in args.keys() else 0.01
            self.pid_Kp = args['pid_Kp'] if 'pid_Kp' in args.keys() else 0.25
            self.pid_Kd = args['pid_Kd'] if 'pid_Kd' in args.keys() else 4

        self.noise = None
        self.optimizer = None
        self.compute_reward = None  # attribution

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        if (self.if_auto_weights) and (reward_dim > 1):
            if self.weights is None:
                self.weights = np.ones(reward_dim) * (1. / reward_dim)
            self.cost_threshold = torch.ones((reward_dim - 1), dtype=torch.float32, device=self.device) * -1e-6
            self.pid_Ki = torch.ones((reward_dim - 1), dtype=torch.float32, device=self.device) * 0.01
            self.pid_Kp = torch.ones((reward_dim - 1), dtype=torch.float32, device=self.device) * 0.25
            self.pid_Kd = torch.ones((reward_dim - 1), dtype=torch.float32, device=self.device) * 4
            self.pid_delta_p_ema_alpha = torch.ones((reward_dim - 1), dtype=torch.float32, device=self.device) * 0.95
            self.pid_delta_d_ema_alpha = torch.ones((reward_dim - 1), dtype=torch.float32, device=self.device) * 0.95
            self.pid_i = torch.ones((reward_dim - 1), dtype=torch.float32, device=self.device)
            self._delta_p = torch.zeros((reward_dim - 1), dtype=torch.float32, device=self.device)
            self._cost_d = torch.zeros((reward_dim - 1), dtype=torch.float32, device=self.device)
            self._cost_d_pre = torch.zeros((reward_dim - 1), dtype=torch.float32, device=self.device)
        else:
            self.weights = np.ones(reward_dim) * (1. / reward_dim)
        self.weights = torch.as_tensor(self.weights, dtype=torch.float32, device=self.device)
        self.cri = CriticAdv_Multi(state_dim, net_dim, reward_dim, self.if_use_dn).to(self.device)
        self.act = ActorPPO(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)

        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])

        self.criterion = torch.nn.SmoothL1Loss()
        assert if_per is False  # on-policy don't need PER

    @staticmethod
    def select_action(state, policy):
        """select action for PPO

       :array state: state.shape==(state_dim, )

       :return array action: state.shape==(action_dim, )
       :return array noise: noise.shape==(action_dim, ), the noise
       """
        states = torch.as_tensor((state,), dtype=torch.float32).detach_()
        action = policy.get_action(states)[0]
        return action.detach().numpy()

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = self.act.compute_logprob(buf_state, buf_action).unsqueeze(dim=1)
            buf_r_ret, buf_adv = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
            if self.if_auto_weights:
                self.auto_weights(buf_r_ret)
            del buf_reward, buf_mask

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        for _ in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_ret = buf_r_ret[indices]
            logprob = buf_logprob[indices]
            adv = buf_adv[indices]

            new_logprob = self.act.compute_logprob(state, action).unsqueeze(dim=1)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = adv * ratio
            obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean(dim=0)
            obj_surrogate = (obj_surrogate * self.weights).sum()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            values = self.cri(state)  # critic network predicts the reward_sum (Q value) of state
            obj_critics = [self.criterion(values[:, i], r_ret[:, i]) for i in range(values.shape[1])]

            obj_critic = 0
            for i in range(len(obj_critics)):
                obj_critic += obj_critics[i] / (r_ret[:, i].std() + 1e-5)
            obj_united = obj_actor + obj_critic
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

        self.update_record(obj_a=obj_surrogate.item(),
                           obj_c=sum([obj_critics[i].item() for i in range(len(obj_critics))]),
                           obj_tot=obj_united.item(),
                           a_std=self.act.a_std_log.exp().mean().item(),
                           entropy=(-obj_entropy.item()),
                           weight1=self.weights[1])
        return self.train_record

    def compute_reward_adv(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_ret = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_ret = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                                device=self.device)  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]
        buf_adv = buf_r_ret - (buf_mask * buf_value)
        buf_adv = (buf_adv - buf_adv.mean(dim=0)) / (buf_adv.std(dim=0) + 1e-5)
        return buf_r_ret, buf_adv

    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_ret = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_ret = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                                device=self.device)  # reward sum of previous step
        pre_adv = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                              device=self.device)  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]

            buf_adv[i] = buf_reward[i] + buf_mask[i] * pre_adv - buf_value[i]
            pre_adv = buf_value[i] + buf_adv[i] * self.lambda_gae_adv

        buf_adv = (buf_adv - buf_adv.mean(dim=0)) / (buf_adv.std(dim=0) + 1e-5)
        return buf_r_ret, buf_adv

    def auto_weights(self, buf_ret):
        nagetive_r = buf_ret.mean(dim=0)[1:]
        # PID update here: CRL :cost instead nagetive_r
        cost_mean = -nagetive_r
        delta = cost_mean - self.cost_threshold  # ep_cost_avg: tensor
        zeros = torch.zeros(cost_mean.shape, dtype=torch.float32, device=self.device)
        self.pid_i = torch.max(zeros, self.pid_i + delta * self.pid_Ki)
        a_p = self.pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self.pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(cost_mean)
        pid_d = torch.max(zeros, self._cost_d - self._cost_d_pre)
        pid_o = (self.pid_Kp * self._delta_p + self.pid_i +
                 self.pid_Kd * pid_d)
        self.weights[1:] = torch.max(zeros, pid_o)
        self._cost_d_pre = self._cost_d


# class AgentMPO(AgentBase):
#     def __init__(self, args=None):
#         super().__init__(args)
#         self.epsilon_dual = 0.1 if args is None else args['epsilon_dual']
#         self.epsilon_kl_mu = 0.01 if args is None else args['epsilon_kl_mu']
#         self.epsilon_kl_sigma = 0.01 if args is None else args['epsilon_kl_sigma']
#         self.epsilon_kl = 0.01 if args is None else args['epsilon_kl']
#         self.alpha = 10. if args is None else args['alpha']
#         self._num_samples = 64 if args is None else args['num_samples']
#
#     def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         self.act = ActorMPO(net_dim, state_dim, action_dim).to(self.device)
#         self.act_target = deepcopy(self.act)
#         self.cri = CriticTwin(net_dim, state_dim, action_dim).to(self.device)
#         self.cri_target = deepcopy(self.cri)
#
#         self.criterion = torch.nn.SmoothL1Loss()
#         self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
#         self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
#
#         self.eta = np.random.rand()
#         self.eta_kl_mu = 0.
#         self.eta_kl_sigma = 0.
#         self.eta_kl_mu = 0.
#         self.get_obj_critic = self.get_obj_critic_raw
#
#     @staticmethod
#     def select_action(state, policy):
#         """select action for PPO
#
#        :array state: state.shape==(state_dim, )
#
#        :return array action: state.shape==(action_dim, )
#        :return array noise: noise.shape==(action_dim, ), the noise
#        """
#         states = torch.as_tensor((state,), dtype=torch.float32).detach_()
#         action = policy.get_action(states)[0]
#         return action.detach().numpy()
#
#     def update_net(self, buffer, _target_step, batch_size, repeat_times) -> (float, float):
#         buffer.update_now_len_before_sample()
#         buf_len = buffer.now_len
#         obj_critic = None
#         for _ in range(int(repeat_times * buf_len / batch_size)):
#             # Policy Evaluation
#             obj_critic, state, q_label = self.get_obj_critic(buffer, batch_size)
#             self.cri_optimizer.zero_grad()
#             obj_critic.backward()
#             self.cri_optimizer.step()
#             self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
#
#             # Policy Improvation
#             # Sample N additional action for each state
#             online_loc, online_cholesky = self.act.get_loc_cholesky(state)
#             with torch.no_grad():
#                 target_loc, target_cholesky = self.act_target.get_loc_cholesky(state)  # (B,)
#                 target_pi = MultivariateNormal(target_loc, scale_tril=target_cholesky)  # (B,)
#                 sampled_a = target_pi.sample((self._num_samples,))  # (N, B, dim-a)
#                 expanded_s = state[None, ...].expand(self._num_samples, -1, -1)  # (N, B, dim-s)
#                 target_q = self.cri_target.forward(
#                     expanded_s.reshape(-1, state.shape[1]),  # (N * B, dim-s)
#                     sampled_a.reshape(-1, self.action_dim)  # (N * B, dim-a)
#                 ).reshape(self._num_samples, batch_size)
#                 target_q_np = target_q.cpu().numpy()
#
#             # E step
#             def dual(eta):
#                 max_q = np.max(target_q_np, 0)
#                 return eta * self.epsilon_dual + np.mean(max_q) \
#                        + eta * np.mean(np.log(np.mean(np.exp((target_q_np - max_q) / eta), axis=0)))
#
#             bounds = [(1e-6, None)]
#             res = minimize(dual, np.array([self.eta]), method='SLSQP', bounds=bounds)
#             self.eta = res.x[0]
#
#             normalized_weights = torch.softmax(target_q / self.eta, dim=0)  # (N, B) or (da, B)
#
#             # M step
#
#             fixed_stddev_dist = MultivariateNormal(loc=online_loc, scale_tril=target_cholesky)
#             fixed_loc_dist = MultivariateNormal(loc=target_loc, scale_tril=online_cholesky)
#             loss_p = torch.mean(
#                 normalized_weights * (
#                         fixed_stddev_dist.expand((self._num_samples, batch_size)).log_prob(sampled_a)  # (N, B)
#                         + fixed_loc_dist.expand((self._num_samples, batch_size)).log_prob(sampled_a)  # (N, B)
#                 )
#             )
#             kl_mu, kl_sigma = gaussian_kl(mu_i=target_loc, mu=online_loc, A_i=target_cholesky, A=online_cholesky)
#
#             self.eta_kl_mu -= self.alpha * (self.epsilon_kl_mu - kl_mu).detach().item()
#             self.eta_kl_sigma -= self.alpha * (self.epsilon_kl_sigma - kl_sigma).detach().item()
#             self.eta_kl_mu = 0.0 if self.eta_kl_mu < 0.0 else self.eta_kl_mu
#             self.eta_kl_sigma = 0.0 if self.eta_kl_sigma < 0.0 else self.eta_kl_sigma
#             self.act_optimizer.zero_grad()
#             obj_actor = -(
#                     loss_p
#                     + self.eta_kl_mu * (self.epsilon_kl_mu - kl_mu)
#                     + self.eta_kl_sigma * (self.epsilon_kl_sigma - kl_sigma)
#             )
#             self.act_optimizer.zero_grad()
#             obj_actor.backward()
#             self.act_optimizer.step()
#             self.soft_update(self.act_target, self.act, self.soft_update_tau)
#
#         self.update_record(obj_a=obj_actor.item(),
#                            obj_c=obj_critic.item(),
#                            loss_pi=loss_p.item(),
#                            est_q=q_label.mean().item(),
#                            max_kl_mu=kl_mu.item(),
#                            max_kl_sigma=kl_sigma.item(),
#                            eta=self.eta)
#         return self.train_record
#
#     def get_obj_critic_raw(self, buffer, batch_size):
#         with torch.no_grad():
#             reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
#             pi_next_s = self.act_target.get_distribution(next_s)
#             sampled_next_a = pi_next_s.sample((self._num_samples,)).transpose(0, 1)  # (B, N, dim-action)
#             ex_next_s = next_s[:, None, :].expand(-1, self._num_samples, -1)  # (B, N, dim-action)
#             next_q = (
#                 self.cri_target(
#                     ex_next_s.reshape(-1, self.state_dim),
#                     sampled_next_a.reshape(-1, self.action_dim)
#                 ).reshape(batch_size, self._num_samples)
#             ).mean(dim=1).unsqueeze(dim=1)
#             q_label = reward + mask * next_q
#         q = self.cri(state, action)
#         obj_critic = self.criterion(q, q_label)
#         return obj_critic, state, q_label


'''Utils'''


def rescale(x):
    epsilon = 0.01
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) + 1) + epsilon * x


def ni_rescale(x):
    epsilon = 0.01
    return torch.sign(x) * ((torch.sqrt(1 + 4 * epsilon * (torch.abs(x) + 1 + epsilon) - 1) / (2 * epsilon)) ^ 2 - 1)


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def gaussian_kl(mu_i, mu, A_i, A):
    """
    decoupled KL between two multivariate gaussian distribution
    C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))
    :param μi: (B, dim-a)
    :param μ: (B, dim-a)
    :param Ai: (B, dim-a, dim-a)
    :param A: (B, dim-a, dim-a)
    :return: C_μ, C_Σ: mean and covariance terms of the KL
    """
    n = A.size(-1)
    mu_i = mu_i.unsqueeze(-1)  # (B, n, 1)
    mu = mu.unsqueeze(-1)  # (B, n, 1)
    sigma_i = A_i @ bt(A_i)  # (B, n, n)
    sigma = A @ bt(A)  # (B, n, n)
    sigma_i_inv = sigma_i.inverse()  # (B, n, n)
    sigma_inv = sigma.inverse()  # (B, n, n)
    inner_mu = ((mu - mu_i).transpose(-2, -1) @ sigma_i_inv @ (mu - mu_i)).squeeze()  # (B,)
    inner_sigma = torch.log(sigma_inv.det() / sigma_i_inv.det()) - n + btr(sigma_i_inv @ sigma_inv)  # (B,)
    C_mu = 0.5 * torch.mean(inner_mu)
    C_sigma = 0.5 * torch.mean(inner_sigma)
    return C_mu, C_sigma


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """The noise of Ornstein-Uhlenbeck Process

        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.

        :int size: the size of noise, noise.shape==(-1, action_dim)
        :float theta: related to the not independent of OU-noise
        :float sigma: related to action noise std
        :float ou_noise: initialize OU-noise
        :float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """output a OU-noise
        :return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise
