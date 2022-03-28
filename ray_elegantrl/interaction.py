import ray
import torch
import os
import time
import numpy as np
import numpy.random as rd
from collections import deque
import datetime
from copy import deepcopy
from ray_elegantrl.buffer import ReplayBuffer, ReplayBufferMP
from ray_elegantrl.evaluate import RecordEpisode, RecordEvaluate, Evaluator
from ray_elegantrl.config import default_config

"""
Modify [ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)
by https://github.com/GyChou
"""


class Arguments:
    def __init__(self, configs=default_config):
        self.gpu_id = configs['gpu_id']  # choose the GPU for running. gpu_id is None means set it automatically
        # current work directory. cwd is None means set it automatically
        self.cwd = configs['cwd'] if 'cwd' in configs.keys() else None
        # current work directory with time.
        self.if_cwd_time = configs['if_cwd_time'] if 'cwd' in configs.keys() else False
        # initialize random seed in self.init_before_training()

        self.random_seed = 0
        # id state_dim action_dim reward_dim target_return horizon_step
        self.env = configs['env']
        # Deep Reinforcement Learning algorithm
        self.agent = configs['agent']
        self.agent['agent_name'] = self.agent['class_name']().__class__.__name__
        self.trainer = configs['trainer']
        self.interactor = configs['interactor']
        self.buffer = configs['buffer']
        self.evaluator = configs['evaluator']
        self.config = deepcopy(configs)

        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)

        '''if_per_explore'''
        if self.buffer['if_on_policy']:
            self.if_per_explore = False
        else:
            self.if_per_explore = configs['interactor']['random_explore'] if 'random_explore' in configs[
                'interactor'].keys() else False
        self.buffer['if_rnn'] = self.agent['if_rnn']
        if self.agent['if_rnn']:
            self.buffer['hidden_state_dim'] = self.agent['hidden_state_dim']
        self.buffer['action_type'] = self.env['action_type']
        self.buffer['state_dim'] = self.env['state_dim']
        self.buffer['action_dim'] = self.env['action_dim']
        self.buffer['reward_dim'] = self.env['reward_dim']
        self.buffer['rollout_num'] = self.interactor['rollout_num']

    def init_before_training(self, if_main=True):
        '''set gpu_id automatically'''
        if self.gpu_id is None:  # set gpu_id automatically
            import sys
            self.gpu_id = sys.argv[-1][-4]
        else:
            self.gpu_id = str(self.gpu_id)
        if not self.gpu_id.isdigit():  # set gpu_id as '0' in default
            self.gpu_id = '0'

        '''set cwd automatically'''
        if self.cwd is None:
            if self.if_cwd_time:
                curr_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            else:
                curr_time = 'current'
            if 'carla' in self.env["id"]:
                a = ''
                if isinstance(self.env["action_dim"], list):
                    for e in self.env["action_dim"]:
                        a += str(e) + '_'
                else:
                    a = str(self.env["action_dim"]) + '_'
                self.cwd = f'./veh_control_logs/{self.env["id"]}' \
                           f'_{self.env["params_name"]["params"]["town"]}' \
                           f'_{self.env["params_name"]["params"]["task_mode"]}' \
                           f'_s{self.env["state_dim"]}_a{a}r{self.env["reward_dim"]}' \
                           f'_tr{self.env["target_return"]}_ms{self.env["max_step"]}' \
                           f'_{self.env["params_name"]["params"]["if_dest_end"]}/' \
                           f'{self.agent["agent_name"]}_{self.agent["policy_type"]}_{self.agent["objective_type"]}/' \
                           f'exp_{curr_time}_cuda:{self.gpu_id}'
            else:
                self.cwd = f'./veh_control_logs/{self.env["id"]}_s{self.env["state_dim"]}_' \
                           f'a{self.env["action_dim"]}_r{self.env["reward_dim"]}' \
                           f'_tr{self.env["target_return"]}_ms{self.env["max_step"]}/' \
                           f'{self.agent["agent_name"]}_{self.agent["policy_type"]}_{self.agent["objective_type"]}/' \
                           f'exp_{curr_time}_cuda:{self.gpu_id}'

        if if_main:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)

            '''save exp parameters'''
            from ruamel.yaml.main import YAML
            yaml = YAML()
            del self.config['agent']['class_name']
            del self.config['if_cwd_time']
            self.config['cwd'] = self.cwd
            with open(self.cwd + '/parameters.yaml', 'w', encoding="utf-8") as f:
                yaml.dump(self.config, f)
            del self.config

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


def make_env(env_dict, id=None, seed=0):
    import gym
    import gym.envs
    import gym_carla_feature
    import safety_env
    import safety_gym
    import sumo_env
    if 'params_name' in env_dict:
        if env_dict['params_name'] == 'params':
            env_dict['params_name']['params']['port'] = env_dict['params_name']['params']['port'] + id * 4
            env_dict['params_name']['params']['label'] = id
        env = gym.make(env_dict['id'], **env_dict['params_name'])
    else:
        env = gym.make(env_dict['id'])
    env.seed(seed=(id + seed))
    return env


@ray.remote
class InterActor(object):

    def __init__(self, id, args):
        self.id = id
        args.init_before_training(if_main=False)
        self.env = make_env(args.env, self.id, seed=args.random_seed)
        self.env_max_step = args.env['max_step']
        self.env_horizon = args.interactor[
            'env_horizon'] if 'env_horizon' in args.interactor.keys() else self.env_max_step
        self.reward_scale = np.array(args.interactor['reward_scale'])
        self._horizon_step = args.interactor['horizon_step'] // args.interactor['rollout_num']
        self.gamma = np.array(args.interactor['gamma']).reshape(-1) if type(
            args.interactor['gamma']) is np.ndarray else np.ones(
            args.env['reward_dim']) * args.interactor['gamma']
        self.action_dim = args.env['action_dim']
        # choose -1 discrete action space | 1 continuous action space | 0 hybird action space |
        self.action_type = args.env['action_type']
        self.agent_config = args.agent
        if self.agent_config['if_rnn']:
            self.hidden_state_dim = args.buffer['hidden_state_dim']
        if args.agent['agent_name'] in [  # 'AgentPPO',
            # 'AgentPPO2',
            'AgentPPO2CMA',
            'AgentPPO2RS'
            'AgentMPO',
            'AgentPPOMO',
            'AgentPPOMO2',
        ] and (args.agent['policy_type'] not in ['beta', 'beta2']):
            self.modify_action = lambda x: np.tanh(x)
        elif args.agent['agent_name'] in ['AgentHybridPPO']:
            def modify_action(action):
                action[:-1] = np.tanh(action[:-1])
                return action

            self.modify_action = modify_action
        elif args.agent['agent_name'] in ['AgentHybridPPO2', 'AgentHierarchicalPPO2']:
            if args.agent['discrete_degree'] == 3:
                def modify_action(action):
                    def mapping(da_dim, x):
                        if da_dim == 0:
                            return np.clip(x, -1, 0)
                        elif da_dim == 2:
                            return np.clip(x, 0, 1)
                        else:
                            return 0.

                    da_idx = int(action[-1])
                    mod_a = np.zeros(action[:-1].shape)
                    mod_a[0] = mapping(da_idx // 3, action[0])
                    mod_a[1] = mapping(da_idx % 3, action[1])
                    return mod_a
                    # return action[:-1]
            elif args.agent['discrete_degree'] == 2:
                def modify_action(action):
                    def mapping(da_dim, x):
                        if da_dim == 1:
                            return x
                        else:
                            return 0.

                    da_idx = int(action[-1])
                    mod_a = np.zeros(action[:-1].shape)
                    mod_a[0] = mapping(da_idx // 2, action[0])
                    mod_a[1] = mapping(da_idx % 2, action[1])
                    return mod_a
            self.modify_action = modify_action
        elif args.agent['agent_name'] in ['AgentSafePPO2']:
            def modify_action(origin_action):
                safe_action = origin_action[-1]
                action = np.tanh(origin_action[:-1])
                if abs(action[1]) > safe_action:
                    action[1] = action[1] * safe_action
                return action

            self.modify_action = modify_action
        else:
            self.modify_action = lambda x: x

        # if args.agent['agent_name'] in ['HybridSAC']:
        #     self.exploit_policy = self.exploit_policys
        # elif args.agent['agent_name'] in ['AgentSafePPO2']:
        #     self.exploit_policy = self.safe_exploit_policy
        # elif args.agent['agent_name'] in ['AgentRNNPPO2']:
        #     self.exploit_policy = self.exploit_rnn_policy
        # else:
        #     self.exploit_policy = self.exploit_one_policy
        if self.agent_config['if_rnn']:
            self.buffer = ReplayBuffer(
                max_len=args.buffer['max_buf'] // args.interactor['rollout_num'] + args.env['max_step'],
                if_on_policy=args.buffer['if_on_policy'],
                state_dim=args.env['state_dim'],
                action_dim=1 if args.env['action_type'] == -1 else (
                    args.env['action_dim'][0] + 1 if args.env['action_type'] == 0 else args.env['action_dim']),
                reward_dim=args.env['reward_dim'],
                if_per=args.buffer['if_per'],
                if_discrete_action=(args.buffer[
                                        'action_type'] == -1) if 'action_type' in args.buffer.keys() else False,
                if_rnn=self.agent_config['if_rnn'],
                hidden_state_dim=args.buffer['hidden_state_dim'],
                if_gpu=False)
        else:
            self.buffer = ReplayBuffer(
                max_len=args.buffer['max_buf'] // args.interactor['rollout_num'] + args.env['max_step'],
                if_on_policy=args.buffer['if_on_policy'],
                state_dim=args.env['state_dim'],
                action_dim=1 if args.env['action_type'] == -1 else (
                    args.env['action_dim'][0] + 1 if args.env['action_type'] == 0 else args.env['action_dim']),
                reward_dim=args.env['reward_dim'],
                if_per=args.buffer['if_per'],
                if_discrete_action=(args.buffer[
                                        'action_type'] == -1) if 'action_type' in args.buffer.keys() else False,
                if_gpu=False)

        self.record_episode = RecordEpisode(args.env)

    @ray.method(num_returns=1)
    def explore_env(self, select_action, policy):
        self.buffer.empty_buffer_before_explore()
        actual_step = 0
        terminal = True
        while actual_step < self._horizon_step:
            state_list = []
            action_list = []
            reward_list = []
            gamma_list = []
            if terminal:
                state = self.env.reset()
                terminal = False
            if self.agent_config['if_rnn']:
                hidden_state = None
                cell_state = None
                hidden_state_list = []
                cell_state_list = []
                if self.agent_config['infer_by_sequence']:
                    sq_state = state.reshape(1, -1)
            for i in range(self.env_horizon):
                if self.agent_config['if_rnn']:
                    if self.agent_config['infer_by_sequence']:
                        idx = len(hidden_state_list) if len(hidden_state_list) < self.agent_config['rnn_timestep'] else \
                            self.agent_config['rnn_timestep']
                        hidden_state_input = hidden_state_list[-idx] if len(hidden_state_list) > 0 else None
                        cell_state_input = cell_state_list[-idx] if len(cell_state_list) > 0 else None
                        action, hidden_state, cell_state = select_action(sq_state,
                                                                         policy,
                                                                         hidden_state_input,
                                                                         cell_state_input,
                                                                         explore_rate=self.agent_config[
                                                                             'explore_rate'] if 'explore_rate' in self.agent_config.keys() else 1.,
                                                                         infer_by_sequence=self.agent_config[
                                                                             'infer_by_sequence'])
                    else:
                        action, hidden_state, cell_state = select_action(state,
                                                                         policy,
                                                                         hidden_state,
                                                                         cell_state,
                                                                         explore_rate=1.)
                else:
                    action = select_action(state,
                                           policy,
                                           explore_rate=self.agent_config[
                                               'explore_rate'] if 'explore_rate' in self.agent_config.keys() else 1., )
                next_s, reward, terminal, _ = self.env.step(self.modify_action(action))
                done = True if i == (self.env_horizon - 1) else terminal
                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward * self.reward_scale)
                gamma_list.append(np.zeros(self.gamma.shape) if done else self.gamma)
                if self.agent_config['if_rnn']:
                    hidden_state_list.append(hidden_state)
                    cell_state_list.append(cell_state)
                actual_step += 1

                if self.agent_config['if_rnn'] and self.agent_config['infer_by_sequence']:
                    idx = max(sq_state.shape[0] - self.agent_config['rnn_timestep'] + 1, 0)
                    sq_state = np.vstack((sq_state[idx, :], next_s))
                state = next_s
                if done:
                    if self.agent_config['if_rnn']:
                        self.buffer.extend_buffer(np.array(state_list),
                                                  np.array(action_list),
                                                  np.array(reward_list),
                                                  np.array(gamma_list),
                                                  np.array(hidden_state_list),
                                                  np.array(cell_state_list))
                    else:
                        self.buffer.extend_buffer(np.array(state_list),
                                                  np.array(action_list),
                                                  np.array(reward_list),
                                                  np.array(gamma_list))
                    break

        self.buffer.update_now_len_before_sample()
        if self.agent_config['if_rnn']:
            return actual_step, \
                   self.buffer.buf_state[:self.buffer.now_len], \
                   self.buffer.buf_action[:self.buffer.now_len], \
                   self.buffer.buf_reward[:self.buffer.now_len], \
                   self.buffer.buf_gamma[:self.buffer.now_len], \
                   self.buffer.buf_hidden_state[:self.buffer.now_len], \
                   self.buffer.buf_cell_state[:self.buffer.now_len]
        else:
            return actual_step, \
                   self.buffer.buf_state[:self.buffer.now_len], \
                   self.buffer.buf_action[:self.buffer.now_len], \
                   self.buffer.buf_reward[:self.buffer.now_len], \
                   self.buffer.buf_gamma[:self.buffer.now_len]

    @ray.method(num_returns=1)
    def random_explore_env(self, r_horizon_step=None):
        self.buffer.empty_buffer_before_explore()
        if r_horizon_step is None:
            r_horizon_step = self._horizon_step
        else:
            r_horizon_step = max(min(r_horizon_step, self.buffer.max_len - 1), self._horizon_step)
        actual_step = 0
        while actual_step < r_horizon_step:
            state_list = []
            action_list = []
            reward_list = []
            gamma_list = []
            if self.agent_config['if_rnn']:
                hidden_state_list = []
                cell_state_list = []
            state = self.env.reset()
            for _ in range(self.env_max_step):
                # action = rd.randint(self.action_dim) if self.action_type==-1 else rd.uniform(-1, 1,
                if self.action_type == 0:
                    action = np.hstack((rd.uniform(-1, 1, self.action_dim[0]), rd.randint(self.action_dim[1])))
                else:
                    action = self.env.action_space.sample()
                next_s, reward, done, _ = self.env.step(action)
                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward * self.reward_scale)
                gamma_list.append(np.zeros(self.gamma.shape) if done else self.gamma)
                if self.agent_config['if_rnn']:
                    hidden_state = np.zeros((1, self.hidden_state_dim), dtype=np.float32)
                    cell_state = np.zeros((1, self.hidden_state_dim), dtype=np.float32)
                    hidden_state_list.append(hidden_state)
                    cell_state_list.append(cell_state)
                actual_step += 1
                if done:
                    if self.agent_config['if_rnn']:
                        self.buffer.extend_buffer(np.array(state_list),
                                                  np.array(action_list),
                                                  np.array(reward_list),
                                                  np.array(gamma_list),
                                                  np.array(hidden_state_list),
                                                  np.array(cell_state_list))
                    else:
                        self.buffer.extend_buffer(np.array(state_list),
                                                  np.array(action_list),
                                                  np.array(reward_list),
                                                  np.array(gamma_list))
                    break
                state = next_s
        self.buffer.update_now_len_before_sample()
        if self.agent_config['if_rnn']:
            return actual_step, \
                   self.buffer.buf_state[:self.buffer.now_len], \
                   self.buffer.buf_action[:self.buffer.now_len], \
                   self.buffer.buf_reward[:self.buffer.now_len], \
                   self.buffer.buf_gamma[:self.buffer.now_len], \
                   self.buffer.buf_hidden_state[:self.buffer.now_len], \
                   self.buffer.buf_cell_state[:self.buffer.now_len]
        else:
            return actual_step, \
                   self.buffer.buf_state[:self.buffer.now_len], \
                   self.buffer.buf_action[:self.buffer.now_len], \
                   self.buffer.buf_reward[:self.buffer.now_len], \
                   self.buffer.buf_gamma[:self.buffer.now_len]

    def exploit_env(self, select_action, policy, eval_times):
        self.record_episode.clear()
        eval_record = RecordEvaluate()

        for _ in range(eval_times):
            state = self.env.reset()
            if self.agent_config['if_rnn']:
                hidden_state = None
                cell_state = None
                hidden_state_list = []
                cell_state_list = []
                if self.agent_config['infer_by_sequence']:
                    sq_state = state.reshape(1, -1)
            for _ in range(self.env_max_step):
                if self.agent_config['if_rnn']:
                    if self.agent_config['infer_by_sequence']:
                        idx = len(hidden_state_list) if len(hidden_state_list) < self.agent_config['rnn_timestep'] else \
                            self.agent_config['rnn_timestep']
                        hidden_state_input = hidden_state_list[-idx] if len(hidden_state_list) > 0 else None
                        cell_state_input = cell_state_list[-idx] if len(cell_state_list) > 0 else None
                        action, hidden_state, cell_state = select_action(sq_state,
                                                                         policy,
                                                                         hidden_state_input,
                                                                         cell_state_input,
                                                                         explore_rate=0.,
                                                                         infer_by_sequence=self.agent_config[
                                                                             'infer_by_sequence'])
                    else:
                        action, hidden_state, cell_state = select_action(state,
                                                                         policy,
                                                                         hidden_state,
                                                                         cell_state,
                                                                         explore_rate=0.)
                else:
                    action = select_action(state,
                                           policy,
                                           explore_rate=0.)
                next_s, reward, done, info = self.env.step(self.modify_action(action))
                self.record_episode.add_record(reward, info)
                if self.agent_config['if_rnn']:
                    hidden_state_list.append(hidden_state)
                    cell_state_list.append(cell_state)
                if done:
                    break
                if self.agent_config['if_rnn'] and self.agent_config['infer_by_sequence']:
                    idx = max(sq_state.shape[0] - self.agent_config['rnn_timestep'] + 1, 0)
                    sq_state = np.vstack((sq_state[idx, :], next_s))
                    sq_state = np.vstack((sq_state[idx, :], next_s))
                state = next_s
            cost_threshold = self.agent_config[
                'cost_threshold'] if 'cost_threshold' in self.agent_config.keys() else None
            eval_record.add(self.record_episode.get_result(cost_threshold))
            self.record_episode.clear()
        return eval_record.results

    # @staticmethod
    # def exploit_policys(state, policy):
    #     state = torch.as_tensor((state,), dtype=torch.float32).detach_()
    #     action_c = policy[0](state)
    #     action_d_int = policy[1].get_a_prob(torch.cat((state, action_c), dim=1)).argmax(dim=1, keepdim=True)
    #     return torch.cat((action_c, action_d_int), dim=1)[0].detach().numpy()
    #
    # @staticmethod
    # def safe_exploit_policy(state, policy):
    #     safe_actions = [0., 0.2, 0.4, 0.6, 0.8, 1.]
    #     state = torch.as_tensor((state,), dtype=torch.float32).detach_()
    #     action_tensor = policy[0](state)
    #     action = action_tensor[0].detach().numpy()
    #     # a_prob = policy[1].get_a_prob(torch.cat([state, action_tensor], dim=1))[0].detach().numpy()
    #     # # steer_clip = rd.choice(a_prob.shape[0], p=a_prob)
    #     # steer_clip = safe_actions[a_prob.argmax(axis=0)]
    #     # if abs(action[1]) > steer_clip:
    #     #     action[1] = action[1] * steer_clip
    #     return action
    #
    # @staticmethod
    # def exploit_one_policy(state, policy):
    #     if isinstance(policy, list):
    #         return policy[0](torch.as_tensor((state,), dtype=torch.float32).detach_())[0].detach().numpy()
    #     else:
    #         return policy(torch.as_tensor((state,), dtype=torch.float32).detach_())[0].detach().numpy()
    #
    # @staticmethod
    # def exploit_rnn_policy(state, policy, hidden_state, cell_state, hidden_state_dim):
    #     state = torch.as_tensor((state,), dtype=torch.float32).detach_()
    #     if hidden_state is None or cell_state is None:
    #         hidden_state = torch.zeros([1, hidden_state_dim], dtype=torch.float32)
    #         cell_state = torch.zeros([1, hidden_state_dim], dtype=torch.float32)
    #     else:
    #         hidden_state = torch.as_tensor((hidden_state,), dtype=torch.float32).detach_()
    #         cell_state = torch.as_tensor((cell_state,), dtype=torch.float32).detach_()
    #     action, hidden_state_next, cell_state_next = policy.actor_forward(state, hidden_state, cell_state)
    #     return action[0].detach().numpy(), \
    #            hidden_state_next[0].detach().numpy(), \
    #            cell_state_next[0].detach().numpy()


class Trainer(object):

    def __init__(self, args_trainer, agent, buffer):
        self.agent = agent
        self.buffer = buffer
        self.sample_step = args_trainer['sample_step']
        self.batch_size = args_trainer['batch_size']
        self.policy_reuse = args_trainer['policy_reuse']

    def train(self):
        self.agent.to_device()
        train_record = self.agent.update_net(self.buffer, self.sample_step, self.batch_size, self.policy_reuse)
        if self.buffer.if_on_policy:
            self.buffer.empty_buffer_before_explore()
        return train_record


def beginer(config, params=None):
    args = Arguments(config)
    args.init_before_training()
    args_id = ray.put(args)
    #######Init######
    agent = args.agent['class_name'](args=args.agent)
    agent.init(args.agent['net_dim'],
               args.env['state_dim'],
               args.env['action_dim'],
               args.env['reward_dim'],
               args.buffer['if_per'])
    interactors = []
    for i in range(args.interactor['rollout_num']):
        time.sleep(1)
        interactors.append(InterActor.remote(i, args_id))
    if args.buffer['if_rnn']:
        buffer_mp = ReplayBufferMP(
            max_len=args.buffer['max_buf'] + args.env['max_step'] * args.interactor['rollout_num'],
            state_dim=args.env['state_dim'],
            action_dim=1 if args.env['action_type'] == -1 else (
                args.env['action_dim'][0] + 1 if args.env['action_type'] == 0 else args.env['action_dim']),
            reward_dim=args.env['reward_dim'],
            if_on_policy=args.buffer['if_on_policy'],
            if_per=args.buffer['if_per'],
            if_discrete_action=(args.buffer[
                                    'action_type'] == -1) if 'action_type' in args.buffer.keys() else False,
            rollout_num=args.interactor['rollout_num'],
            if_rnn=args.buffer['if_rnn'],
            hidden_state_dim=args.buffer['hidden_state_dim'],
            if_gpu=args.buffer['if_gpu']
        )
    else:
        buffer_mp = ReplayBufferMP(
            max_len=args.buffer['max_buf'] + args.env['max_step'] * args.interactor['rollout_num'],
            state_dim=args.env['state_dim'],
            action_dim=1 if args.env['action_type'] == -1 else (
                args.env['action_dim'][0] + 1 if args.env['action_type'] == 0 else args.env['action_dim']),
            reward_dim=args.env['reward_dim'],
            if_on_policy=args.buffer['if_on_policy'],
            if_per=args.buffer['if_per'],
            if_discrete_action=(args.buffer[
                                    'action_type'] == -1) if 'action_type' in args.buffer.keys() else False,
            rollout_num=args.interactor['rollout_num'],
            if_gpu=args.buffer['if_gpu']
        )
    trainer = Trainer(args.trainer, agent, buffer_mp)
    evaluator = Evaluator(args)
    rollout_num = args.interactor['rollout_num']

    #######Random Explore Before Interacting#######
    if args.if_per_explore:
        episodes_ids = [interactors[i].random_explore_env.remote() for i in range(rollout_num)]
        assert len(episodes_ids) > 0
        for i in range(len(episodes_ids)):
            done_id, episodes_ids = ray.wait(episodes_ids)
            if args.buffer['if_rnn']:
                actual_step, buf_state, buf_action, buf_reward, buf_gamma, buf_hidden_state, buf_cell_state = ray.get(
                    done_id[0])
                buffer_mp.extend_buffer(buf_state, buf_action, buf_reward, buf_gamma, i, buf_hidden_state,
                                        buf_cell_state)
            else:
                actual_step, buf_state, buf_action, buf_reward, buf_gamma = ray.get(done_id[0])
                buffer_mp.extend_buffer(buf_state, buf_action, buf_reward, buf_gamma, i)

    #######Interacting Begining#######
    start_time = time.time()
    eval_step = 0
    while (evaluator.record_totalstep < evaluator.break_step) or (evaluator.record_satisfy_return):
        agent.to_cpu()
        policy_id = ray.put(agent.policy)
        #######Explore Environment#######
        episodes_ids = [interactors[i].explore_env.remote(agent.select_action, policy_id) for i in
                        range(rollout_num)]
        sample_step = 0
        for i in range(len(episodes_ids)):
            done_id, episodes_ids = ray.wait(episodes_ids)
            if args.buffer['if_rnn']:
                actual_step, buf_state, buf_action, buf_reward, buf_gamma, buf_hidden_state, buf_cell_state = ray.get(
                    done_id[0])
                sample_step += actual_step
                buffer_mp.extend_buffer(buf_state, buf_action, buf_reward, buf_gamma, i, buf_hidden_state,
                                        buf_cell_state)
            else:
                actual_step, buf_state, buf_action, buf_reward, buf_gamma = ray.get(done_id[0])
                sample_step += actual_step
                buffer_mp.extend_buffer(buf_state, buf_action, buf_reward, buf_gamma, i)
        evaluator.update_totalstep(sample_step)
        #######Training#######
        train_record = trainer.train()
        eval_step += sample_step
        #######Evaluate#######
        if evaluator.eval_gap < eval_step:
            evaluator.tb_train(train_record)

            eval_step = 0
            agent.to_cpu()
            policy_id = ray.put(agent.policy)
            evalRecorder = RecordEvaluate()

            if_eval = True
            #######pre-eval#######
            if evaluator.pre_eval_times > 0:
                eval_results = ray.get(
                    [interactors[i].exploit_env.remote(agent.select_action,
                                                       policy_id,
                                                       eval_times=evaluator.pre_eval_times)
                     for i in range(rollout_num)])
                for eval_result in eval_results:
                    evalRecorder.add_many(eval_result)
                eval_record = evalRecorder.eval_result()
                if eval_record['return'][0]['max'] < evaluator.target_return:
                    if_eval = False
                    evaluator.tb_eval(eval_record)
            #######eval#######
            if if_eval:
                eval_results = ray.get(
                    [interactors[i].exploit_env.remote(agent.select_action,
                                                       policy_id,
                                                       eval_times=evaluator.eval_times)
                     for i in range(rollout_num)])
                for eval_result in eval_results:
                    evalRecorder.add_many(eval_result)
                eval_record = evalRecorder.eval_result()
                evaluator.tb_eval(eval_record)
            #######Save Model#######
            evaluator.analyze_result(eval_record)
            evaluator.iter_print(train_record, eval_record, use_time=(time.time() - start_time))
            evaluator.save_model(agent)
            start_time = time.time()

    print(f'#######Experiment Finished!\t TotalTime:{evaluator.total_time:8.0f}s #######')
