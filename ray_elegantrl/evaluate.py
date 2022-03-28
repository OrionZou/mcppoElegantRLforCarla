import os
import shutil
import numpy as np
import torch
from tensorboardX import SummaryWriter
from copy import deepcopy

"""
By https://github.com/GyChou
"""


class TensorBoard:
    _writer = None

    @classmethod
    def get_writer(cls, load_path=None):
        if cls._writer:
            return cls._writer
        cls._writer = SummaryWriter(load_path)
        return cls._writer


class RecordEpisode:
    def __init__(self, args_env):
        self.env_mark = 1 if 'carla' in args_env['id'] else 0
        self.env_mark = 2 if 'Safety' in args_env['id'] else self.env_mark
        self.env_mark = 3 if 'sumo' in args_env['id'] else self.env_mark
        self.l_reward = []
        self.record = {}

    def add_record(self, reward, info=None):
        self.l_reward.append(reward)
        if (info is not None) and (self.env_mark in {1, 2, 3}):
            for k, v in info.items():
                if k not in self.record.keys():
                    self.record[k] = []
                self.record[k].append(v)

    def get_result(self, cost_threshold=None):  # cost_threshold -> list
        results = {}
        #######Reward#######
        results['reward'] = {}
        results['return'] = {}
        rewards = np.array(self.l_reward)
        if len(rewards.shape) > 1:
            # Multi Objective RL and Constrained RL
            # results['reward'][0] = rewards.mean()
            # results['return'][0] = rewards.sum()
            cost_threshold = [0] * rewards.shape[1] if cost_threshold is None else cost_threshold
            # cost_threshold = [0.07, 0.03]
            sum_reward = []
            sum_reward.append(rewards[:, 0])
            for i in range(1, rewards.shape[1]):
                sc = (rewards[:, i] + cost_threshold[i - 1]) < 0
                reward_for_cost = sc * (rewards[:, i] + cost_threshold[i - 1])
                sum_reward.append(reward_for_cost)
            sum_reward = np.array(sum_reward).sum(axis=0)
            results['reward'][0] = sum_reward.mean()
            results['return'][0] = sum_reward.sum()
            for i in range(rewards.shape[1]):
                results['reward'][i + 1] = rewards[:, i].mean()
                results['return'][i + 1] = rewards[:, i].sum()
        else:
            results['reward'][0] = rewards.mean()
            results['return'][0] = rewards.sum()

        #######Total#######
        results['total'] = {}
        results['total']['step'] = rewards.shape[0]
        if self.env_mark == 1:  ##For Carla
            results['total']['delta_steer'] = np.abs(np.array(self.record['delta_steer'])).mean()
            results['total']['velocity'] = np.array(self.record['velocity']).mean()
            # results['total']['lane_skewness'] = np.array(self.record['lane_skewness']).mean()
            results['total']['outroute'] = np.array(self.record['outroute']).sum()
            results['total']['outlane'] = np.array(self.record['outlane']).sum()
            results['total']['collision'] = np.array(self.record['collision']).sum()
            results['total']['finish'] = np.array(self.record['finish']).sum()
            results['total']['standingStep'] = np.array(self.record['standingStep']).sum()
            results['total']['distance'] = np.array(self.record['distance'])[-1]
            results['total']['lat_distance'] = np.abs(np.array(self.record['lat_distance'])).mean()
            results['total']['yaw_angle'] = np.abs(np.array(self.record['yaw_angle'])).mean()
            results['total']['delta_yaw'] = np.abs(np.array(self.record['delta_yaw'])).mean()
            results['total']['jerk_lat'] = np.abs(np.array(self.record['jerk_lat'])).mean()
            results['total']['jerk_lon'] = np.abs(np.array(self.record['jerk_lon'])).mean()
            results['total']['acc_lat'] = np.abs(np.array(self.record['acc_lat'])).mean()
            results['total']['acc_lon'] = np.abs(np.array(self.record['acc_lon'])).mean()
        elif self.env_mark == 2:  ##For Safety-gym
            for k, v in self.record.items():
                results['total'][k] = np.array(v).sum()
        elif self.env_mark == 3:  ##For sumo
            results['total']['velocity'] = np.array(self.record['velocity']).mean()
            results['total']['distance'] = np.array(self.record['distance'])[-1]
            results['total']['broken_traffic'] = np.array(self.record['broken_traffic']).sum()
        else:
            if len(self.record) > 0:
                for k, v in self.record.items(): \
                        results['total'][k] = np.array(v).sum()
        return results

    def clear(self):
        self.l_reward = []
        self.record = {}


def calc(np_array):
    if len(np_array.shape) > 1:
        np_array = np_array.sum(axis=1)
    return {'avg': np_array.mean(),
            'std': np_array.std(),
            'max': np_array.max(),
            'min': np_array.min(),
            'mid': np.median(np_array)}


class RecordEvaluate:

    def __init__(self):
        self.results = {}

    def add(self, result):
        if len(self.results) == 0:
            self.results = result
            for k in result.keys():
                for i, v in result[k].items():
                    self.results[k][i] = [v]
        else:
            for k in result.keys():
                if k not in self.results.keys():
                    self.results[k] = {}
                for i, v in result[k].items():
                    self.results[k][i].append(v)

    def add_many(self, results):
        if len(self.results) == 0:
            self.results = deepcopy(results)
        else:
            for k in results.keys():
                for i, v in results[k].items():
                    self.results[k][i] += results[k][i]

    def eval_result(self):
        result = {}
        for k in self.results.keys():
            result[k] = {}
            for i, v in self.results[k].items():
                result[k][i] = calc(np.array(self.results[k][i]))
        return result

    def clear(self):
        self.results = {}


class Evaluator():
    def __init__(self, args):
        self.cwd = args.cwd
        self.writer = TensorBoard.get_writer(args.cwd)
        self.target_return = args.env['target_return']
        self.eval_times = args.evaluator['eval_times']
        self.eval_gap = args.evaluator['eval_gap']
        self.break_step = args.evaluator['break_step']
        self.satisfy_return_stop = args.evaluator['satisfy_return_stop']
        self.pre_eval_times = args.evaluator['pre_eval_times']
        # self.crl_thresholds = args.agent['cost_threshold'] if 'cost_threshold' in args.agent.keys() else None todo for CRL eval
        self.device = torch.device('cpu')

        self.record_totalstep = 0
        self.curr_step = 0
        self.record_satisfy_return = False
        self.curr_max_avg_return = -1e10

        self.model_dim = args.env['reward_dim'] + 1 if args.env['reward_dim'] > 1 else args.env['reward_dim']
        self.save_model_path_list = [None] * self.model_dim
        self.total_time = 0

        self.max_lan_modelpool = 5
        self.modelspools = [{} for _ in range(self.model_dim)]  # key: return|float; value: title|model_path str

    def update_totalstep(self, totalstep):
        self.curr_step = totalstep
        self.record_totalstep += totalstep

    def _update_modelpool(self, avg_returns):  # avg_return []*reward_dim
        for i, avg_return in enumerate(avg_returns):
            if len(self.modelspools[i]) < self.max_lan_modelpool or avg_return > min(self.modelspools[i].keys()):
                self.modelspools[i][avg_return] = str(self.record_totalstep).zfill(10) + f"_{round(avg_return, 4)}"
                model_path = self._get_model_path(self.modelspools[i][avg_return], r_idx=i)
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                self.save_model_path_list[i] = model_path
            while len(self.modelspools[i]) > self.max_lan_modelpool:
                key = min(self.modelspools[i].keys())
                title = self.modelspools[i][key]
                model_path = self._get_model_path(title, r_idx=i)
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                self.modelspools[i].pop(key)

    def _get_model_path(self, title=None, r_idx=0):
        cwd = self.cwd + f"/model{str(r_idx).zfill(2)}"
        if not os.path.exists(cwd):
            os.mkdir(cwd)
        return cwd + f"/{title}"

    def analyze_result(self, result):
        avg_return = [result['return'][i]['avg'] for i in range(len(result['return']))]

        self._update_modelpool(avg_return)
        if avg_return[0] > self.curr_max_avg_return:
            self.curr_max_avg_return = avg_return[0]
        if len(self.modelspools[0]) > 0 and \
                ((min(self.modelspools[0].keys()) > self.target_return) and (self.satisfy_return_stop)):
            self.record_satisfy_return = True

    def tb_train(self, train_record):
        for key, value in train_record.items():
            self.writer.add_scalar(f'algorithm/{key}', value, self.record_totalstep - self.curr_step)

    def tb_eval(self, eval_record):
        for k in eval_record.keys():
            for i in eval_record[k].keys():
                for key, value in eval_record[k][i].items():
                    self.writer.add_scalar(f'{k}_{i}/{key}', value, self.record_totalstep - self.curr_step)

    def iter_print(self, train_record, eval_record, use_time):
        print_info = f"|{'Step':>8}  {'MaxR':>8}|" + \
                     f"{'avgR':>8}  {'stdR':>8}" + \
                     f"{'avgS':>6}  {'stdS':>4} |"
        for key in train_record.keys():
            print_info += f"{key:>8}"
        print_info += " |"
        print(print_info)
        print_info = f"|{self.record_totalstep:8.2e}  {self.curr_max_avg_return:8.2f}|" + \
                     f"{eval_record['return'][0]['avg']:8.2f}  {eval_record['return'][0]['std']:8.2f}" + \
                     f"{eval_record['total']['step']['avg']:6.0f}  {eval_record['total']['step']['std']:4.0f} |"
        for key in train_record.keys():
            print_info += f"{train_record[key]:8.2f}"
        print_info += " |"
        print(print_info)
        self.total_time += use_time
        print_info = f"| UsedTime:{use_time:8.3f}s  TotalTime:{self.total_time:8.0f}s"
        if any(path is not None for path in self.save_model_path_list):
            print_info += f" |{[bool(path) for path in self.save_model_path_list]}  Save model!"
        print(print_info)

    def save_model(self, agent):
        for i, path in enumerate(self.save_model_path_list):
            if path is not None:
                agent.save_model(path)
                self.save_model_path_list[i] = None

            # if isinstance(agent.act, list):
            #     for i in range(len(agent.act)):
            #         cri_save_path = f'{self.cwd}/actor{i}.pth'
            #         torch.save(agent.act[i].state_dict(), cri_save_path)
            # else:
            #     act_save_path = f'{self.cwd}/actor.pth'
            #     torch.save(agent.act.state_dict(), act_save_path)
            # if isinstance(agent.cri, list):
            #     for i in range(len(agent.cri)):
            #         cri_save_path = f'{self.cwd}/critic{i}.pth'
            #         torch.save(agent.cris[i].state_dict(), cri_save_path)
            # else:
            #     cri_save_path = f'{self.cwd}/critic.pth'
            #     torch.save(agent.cri.state_dict(), cri_save_path)
            # if hasattr(agent, "irward_module"):
            #     rnd_save_path = f'{self.cwd}/rnd.pth'
            #     torch.save(agent.irward_module.rnd.state_dict(), rnd_save_path)
            #     rnd_tar_save_path = f'{self.cwd}/rnd_tar.pth'
            #     torch.save(agent.irward_module.rnd_target.state_dict(), rnd_tar_save_path)
