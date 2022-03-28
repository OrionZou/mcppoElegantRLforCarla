import os
import torch
from torch import nn


class IReward():
    def __init__(self, state_dim, action_dim=None, mid_dim=128, type=None):
        self.start_idx = 0
        self.end_idx = 0
        self.mid_dim = 128
        self.num_epoch = 2
        self.batch_size = 256
        self.learning_rate = 1e-4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.if_rnd=True
        self.if_ngu=False
        if self.if_rnd:
            self.rnd_target = RNDTarget(self.mid_dim, state_dim).to(self.device)
            self.rnd = RND(self.mid_dim, state_dim).to(self.device)
            self.rnd_optimizer = torch.optim.Adam(params=self.rnd.parameters(), lr=self.learning_rate)
        if self.if_ngu:
            self.siameseNet = SiameseNet(self.mid_dim, state_dim, action_dim)
            self.siamese_optimizer = torch.optim.Adam(params=self.siameseNet.parameters(), lr=self.learning_rate)
        # self.criterion = torch.nn.SmoothL1Loss()
        self.criterion = torch.nn.MSELoss()

    def deal_episodes_sequence(self, buf_state, buf_action, buf_reward, buf_gamma):
        for idx in reversed(range(self.start_idx, self.start_idx + buf_gamma.shape[0])):
            pass
            buf_reward[idx] = buf_reward[idx]
        return buf_reward

    def calc_rnd(self, buf_state, buf_reward, buf_gamma):
        buf_err = torch.zeros((buf_state.size(0),1))
        for i in range(0, buf_state.size(0) - 1):
            if buf_gamma[i + 1] > 0:
                buf_err[i,:] = self.criterion(self.rnd(buf_state[i + 1]), self.rnd_target(buf_state[i + 1]))
        buf_err=buf_err.to(device=buf_reward.get_device())
        return buf_reward + buf_err, buf_err

    def calc_ngu(self, buf_state, buf_reward, buf_gamma):
        # Compute the episodic intrinsic reward
        buf_r_episode = torch.zeros((buf_state.size(0),1))
        length = 0
        for i in range(0, buf_state.size(0) - 1):
            length += 1
            if buf_gamma[i] == 0:
                buf_r_episode[i - length:i,:] = self.siameseNet.infer_state_episode(buf_state[i - length:i])
                length = 1
        # Compute alpha
        buf_err = torch.zeros((buf_state.size(0),1))
        for i in range(0, buf_state.size(0) - 1):
            if buf_gamma[i + 1] > 0:
                buf_err[i,:] = self.criterion(self.rnd(buf_state[i + 1]), self.rnd_target(buf_state[i + 1]))
        buf_alpha = 1 + (buf_err - buf_err.mean()) / buf_err.std()
        buf_ir = buf_r_episode * buf_alpha.clamp(1, 5)
        buf_ir = buf_ir.to(device=buf_reward.get_device())
        return buf_reward + buf_ir, buf_ir

    def update_rnd(self, buf_state, buf_len):
        obj_rnd = None
        for _ in range(int(self.num_epoch * buf_len / self.batch_size)):
            indices = torch.randint(buf_len - 1, size=(self.batch_size,), requires_grad=False, device=self.device)
            state = buf_state[indices]
            obj_rnd = self.criterion(self.rnd(state), self.rnd_target(state))
            self.rnd_optimizer.zero_grad()
            obj_rnd.backward()
            self.rnd_optimizer.step()
        return obj_rnd


    def save_load_model(self, cwd, if_save=False):
        rnd_module_save_path = f"{cwd}/rnd.pth"
        rnd_target_module_save_path = f"{cwd}/rnd_target.pth"

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.rnd is not None:
                torch.save(self.rnd.state_dict(), rnd_module_save_path)
            if self.rnd_target is not None:
                torch.save(self.rnd_target.state_dict(), rnd_target_module_save_path)
        elif (self.rnd is not None) and os.path.exists(rnd_module_save_path):
            load_torch_file(self.rnd, rnd_module_save_path)
            print("Loaded rnd:", cwd)
        elif (self.rnd_target is not None) and os.path.exists(rnd_target_module_save_path):
            load_torch_file(self.rnd_target, rnd_target_module_save_path)
            print("Loaded rnd_target:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))


class RND(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, mid_dim, state_dim, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, output_dim))

    def forward(self, state):
        return self.net(state)  # g

class RNDTarget(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, mid_dim, state_dim, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, output_dim))
        layer_norm(self.net[0])
        layer_norm(self.net[2])

    def forward(self, state):
        return self.net(state)  # g

class SiameseNet(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        output_dim = 32
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, output_dim), nn.ReLU(), )
        self.action_predictor = nn.Sequential(nn.Linear(state_dim * 2, mid_dim), nn.ReLU(),
                                              nn.Linear(mid_dim, action_dim))
        self.epsilon = 1e-3
        self.c = 1e-3
        self.topk = 32
        self.distance = torch.nn.functional.pairwise_distance

    def forward(self, state, next_state):
        state_output = self.net(state)
        next_state_output = self.net(next_state)
        tmp_input = torch.cat((state_output, next_state_output), dim=1)
        return self.action_predictor(tmp_input)

    def calc_similarity(self, prev_state_episode):  # [batch, state_dim]
        x = torch.unsqueeze(prev_state_episode[-1, :], dim=0)
        y = prev_state_episode[:-1, :]
        distance_x_all = self.distance(self.net(x), self.net(y), p=2)
        distance_x_topk = torch.topk(distance_x_all, self.topk)
        kernel_value = 0
        for y in distance_x_topk:
            normalize_distance_xy = max(y - 1e-6, 0) / distance_x_topk.mean()
            kernel_value += self.epsilon / ((normalize_distance_xy) + self.epsilon)
        r_episode = 1. / (torch.sqrt(kernel_value) + self.c)
        return r_episode

    def infer_state_episode(self, state_episode):
        buf_r_episode = torch.zeros((state_episode.shape[0],))
        for idx in range(state_episode.shape[0] - 1):
            if idx < self.topk + 10:
                continue
            else:
                buf_r_episode[idx] = self.calc_similarity(state_episode[:idx + 1])
        return buf_r_episode


def layer_norm(layer):
    torch.nn.init.uniform_(layer.weight, 0, 0.1)
    torch.nn.init.uniform_(layer.bias, 0, 0.5)
