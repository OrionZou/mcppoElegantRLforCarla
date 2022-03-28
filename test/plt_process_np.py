import os
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':

    # Label = ['PID']
    # names = ['pid']
    Label = ['MCPPO']
    names = ['cppo']
    # Label = ['PID', 'PPO2', 'MCPPO']
    # names = ['pid', 'ppo','cppo']
    # Label = ['PID', 'PPO2', 'D3QN', 'SAC','MCPPO']
    # names = ['pid', 'ppo', 'd3qn', 'sac','cppo']
    # Label = ['PID', 'PPO2', 'SAC', 'MCPPO']
    # names = ['pid', 'ppo', 'sac', 'cppo']

    # Label = ['PID', 'DPPO2', 'PPO2', 'HPPO2-3', 'HPPO2-2', 'MIXPPO2']
    # names = ['pid', 'discreteppo', 'ppo', 'hybridppo-3', 'hybridppo-2', 'sadppo']

    # Label = ['PID', 'DPPO2', 'PPO2', 'D3QN','MIXPPO2']
    # names = ['pid', 'discreteppo', 'ppo', 'd3qn', 'sadppo']

    # Label = [ 'PPO2', 'PPO2-MN', 'PPO2-MN-CMAES', 'PPO2-BETA', 'PPO2-D', 'PPO2-MD']
    # names = ['ppo', 'mgppo', 'ppocmaes', 'betappo', 'discreteppo', 'sadppo']

    # Label = ['PID', 'PPO2', 'PPO2-MG', 'PPO2-MG-CMAES', 'PPO2-BETA', 'MIXPPO2']
    # names = ['pid', 'ppo', 'mgppo', 'ppocmaes', 'betappo', 'sadppo']

    # Label = ['PID', 'PPO2', 'RNNPPO2-SS', 'RNNPPO2-ZS']
    # names = ['pid', 'ppo', 'rnnppo-storestate', 'rnnppo-zerostate']

    # evel_path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/eval'
    # evel_path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/eval_town07-v2'
    evel_path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/eval_town03-v1'
    metrics = [
        'position',
    ]
    # metric = 'delta_a_lat'
    dt = 0.05
    # dt = 0.1
    # dt = 1
    fig_dims = (5., 5.5)
    font={'weight':'bold','size':20}
    for metric in metrics:
        scale = 1. / dt
        # fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=fig_dims)

        for i, name in enumerate(names):
            position = np.load(evel_path + f'/{name}/ep-{metric}.npy')
            y = position[:, 0]
            x = position[:, 1]
            plt.plot(x, y,lw=4, label=f'{Label[i]}')
        # plt.legend()
        plt.xticks([])
        plt.yticks([])
        # plt.title("Town07_maintainRoad_V1",fontdict=font)
        # plt.title("Town07_maintainRoad_V2",fontdict=font)
        plt.title("Town03_urbanRoad_V1",fontdict=font)

        plt.show()
