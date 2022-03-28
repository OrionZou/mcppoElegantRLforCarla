import os
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set(style="darkgrid")
if __name__ == '__main__':
    # Label = ['PID']
    # names = ['pid']
    # Label = ['PID', 'PPO2', 'D3QN', 'SAC','MCPPO']
    # names = ['pid', 'ppo', 'd3qn', 'sac','cppo']
    # Label = ['PID', 'PPO2', 'SAC', 'MCPPO']
    # names = ['pid', 'ppo', 'sac', 'cppo']
    # Label = ['PID', 'PPO2', 'PPO2-MD', 'MCPPO']
    # names = ['pid', 'ppo', 'sadppo','cppo']

    Label = ['PID', 'PPO2', 'MCPPO']
    names = ['pid', 'ppo', 'cppo']

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

    evel_path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/eval'
    metrics = [
        # 'position',
        # 'steer',
        'lon_action',
        'a0',
        # 'a1',
        # 'lat_distance',
        # 'velocity_lat',
        # 'delta_a_lat',
        # 'delta_jerk_lat',
        # 'delta_steer',
        # 'delta_yaw',
        # 'reward',
        # 'yaw_angle',
        # 'velocity_lon',
        # 'delta_a_lon',
        # 'delta_jerk_lon',

    ]
    metrics_y_label = [
        # 'position',
        # '$steer$',
        '$lon_{a}$',
        '$a_{lo}$',
        # '$a_{la}$',
        # '$lat\_distance$',
        # '$V_{la}$',
        # '$Acc_{la}$',
        # '$Jerk_{la}$',
        # 'delta_steer',
        # 'delta_yaw',
        # 'reward',
        # 'yaw_angle',
        # '$V_{lo}$',
        # '$Acc_{lo}$',
        # '$Jerk_{lo}$',

    ]
    # metric = 'delta_a_lat'
    # dt=0.05
    # dt = 0.1
    dt = 1
    fig_dims = (6., 3.)
    font = {'weight': 'bold', 'size': 17}
    for j,metric in enumerate(metrics):
        scale = 1. / dt

        fig, ax = plt.subplots(figsize=fig_dims)
        df = pd.DataFrame()
        for i, name in enumerate(names):
            if metric in ['position']:
                position = np.load(evel_path + f'/{name}/ep-{metric}.npy')
                x = position[:,0]
                y = position[:, 1]
            else:
                y = np.load(evel_path + f'/{name}/ep-{metric}.npy')
                x = np.arange(0, y.shape[0], 1)
            tmp_df = pd.DataFrame(np.vstack((x, y)).T)
            tmp_df.columns = ['TimeStep', f'{metric}']
            tmp_df['TimeStep'] = (scale * (tmp_df['TimeStep'] // scale))
            tmp_df['AlgoType'] = Label[i]
            df = pd.concat([df, tmp_df], axis=0)

        df = df.reset_index()
        sns.lineplot(x="TimeStep", y=f"{metric}", hue='AlgoType', style='AlgoType',
                     ci=95, data=df, ax=ax, linewidth=3)
        # sns.relplot(x="TimeStep", y=f"{metric}", hue='AlgoType', style='AlgoType', kind="line",
        #             ci=95, data=df,ax=ax)
        # ax.set(xlabel=None)
        # if not (j==3 or j==6):
        #     ax.set_xticks([])
        ax.get_legend().remove()
        plt.ylabel(f"{metrics_y_label[j]}", fontdict=font)
        plt.xlabel(f"$TimeStep$", fontdict=font)
        ax.tick_params(axis='y', labelsize=17)
        ax.tick_params(axis='x', labelsize=17)
        # plt.xticks(fontsize=20)

        plt.show()
        fig.savefig(f'/home/zgy/fig/{metric}.svg', format='svg', dpi=5000)
