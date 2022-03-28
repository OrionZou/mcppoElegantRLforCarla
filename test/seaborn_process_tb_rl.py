import os
from tensorboard.backend.event_processing import event_accumulator
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# print(matplotlib.get_configdir())
# print(os.environ)
def search_event(path):
    # 用于获取path目录下所有的event文件的数据,需要注意的有以下几点: (ps:通过pytorch_lightning调用的tensorboard)
    # 1. 没有通过'/'分层的全在path目录下的那个events文件中,通过'/'分层的数据,即使是通过tb.add_scalars()API添加的,也是一个scalar一个文件夹
    # 2. 通过'/'分层的数据,单个scalar独享一个文件夹,且只有该文件夹下最大的event文件中的数据是正常的；另外,该event文件中scalar的名字是(例:'树干/树支')干上的名字:
    # 例如:train_loss_dict = {"loss": loss, "ae_loss": loss_ae, "recon_loss": loss_recon, "liear_loss": loss_linear}
    #     tensorboard.add_scalars("train/loss", train_loss_dict, self.global_step)
    # 则 event 文件中的scalar的名字为:'train/loss',所有上面四个文件夹中的event文件里的scalar的名字都是'train/loss'
    dir_name = []
    file_name = []
    file_size = []
    event_num = 0  # 用于计数当前path目录下(不包括子目录)的event文件个数
    for filename in os.listdir(path):
        # print(filename)
        temp_path = os.path.join(path, filename)
        if os.path.isdir(temp_path):
            # 如果是文件夹的话，除了根目录，就是通过"/"add的scalar的文件夹了
            # 如果是文件夹，则递归地取出文件夹下有效的events文件
            temp_file_name = search_event(temp_path)
            # 将该文件夹下有效的events文件路径添加到list中
            file_name.extend(temp_file_name)
        elif os.path.isfile(temp_path):
            # 如果文件名中包含有'tfevents'字符串，则认为其是一个events文件
            if 'tfevents' in temp_path:
                event_num += 1
                # 记录该目录下的events文件数量、路径、文件尺寸
                file_name.append(temp_path)
                file_size.append(os.path.getsize(temp_path))
    if event_num > 1:
        # 如果当前目录下的event文件个数>1,则取size最大的那个
        index = file_size.index(max(file_size))
        temp_file_path = file_name[index]
        if isinstance(temp_file_path, str):
            temp_file_path = [temp_file_path]
        return temp_file_path
    return file_name


def readEvent(event_path):
    '''返回tensorboard生成的event文件中所有的scalar的值和名字
            event_path:event文件路径
    '''
    event = event_accumulator.EventAccumulator(event_path)
    event.Reload()
    print("\033[1;34m数据标签：\033[0m")
    print(event.Tags())
    print("\033[1;34m标量数据关键词：\033[0m")
    # print(event.scalars.Keys())
    scalar_name = []
    scalar_data = []
    for name in event.scalars.Keys():
        # print(name)
        if 'hp_metric' not in name:
            scalar_name.append(name)
            # event.scalars.Items(name)返回的是list,每个元素为ScalarEvent,有wall_time,step(即我们add_scalar时的step)，value（该scalar在step时的值）
            scalar_data.append(event.scalars.Items(name))
    return scalar_name, scalar_data


def exportToexcel(file_name, excelName):
    '''
        将不同的标量数据导入到同一个excel中，放置在不同的sheet下
            注：excel中sheet名称的命名不能有：/\?*这些符号
    '''
    writer = pd.ExcelWriter(excelName)
    for i in range(len(file_name)):
        event_path = file_name[i]
        scalar_name, scalar_data = readEvent(event_path)
        for i in np.arange(len(scalar_name)):
            scalarValue = scalar_data[i]
            scalarName = scalar_name[i]
            if scalarName in ['Train/MaxActionExplor', 'Train/MeanActionExplor', 'Train/MeanActionStd']:
                continue
            if "/" in scalarName:
                temp_names = scalar_name[i].split("/")
                temp_paths = os.path.split(event_path)
                scalarName = os.path.split(temp_paths[0])[1]
            data = pd.DataFrame(scalarValue)
            data.to_excel(writer, sheet_name=scalarName)
    writer.save()
    print("数据保存成功")


def excel_to_array(excel_path, save_dir=None):
    '''
    保存excel所有表格中的数据到.mat
    Args:
        excel_path: excel表格路径
        save_dir: 保存.mat文件路径
    Returns: 容纳所有表格数据的字典
    '''
    # 将None传递给read_excel函数，返回的是一个key为表名的字典
    f = pd.read_excel(excel_path, sheet_name=None)
    data_dict = dict()
    for key in f.keys():
        sheet = f[key]
        sheet = sheet.head(n=-1)
        value = sheet.values
        # 只保留最后两列,一列是step,另一列是值
        data_dict[key] = value[:, 2:4]
        # print(value)
    if save_dir is not None:
        save_dict_as_mat(data_dict, save_dir)
    return data_dict


def save_dict_as_mat(dict, save_dir):
    # scio.savemat(save_dir, dict)
    pass


def smooth(data, weight=0.85):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

    save = pd.DataFrame({'Step': data['Step'].values, 'Value': smoothed})
    save.to_csv('smooth_' + csv_path)


if __name__ == '__main__':
    # Label = ['PPO2', 'SAC']
    # eve_paths = [
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_None_clip/exp_2021-12-03-21-56-07_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentModSAC_None_None_RS5/exp_2021-12-06-20-53-45_cuda:1',
    # ]
    # time = [17194, 53129]  # s

    # Label = ['PPO2','SAC-RS32', 'SAC-RS16']
    # eve_paths = [
    #     # '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentD3QN_None_None/a21-0.6-exp_2021-11-10-13-03-45_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_None_clip/exp_2021-11-10-13-02-45_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentModSAC_None_None/exp_2021-11-17-00-14-23_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r1_tr700_ms200_False/AgentModSAC_None_None/exp_2021-11-23-13-09-54_cuda:1'
    # ]
    #
    Label = ['D3QN',
             'SAC',
             'PPO',
             'MCPPO',
             ]
    # town07-v1
    # eve_paths = [
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/discrete/AgentD3QN_None_None/exp_2021-12-04-01-01-20_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentModSAC_None_None_RS5/exp_2021-12-06-20-53-45_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_None_clip/exp_2021-12-03-21-56-07_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/constriant/AgentConstriantPPO2_None_clip/r17/exp_2021-12-12-13-26-49_cuda:1',
    # ]
    # town07-v2
    # eve_paths = [
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentD3QN_None_None/exp_2021-12-09-23-37-27_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentModSAC_None_None/exp_2021-12-10-11-12-56_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentPPO2_None_clip/exp_2021-12-09-23-36-27_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentConstriantPPO2_None_clip/exp_2021-12-12-19-46-35_cuda:1',
    # ]
    # town03-v1
    eve_paths = [
        '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentD3QN_None_None/exp_2021-12-11-11-49-36_cuda:1',
        '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentModSAC_None_None/exp_2021-12-11-11-49-06_cuda:1',
        '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentPPO2_None_clip/exp_2021-12-12-04-40-01_cuda:1',
        '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentConstriantPPO2_None_clip/exp_2021-12-12-21-56-36_cuda:1',
    ]
    # Label = ['PPO', 'PPO-MN', 'PPO-BETA', 'PPO-D', 'PPO-MD']
    # eve_paths = [
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_None_clip/exp_2021-12-03-21-56-07_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_mg_clip/exp_2021-12-05-00-26-54_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_beta_clip/exp_2021-12-05-00-27-24_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/discrete/AgentPPO2_discrete_clip/exp_2021-12-05-14-43-25_cuda:0',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/discrete/AgentPPO2_discrete_action_dim_clip/exp_2021-12-05-14-43-56_cuda:0',
    # ]

    # Label = ['PPO2', 'PPO2-MN', 'PPO2-MN-CMAES', 'PPO2-BETA', 'PPO2-D', 'PPO2-MD']
    # eve_paths = [
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_None_clip/exp_2021-12-03-21-56-07_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_mg_clip/exp_2021-12-05-00-26-54_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2CMAES_None_clip/exp_2021-12-05-10-43-48_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_beta_clip/exp_2021-12-05-00-27-24_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/discrete/AgentPPO2_discrete_clip/exp_2021-12-05-14-43-25_cuda:0',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/discrete/AgentPPO2_discrete_action_dim_clip/exp_2021-12-05-14-43-56_cuda:0',
    # ]

    # Label = ['DPPO2', 'PPO2', 'HPPO2-3', 'HPPO2-2','MIXPPO2']
    # eve_paths = [
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_discrete_clip/exp_2021-11-19-23-15-36_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_None_clip/exp_2021-11-10-13-02-45_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentHybridPPO2_None_clip/exp_2021-11-27-16-59-02_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentHybridPPO2_None_clip/exp_2021-11-27-17-08-03_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_discrete_action_dim_clip/exp_2021-11-22-16-27-08_cuda:1',
    # ]

    # Label = ['PPO2','PPO2-MG', 'PPO2-MG-CMAES','PPO2-BETA','MIXPPO2']
    # eve_paths = [
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_None_clip/exp_2021-11-10-13-02-45_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_mg_clip/exp_2021-11-20-00-26-56_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2CMAES_None_clip/exp_2021-11-20-22-52-27_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_beta_clip/exp_2021-11-20-00-27-26_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_discrete_action_dim_clip/exp_2021-11-22-16-27-08_cuda:1',
    # ]

    # Label = ['PPO','PPO-LSTM-SS', 'PPO-LSTM-ZS']
    # eve_paths = [
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_None_clip/exp_2021-12-03-21-56-07_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentRNNPPO2_None_clip/hiddenstate_exp_2021-12-03-21-12-46_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentRNNPPO2_None_clip/zerostate_exp_2021-12-05-10-51-39_cuda:1',
    # ]

    # Label = ['PPO2','RNNPPO2-SS', 'RNNPPO2-ZS','RNNPPO2-SS-IS', 'RNNPPO2-ZS-IS']
    # eve_paths = [
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_None_clip/exp_2021-11-10-13-02-45_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentRNNPPO2_None_clip-rnnppo/store_state/exp_2021-11-28-12-12-19_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentRNNPPO2_None_clip-rnnppo/zero-state/exp_2021-11-27-22-15-12_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentRNNPPO2_None_clip-rnnppo/store_state-sequence/exp_2021-11-28-12-13-20_cuda:1',
    #     '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentRNNPPO2_None_clip-rnnppo/zero_state-sequence/exp_2021-11-28-12-13-51_cuda:1',
    # ]

    file_name = []
    for eve_path in eve_paths:
        file_name.append(search_event(eve_path))
    label = 'reward_0'
    sample_step = 2000000
    num_points = 50
    sp_length = 1
    scale = (sample_step / num_points)
    sns.set(style="darkgrid")
    # sns.set(style="darkgrid", font_scale=1.5)
    # fig_dims = (5 * 1.2, 3 * 1.2)
    fig_dims = (3.5 * 1.2, 3.2 * 1.2)
    fig, ax = plt.subplots(figsize=fig_dims)
    plt.figure()
    df = pd.DataFrame()
    for i in range(1, len(file_name) + 1):
        event_path = eve_paths[i - 1]
        scalar_name, scalar_data = readEvent(event_path)
        label = 'return_0' if i == 2 else 'return_0'  # 'return_0' reward_0
        label = 'return_1' if i == 4 else label
        data_mean = event_accumulator.ScalarEvent(*zip(*scalar_data[scalar_name.index(label + "/avg")]))
        data_max = event_accumulator.ScalarEvent(*zip(*scalar_data[scalar_name.index(label + "/max")]))
        data_min = event_accumulator.ScalarEvent(*zip(*scalar_data[scalar_name.index(label + "/min")]))
        length = int(len(data_mean.value) / sp_length)
        y = np.array(data_mean.value)[:length]
        # x = np.arange(0, time[i - 1], (time[i - 1] / length))[:length]
        x = np.array(data_mean.step)[:length]
        for data in [data_max, data_min]:
            y = np.hstack((y, np.array(data.value)[:length]))
            # x = np.hstack((x, x[:length]))
            x = np.hstack((x, np.array(data.step)[:length]))
        tmp_df = pd.DataFrame(np.vstack((x, y)).T)
        ylabel = "Return"
        # xlabel = "Time(s)"
        xlabel = "Interaction Step"
        typelabel = "AlgoType"
        tmp_df.columns = [xlabel, ylabel]
        tmp_df[xlabel] = scale * ((tmp_df[xlabel]) // scale)
        tmp_df[typelabel] = Label[i - 1]
        df = pd.concat([df, tmp_df], axis=0)
        # sns.lineplot(x="interaction step",
        #              y="return",
        #              hue='algo type',
        #              style='algo type',
        #              color='algo type',
        #              # col='algo type',
        #              ci=95,
        #              data=tmp_df)
    df = df.reset_index()
    sns.lineplot(x=xlabel, y=ylabel, hue=typelabel, style=typelabel,
                 ci=95, data=df, ax=ax)
    # plt.plot(time,[tmp_df[ylabel],tmp_df[ylabel]], 'ro', label="2e6")
    # sns.relplot(x="Interaction Step", y="Return", hue='AlgoType', style='AlgoType', kind="line",
    #             ci=95, data=df)
    plt.show()
