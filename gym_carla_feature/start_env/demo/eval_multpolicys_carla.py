from ray_elegantrl.interaction import make_env
from ray_elegantrl.net import *
import torch
import numpy as np
import os
import time

# os.environ["DISPLAY"] = "localhost:13.0"
os.environ["SDL_VIDEODRIVER"] = "dummy"

RENDER = False
ENV_ID = "carla-v2"
STATE_DIM = 50
ACTION_DIM = 2
REWARD_DIM = 1
TARGET_RETURN = 700


def demo(policy, args):
    from gym_carla_feature.start_env.config import params
    params['port'] = (args.port[0] if len(args.port) == 1 else args.port) if hasattr(args, 'port') else 2280
    params['max_waypt'] = args.max_waypt if hasattr(args, 'max_waypt') else 12
    params['max_step'] = args.max_step if hasattr(args, 'max_step') else 1000
    params['dt'] = args.dt if hasattr(args, 'dt') else 1 / 20
    params['if_noise_dt'] = args.if_noise_dt if hasattr(args, 'if_noise_dt') else False
    params['noise_std'] = args.noise_std if hasattr(args, 'noise_std') else 0.01
    params['noise_interval'] = args.noise_interval if hasattr(args, 'noise_interval') else 10
    params['render'] = RENDER
    params['autopilot'] = args.autopilot if hasattr(args, 'autopilot') else False
    params['desired_speed'] = args.desired_speed if hasattr(args, 'desired_speed') else 20
    params['out_lane_thres'] = args.out_lane_thres if hasattr(args, 'out_lane_thres') else 5
    params['sampling_radius'] = args.sampling_radius if hasattr(args, 'sampling_radius') else 3
    # params['obs_space_type'] = args.obs_space_type if hasattr(args,'obs_space_type') else ['orgin_state', 'waypoint']

    params['town'] = args.town if hasattr(args, 'town') else 'Town07'
    params['task_mode'] = args.task_mode if hasattr(args, 'task_mode') else 'mountainroad'
    params['reward_type'] = args.reward_type if hasattr(args, 'reward_type') else 1
    params['if_dest_end'] = args.if_dest_end if hasattr(args, 'if_dest_end') else False
    test_num = args.test_num if hasattr(args, 'test_num') else 1000

    env = {
        'id': ENV_ID,
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'action_type': params['action_type'],
        'reward_dim': REWARD_DIM,
        'target_reward': TARGET_RETURN,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }

    total_result = []
    dir, subdirs, subfiles = os.walk(args.path).__next__()
    if len(subdirs) == 0:  # one policy eval
        policy_path = dir + "/actor.pth"
        policy.load_state_dict(torch.load(policy_path))
        policy.eval()
        result = eval_policy(policy, env, test_num, seed=0, net=args.net, action_space=args.action_space)
        total_result.append(result)
    else:  # multi policy eval
        for subdir in subdirs:
            policy_path = dir + "/" + subdir + "/actor.pth"
            policy.load_state_dict(torch.load(policy_path))
            policy.eval()
            result = eval_policy(policy, env, test_num, seed=0, net=args.net, action_space=args.action_space,
                                 if_point=True)
            total_result.append(result)
        print(f"path:{args.path}")
        safe_data = []
        comfort_data = []
        for k, vl in total_result[0].items():
            data = []
            for result in total_result:
                data.append(np.array(result[k]).mean())
            print(f'{k}: <mean>{np.array(data).mean()} | <std>{np.array(data).std()}')
            #
            if k in ['outroute', 'collision']:
                safe_data.append(np.array(data))
            if k in ['delta_a_lon', 'delta_a_lat', 'delta_jerk_lon', 'delta_jerk_lat']:
                comfort_data.append(np.array(data))
        safe = []
        for outlane, collsion in zip(safe_data[0], safe_data[1]):
            safe.append((outlane or collsion))
        comfort = []
        for delta_a_lon, delta_a_lat, delta_jerk_lon, delta_jerk_lat in zip(comfort_data[0], comfort_data[1],
                                                                            comfort_data[2], comfort_data[3]):
            comfort.append((1 / (1 + (delta_a_lon + delta_a_lat + delta_jerk_lon + delta_jerk_lat))))
        print(f'safe: <mean>{1 - np.array(safe).mean()} | <std>{np.array(safe).std()}')
        print(f'comfort: <mean>{np.array(comfort).mean()} | <std>{np.array(comfort).std()}')


# for cpu policy
def eval_policy(policy, env_dict, test_num=100, net=None, action_space=None, seed=0, if_point=True):
    result = {}
    result['return'] = []
    result['avg_v'] = []
    result['velocity_lon'] = []
    result['velocity_lat'] = []
    result['distance'] = []
    result['step_num'] = []
    result['delta_yaw'] = []
    result['delta_steer'] = []
    result['lat_distance'] = []
    result['yaw_angle'] = []
    result['delta_a_lon'] = []
    result['delta_a_lat'] = []
    result['delta_jerk_lon'] = []
    result['delta_jerk_lat'] = []
    result['outroute'] = []
    result['collision'] = []
    result['time'] = []
    env = make_env(env_dict, seed)
    for ep_i in range(test_num):
        ss_time = time.time()
        state = env.reset()
        return_ = 0
        delta_yaw = 0
        delta_steer = 0
        a_lon = 0
        a_lat = 0
        jerk_lon = 0
        jerk_lat = 0
        lat_distance = 0
        yaw_angle = 0
        avg_v = 0
        velocity_lon = 0
        velocity_lat = 0
        distance = 0
        outroute = 0
        collision = 0
        time_info = 0
        ep_info = {
            'position':[],
            'reward': [],
            'velocity': [],
            'velocity_lon': [],
            'velocity_lat': [],
            'delta_steer': [],
            'lon_action': [],
            'steer': [],
            'a0': [],
            'a1': [],
            'lat_distance': [],
            'yaw_angle': [],
            'delta_yaw': [],
            'delta_a_lon': [],
            'delta_a_lat': [],
            'delta_jerk_lon': [],
            'delta_jerk_lat': [],
        }
        if hasattr(policy, 'hidden_state_dim'):
            hidden_state = torch.zeros([1, policy.hidden_state_dim], dtype=torch.float32)
            cell_state = torch.zeros([1, policy.hidden_state_dim], dtype=torch.float32)
        for i in range(1, env_dict['max_step'] + 1):
            state = torch.as_tensor((state,), dtype=torch.float32).detach_()
            if net in ["d3qn", "discreteppo"]:
                action_idx = policy(state)
                action = np.array(action_space[action_idx])
            elif net in ["rnnppo"]:
                action, hidden_state, cell_state = policy.actor_forward(state, hidden_state, cell_state)
                action = action.detach().numpy()[0]
            elif net in ["hybridppo-3", "hybridppo-2"]:
                action = policy(state)
                action = action.detach().numpy()[0]
                if net in ["hybridppo-3"]:
                    def modify_action(action):
                        def mapping(da_dim, x):
                            if da_dim == 0:
                                return -abs(x)
                            elif da_dim == 2:
                                return abs(x)
                            else:
                                return x * 0.

                        da_idx = int(action[-1])
                        mod_a = np.zeros(action[:-1].shape)
                        mod_a[0] = mapping(da_idx // 3, action[0])
                        mod_a[1] = mapping(da_idx % 3, action[1])
                        return mod_a
                elif net in ["hybridppo-2"]:
                    def modify_action(action):
                        def mapping(da_dim, x):
                            if da_dim == 1:
                                return x
                            else:
                                return x * 0.

                        da_idx = int(action[-1])
                        mod_a = np.zeros(action[:-1].shape)
                        mod_a[0] = mapping(da_idx // 2, action[0])
                        mod_a[1] = mapping(da_idx % 2, action[1])
                        return mod_a
                else:
                    def modify_action(action):
                        pass

                action = modify_action(action)
            else:
                action = policy(state)
                action = action.detach().numpy()[0]
            next_s, reward, done, info = env.step(action)

            delta_yaw += abs(info['delta_yaw'])
            delta_steer += abs(info['delta_steer'])
            lat_distance += abs(info['lat_distance'])
            yaw_angle += abs(info['yaw_angle'])
            a_lon += abs(info['acc_lon'])
            a_lat += abs(info['acc_lat'])
            jerk_lon += abs(info['jerk_lon'])
            jerk_lat += abs(info['jerk_lat'])
            distance = info['distance']
            outroute += info['outroute']
            collision += info['collision']
            return_ += reward
            avg_v += info['velocity']
            velocity_lon += info['velocity_lon']
            velocity_lat += abs(info['velocity_lat'])
            time_info += info['time']
            # 'velocity': [],
            # 'delta_steer': [],
            # 'lat_distance': [],
            # 'yaw_angle': [],
            # 'delta_yaw': [],
            # 'delta_a_lon': [],
            # 'delta_a_lat': [],
            # 'delta_jerk_lon': [],
            # 'delta_jerk_lat': [],
            ego_location = env.ego.get_transform().location
            ep_info['position'].append([ego_location.x,ego_location.y])
            ep_info['reward'].append(reward)
            ep_info['velocity'].append(info['velocity'])
            ep_info['velocity_lon'].append(info['velocity_lon'])
            ep_info['velocity_lat'].append(info['velocity_lat'])
            ep_info['delta_steer'].append(info['delta_steer'])
            ep_info['lon_action'].append(info['lon_action'])
            ep_info['steer'].append(info['steer'])
            ep_info['a0'].append(info['a0'])
            ep_info['a1'].append(info['a1'])
            ep_info['yaw_angle'].append(info['yaw_angle'])
            ep_info['delta_yaw'].append(info['delta_yaw'])
            ep_info['lat_distance'].append(info['lat_distance'])
            ep_info['delta_a_lon'].append(info['acc_lon'])
            ep_info['delta_a_lat'].append(info['acc_lat'])
            ep_info['delta_jerk_lon'].append(info['jerk_lon'])
            ep_info['delta_jerk_lat'].append(info['jerk_lat'])
            if done:
                break
            state = next_s
        # print("*" * 60)
        # print(f"ep{ep_i} used time:{time.time() - ss_time}s")
        # print("step num:", i,
        #       "\n return:", return_,
        #       "\n avg_v:", avg_v / i,
        #       "\n distance:", distance,
        #       "\n outroute:", outroute,
        #       "\n collision:", collision,
        #       "\n delta_yaw(per step):", delta_yaw / i,
        #       "\n delta_steer(per step):", delta_steer / i,
        #       "\n lat_distance(per step):", lat_distance / i,
        #       "\n yaw_angle(per step):", yaw_angle / i,
        #       "\n delta_a_lon(per step):", a_lon / i,
        #       "\n delta_a_lat(per step):", a_lat / i,
        #       "\n delta_jerk_lon(per step):", jerk_lon / i,
        #       "\n delta_jerk_lat(per step):", jerk_lat / i,
        #       )
        result['step_num'].append(i)
        result['return'].append(return_)
        result['avg_v'].append(avg_v / i)
        result['distance'].append(distance)
        result['collision'].append(collision)
        result['outroute'].append(outroute)
        result['delta_yaw'].append(delta_yaw / i)
        result['delta_steer'].append(delta_steer / i)
        result['lat_distance'].append(lat_distance / i)
        result['yaw_angle'].append(yaw_angle / i)
        result['velocity_lon'].append(velocity_lon / i)
        result['velocity_lat'].append(velocity_lat / i)
        result['delta_a_lon'].append(a_lon / i)
        result['delta_a_lat'].append(a_lat / i)
        result['delta_jerk_lon'].append(jerk_lon / i)
        result['delta_jerk_lat'].append(jerk_lat / i)
        result['time'].append(time_info)
    # print(f'test {test_num} episode finished !!!')
    # print('-' * 60)
    save_path = f'/home/zgy/repos/ray_elegantrl/veh_control_logs/eval/{net}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if if_point:
        for k, vl in ep_info.items():
            np.save(f'{save_path}/ep-{k}', np.array(vl))
        safe_data = []
        comfort_data = []
        for k, vl in result.items():
            print(f'{k}: <mean>{np.array(vl).mean()} | <std>{np.array(vl).std()}')
            if k in ['outroute', 'collision']:
                safe_data.append(np.array(vl))
            if k in ['delta_a_lon', 'delta_a_lat', 'delta_jerk_lon', 'delta_jerk_lat']:
                comfort_data.append(np.array(vl))

        safe = []
        for outlane, collsion in zip(safe_data[0], safe_data[1]):
            safe.append((outlane or collsion))
        comfort = []
        for delta_a_lon, delta_a_lat, delta_jerk_lon, delta_jerk_lat in zip(comfort_data[0], comfort_data[1],
                                                                            comfort_data[2], comfort_data[3]):
            comfort.append((1 / (1 + (delta_a_lon + delta_a_lat + delta_jerk_lon + delta_jerk_lat))))
        print(f'safe: <mean>{1 - np.array(safe).mean()} | <std>{np.array(safe).std()}')
        print(f'comfort: <mean>{np.array(comfort).mean()} | <std>{np.array(comfort).std()}')
        print('-' * 60)
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Carla RL')
    parser.add_argument('--reward_type', type=int, default=12)
    parser.add_argument('--port', nargs='+', type=int, default=[2050])
    parser.add_argument('--desired_speed', type=float, default=12)
    parser.add_argument('--max_step', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--net', type=str, default="ppo")
    parser.add_argument('--hidden_state_dim', type=int, default=128)
    # parser.add_argument('--town', type=str, default='Town03')
    # parser.add_argument('--task_mode', type=str, default='urbanroad')
    parser.add_argument('--action_space', type=int, default=None)
    parser.add_argument('--noise_std', type=float, default=0.01)
    parser.add_argument('--noise_interval', type=int, default=1)
    parser.add_argument('--if_noise_dt', default=False, action="store_true", )
    args = parser.parse_args()

    # args.net = "cppo"
    # print(f'# {args.net}')
    # print("town07-v1")
    # args.max_step = 800
    # print("00")
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r4_tr200_ms200_False/AgentConstriantPPO2_None_clip/exp_2021-12-12-13-39-39_cuda:1/model00'
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r4_tr200_ms200_False/AgentConstriantPPO2_None_clip/exp_2021-12-12-13-26-49_cuda:1/model00'
    # # args.path='/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r4_tr200_ms200_False/AgentConstriantPPO2_None_clip/exp_2021-12-12-00-10-03_cuda:1/model00'
    # # args.path='/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r4_tr200_ms200_False/AgentConstriantPPO2_None_clip/exp_2021-12-11-18-13-47_cuda:1/model00'
    # # args.path='/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r3_tr200_ms200_False/AgentConstriantPPO2_None_clip/exp_2021-12-11-12-21-00_cuda:1/model00'
    # # args.path='/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/constriant/AgentConstriantPPO2_None_clip/r15_v2/0.1_0.01_bad_gap0_exp_2021-12-09-23-30-00_cuda:1/model00'
    # policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # demo(policy, args)
    # print("01")
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r4_tr200_ms200_False/AgentConstriantPPO2_None_clip/exp_2021-12-12-13-39-39_cuda:1/model01'
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r4_tr200_ms200_False/AgentConstriantPPO2_None_clip/exp_2021-12-12-13-26-49_cuda:1/model01'
    # # args.path='/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r4_tr200_ms200_False/AgentConstriantPPO2_None_clip/exp_2021-12-12-00-10-03_cuda:1/model01'
    # # args.path='/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r4_tr200_ms200_False/AgentConstriantPPO2_None_clip/exp_2021-12-11-18-13-47_cuda:1/model01'
    # # args.path='/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r3_tr200_ms200_False/AgentConstriantPPO2_None_clip/exp_2021-12-11-12-21-00_cuda:1/model01'
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/constriant/AgentConstriantPPO2_None_clip/r15_v2/0.1_0.01_bad_gap0_exp_2021-12-09-23-30-00_cuda:1/model01'
    # policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # demo(policy, args)

    # print("town07-v2")
    # args.max_step = 4000
    # print("00")
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentConstriantPPO2_None_clip/exp_2021-12-12-19-46-35_cuda:1/model00'
    # policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # demo(policy, args)
    # print("01")
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentConstriantPPO2_None_clip/exp_2021-12-12-19-46-35_cuda:1/model01'
    # policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # demo(policy, args)

    # print("town03-v1")
    # args.town = 'Town03'
    # args.task_mode = 'urbanroad'
    # args.max_step = 4000
    # print("00")
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentConstriantPPO2_None_clip/exp_2021-12-12-21-56-36_cuda:1/model00'
    # policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # demo(policy, args)
    # print("01")
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentConstriantPPO2_None_clip/exp_2021-12-12-21-56-36_cuda:1/model01'
    # policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # demo(policy, args)

    # args.net = "ppo"
    # print(f'# {args.net}')
    # print("town07-v1")
    # print("00")
    # args.max_step = 800
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_None_clip/exp_2021-12-03-21-56-07_cuda:1/model'
    # print("town07-v2")
    # print("00")
    # args.max_step = 4000
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentPPO2_None_clip/exp_2021-12-09-23-36-27_cuda:1/model00'
    # print("town03-v1")
    # print("00")
    # args.town = 'Town03'
    # args.task_mode = 'urbanroad'
    # args.max_step = 4000
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentPPO2_None_clip/exp_2021-12-12-04-40-01_cuda:1/model00'
    # policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # demo(policy, args)

    # args.net = "mgppo"
    # print(f'# {args.net}')
    # print("town07-v1")
    # print("00")
    # args.max_step = 800
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_mg_clip/exp_2021-12-05-00-26-54_cuda:1/model'
    # policy = ActorPPOMG(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # demo(policy, args)

    # args.net = "ppocmaes"
    # print(f'# {args.net}')
    # print("town07-v1")
    # print("00")
    # args.max_step = 800
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2CMAES_None_clip/exp_2021-12-05-10-43-48_cuda:1/model'
    # policy = ActorPPOMG(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # demo(policy, args)
    #
    # args.net = "betappo"
    # print(f'# {args.net}')
    # print("town07-v1")
    # print("00")
    # args.max_step = 800
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_beta_clip/exp_2021-12-05-00-27-24_cuda:1/model'
    # policy = ActorPPOBeta(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # demo(policy, args)

    # args.net = "d3qn"
    # print(f'# {args.net}')
    # print("town07-v1")
    # print("00")
    # args.max_step=800
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/discrete/AgentD3QN_None_None/exp_2021-12-04-01-01-20_cuda:1/model'
    # # print("town07-v2")
    # # print("00")
    # # args.max_step=4000
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentD3QN_None_None/exp_2021-12-09-23-37-27_cuda:1/model00'
    # print("town03-v1")
    # print("00")
    # args.town = 'Town03'
    # args.task_mode = 'urbanroad'
    # args.max_step = 4000
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentD3QN_None_None/exp_2021-12-11-11-49-36_cuda:1/model00'
    # args.action_space = [[a, b] for a in [-1., 0., 1.] for b in [-0.6, -0.3, -0.1, 0., 0.1, 0.3, 0.6]]
    # policy = QNetTwinDuel(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=len(args.action_space))
    # demo(policy, args)

    args.net = "sac"
    print(f'# {args.net}')
    print("town07-v1")
    print("00")
    args.max_step = 800
    args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentModSAC_None_None_RS5/exp_2021-12-06-20-53-45_cuda:1/model00'
    # # print("town07-v2")
    # # print("00")
    # # args.max_step = 4000
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentModSAC_None_None/exp_2021-12-10-11-12-56_cuda:1/model00'
    # print("town03-v1")
    # print("00")
    # args.town = 'Town03'
    # args.task_mode = 'urbanroad'
    # args.max_step = 4000
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentModSAC_None_None/exp_2021-12-11-11-49-06_cuda:1/model00'
    policy = ActorSAC(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    demo(policy, args)

    # args.net = "discreteppo"
    # print(f'# {args.net}')
    # print("town07-v1")
    # print("00")
    # args.max_step = 800
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/discrete/AgentPPO2_discrete_clip/exp_2021-12-05-14-43-25_cuda:0/model'
    # args.action_space = [[a, b] for a in [-1., 0., 1.] for b in [-0.6, -0.3, -0.1, 0., 0.1, 0.3, 0.6]]
    # policy = ActorDiscretePPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=len(args.action_space))
    # demo(policy, args)
    # #
    # args.net = "sadppo"
    # print(f'# {args.net}')
    # print("town07-v1")
    # print("00")
    # args.max_step = 800
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/discrete/AgentPPO2_discrete_action_dim_clip/exp_2021-12-05-14-43-56_cuda:0/model'
    # policy = ActorSADPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # policy.set_sp_a_num([3, 11])
    # demo(policy, args)
    #
    # args.net = "rnnppo"
    # print(f'# {args.net}:stored-state')
    # print("town07-v1")
    # print("00")
    # args.max_step = 800
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentRNNPPO2_None_clip/hiddenstate_exp_2021-12-03-21-12-46_cuda:1/model'
    # args.hidden_state_dim = 32
    # policy = CarlaRNNPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM,
    #                      hidden_state_dim=args.hidden_state_dim)
    # demo(policy, args)
    #
    # args.net = "rnnppo"
    # print(f'# {args.net}:zero-state')
    # print("town07-v1")
    # print("00")
    # args.max_step = 800
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentRNNPPO2_None_clip/zerostate_exp_2021-12-05-10-51-39_cuda:1/model'
    # args.hidden_state_dim = 32
    # policy = CarlaRNNPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM,
    #                      hidden_state_dim=args.hidden_state_dim)
    # demo(policy, args)
