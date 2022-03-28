from ray_elegantrl.interaction import make_env
from ray_elegantrl.net import *
import torch
import numpy as np
import os
import time

os.environ["DISPLAY"] = "localhost:10.0"
# os.environ["SDL_VIDEODRIVER"] = "dummy"

RENDER = True
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
    if test_num >= 0:
        eval_policy(policy, env, test_num, seed=0, net=args.net, action_space=args.action_space)
    else:
        one_eval_policy(policy, env, seed=0, net=args.net, action_space=args.action_space)
    del env


# for cpu policy
def eval_policy(policy, env_dict, test_num=100, net=None, action_space=None, seed=0):
    result = {}
    result['return'] = []
    result['avg_v'] = []
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
        distance = 0
        outroute = 0
        collision = 0

        ep_info = {
            'reward': [],
            'velocity':[],
            'velocity_lon': [],
            'velocity_lat': [],
            'delta_steer': [],
            'lat_distance': [],
            'yaw_angle': [],
            'delta_yaw': [],
            'delta_a_lon': [],
            'delta_a_lat': [],
            'delta_jerk_lon': [],
            'delta_jerk_lat': [],
        }

        hidden_state = torch.zeros([1, args.hidden_state_dim], dtype=torch.float32)
        cell_state = torch.zeros([1, args.hidden_state_dim], dtype=torch.float32)
        for i in range(1, env_dict['max_step'] + 1):
            state = torch.as_tensor((state,), dtype=torch.float32).detach_()
            if net in ["d3qn", "discreteppo"]:
                action_idx = policy(state)
                action = np.array(action_space[action_idx])
            elif net in ["rnnppo"]:
                action, hidden_state, cell_state = policy.actor_forward(state, hidden_state, cell_state)
                action = action.detach().numpy()[0]
            elif net in ["hybridppo-3","hybridppo-2"]:
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

            # 'velocity': [],
            # 'delta_steer': [],
            # 'lat_distance': [],
            # 'yaw_angle': [],
            # 'delta_yaw': [],
            # 'delta_a_lon': [],
            # 'delta_a_lat': [],
            # 'delta_jerk_lon': [],
            # 'delta_jerk_lat': [],
            ep_info['reward'].append(reward)
            ep_info['velocity'].append(info['velocity'])
            ep_info['velocity_lon'].append(info['velocity_lon'])
            ep_info['velocity_lat'].append(info['velocity_lat'])
            ep_info['delta_steer'].append(info['delta_steer'])
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
        result['delta_a_lon'].append(a_lon / i)
        result['delta_a_lat'].append(a_lat / i)
        result['delta_jerk_lon'].append(jerk_lon / i)
        result['delta_jerk_lat'].append(jerk_lat / i)
    print(f'test {test_num} episode finished !!!')
    print('-' * 60)
    save_path=f'/home/zgy/repos/ray_elegantrl/veh_control_logs/eval/{args.net}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for k, vl in ep_info.items():
        np.save(f'{save_path}/ep-{k}', np.array(vl))
    for k, vl in result.items():
        print(f'{k}: <mean>{np.array(vl).mean()} | <std>{np.array(vl).std()}')
    print('-' * 60)


def one_eval_policy(policy, env_dict, net=None, action_space=None, seed=0):
    env = make_env(env_dict, seed)
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
    distance = 0
    outroute = 0
    collision = 0

    ss_time = time.time()
    hidden_state = None
    cell_state = None
    for i in range(env_dict['max_step']):
        state = torch.as_tensor((state,), dtype=torch.float32).detach_()
        if net in ["d3qn", "discreteppo"]:
            action_idx = policy(state)
            print(action_idx)
            action = np.array(action_space[action_idx])
        elif net in ["rnnppo"]:
            action, hidden_state, cell_state = policy.actor_forward(state, hidden_state, cell_state)
        else:
            action = policy(state)
            action = action.detach().numpy()[0]
        next_s, reward, done, info = env.step(action)

        print("-" * 60)
        print(env.r_info)
        print("step :", i,
              "\n return:", reward,
              "\n avg_v:", info['velocity'],
              "\n delta_yaw(per step):", info['delta_yaw'],
              "\n delta_steer(per step):", info['delta_steer'],
              "\n lat_distance(per step):", info['lat_distance'],
              "\n yaw_angle(per step):", info['yaw_angle'],
              "\n delta_a_lon(per step):", info['acc_lon'],
              "\n delta_a_lat(per step):", info['acc_lat'],
              "\n delta_jerk_lon(per step):", info['jerk_lon'],
              "\n delta_jerk_lat(per step):", info['jerk_lat'],
              )
        delta_yaw += info['delta_yaw']
        delta_steer += info['delta_steer']
        lat_distance += info['lat_distance']
        yaw_angle += info['yaw_angle']
        a_lon += info['acc_lon']
        a_lat += info['acc_lat']
        jerk_lon += info['jerk_lon']
        jerk_lat += info['jerk_lat']
        distance = info['distance']
        outroute += info['outroute']
        collision += info['collision']
        return_ += reward
        avg_v += info['velocity']

        if done:
            break
        state = next_s
    print("*" * 60)
    print(f"used time:{time.time() - ss_time}s")
    print("step num:", i,
          "\n return:", return_,
          "\n avg_v:", avg_v / i,
          "\n distance:", distance,
          "\n outroute:", outroute,
          "\n collision:", collision,
          "\n delta_yaw(per step):", delta_yaw / i,
          "\n delta_steer(per step):", delta_steer / i,
          "\n lat_distance(per step):", lat_distance / i,
          "\n yaw_angle(per step):", yaw_angle / i,
          "\n delta_a_lon(per step):", a_lon / i,
          "\n delta_a_lat(per step):", a_lat / i,
          "\n delta_jerk_lon(per step):", jerk_lon / i,
          "\n delta_jerk_lat(per step):", jerk_lat / i,
          )


# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Carla RL')
#     parser.add_argument('--reward_type', type=int, default=12)
#     parser.add_argument('--port', nargs='+', type=int, default=[2500])
#     parser.add_argument('--desired_speed', type=float, default=12)
#     parser.add_argument('--max_step', type=int, default=1000)
#     parser.add_argument('--dt', type=float, default=0.05)
#     parser.add_argument('--test_num', type=int, default=10)
#     parser.add_argument('--path', type=str, default="")
#     parser.add_argument('--net', type=str, default="ppo")
#     parser.add_argument('--hidden_state_dim', type=int, default=128)
#     parser.add_argument('--action_space', type=int, default=None)
#     args = parser.parse_args()
#     args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_None_clip/exp_2021-11-10-13-02-45_cuda:1/actor.pth'
#     # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a21_r1_tr700_ms200_False/AgentD3QN_None_None/exp_2021-11-10-13-03-45_cuda:1/actor.pth'
#
#     # policy = ActorPPOMG(mid_dim=2 ** 8, state_dim=env['state_dim'], action_dim=env['action_dim'])
#     # policy = ActorSAC(mid_dim=2 ** 8, state_dim=env['state_dim'], action_dim=env['action_dim'])
#     # policy = ActorHybridPPO(mid_dim=2 ** 8, state_dim=env['state_dim'], action_dim=env['action_dim'])
#     # policy = ActorPPOBeta(mid_dim=2 ** 8, state_dim=env['state_dim'], action_dim=env['action_dim'])
#     if args.net in ["ppo"]:
#         policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
#     elif args.net in ["discreteppo"]:
#         policy = ActorDiscretePPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
#     elif args.net in ["hybridppo"]:
#         policy = ActorHybridPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
#     elif args.net in ["mgppo"]:
#         policy = ActorPPOMG(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
#     elif args.net in ["betappo"]:
#         policy = ActorPPOBeta(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
#     elif args.net in ["sadppo"]:
#         policy = ActorSADPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
#     elif args.net in ["beta2ppo"]:
#         policy = ActorPPOBeta2(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
#     elif args.net in ["sac"]:
#         policy = ActorSAC(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
#     elif args.net in ["rnnppo"]:
#         policy = CarlaRNNPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM,
#                              hidden_state_dim=args.hidden_state_dim)
#     elif args.net in ["d3qn"]:
#         args.action_space = [[a, b] for a in [-1., 0., 1.] for b in [-0.6, -0.3, -0.1, 0., 0.1, 0.3, 0.6]]
#         policy = QNetTwinDuel(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=len(args.action_space))
#     policy.load_state_dict(torch.load(args.path))
#     policy.eval()
#
#     demo(policy, args=args)
#     print(f"finished:{args.net}")
#     print(f"path:{args.path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Carla RL')
    parser.add_argument('--reward_type', type=int, default=12)
    parser.add_argument('--port', nargs='+', type=int, default=[2070])
    parser.add_argument('--desired_speed', type=float, default=12)
    parser.add_argument('--max_step', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--net', type=str, default="ppo")
    parser.add_argument('--hidden_state_dim', type=int, default=128)
    parser.add_argument('--action_space', type=int, default=None)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--if_noise_dt', default=False, action="store_true", )
    args = parser.parse_args()


    # args.net = "cppo"
    args.reward_type=12
    print(f'# {args.net}')
    args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r3_tr700_ms200_False/AgentConstriantPPO2_None_clip/exp_2021-12-01-06-33-04_cuda:1/actor.pth'
    policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    policy.load_state_dict(torch.load(args.path))
    policy.eval()
    demo(policy, args=args)
    print(f"path:{args.path}")

    # args.net = "ppo"
    # args.reward_type=12
    # print(f'# {args.net}-cppo')
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r1_tr700_ms200_False/AgentPPO2_None_clip/exp_2021-11-30-15-03-43_cuda:1/actor.pth'
    # policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")
    #
    # args.net = "ppo"
    # args.reward_type=12
    # print(f'# {args.net}')
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_None_clip/exp_2021-11-10-13-02-45_cuda:1/actor.pth'
    # policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")

    # args.net = "mgppo"
    # print(f'# {args.net}')
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_mg_clip/exp_2021-11-20-00-26-56_cuda:1/actor.pth'
    # policy = ActorPPOMG(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")
    #
    # args.net = "ppocmaes"
    # print(f'# {args.net}')
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2CMAES_None_clip/exp_2021-11-20-22-52-27_cuda:1/actor.pth'
    # policy = ActorPPOMG(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")
    #
    # args.net = "betappo"
    # print(f'# {args.net}')
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_beta_clip/exp_2021-11-20-00-27-26_cuda:1/actor.pth'
    # policy = ActorPPOBeta(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")
    #
    # # args.net = "beta2ppo"
    # # print(f'# {args.net}')
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_beta2_clip/exp_2021-11-20-22-52-57_cuda:1/actor.pth'
    # # policy = ActorPPOBeta2(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # # policy.load_state_dict(torch.load(args.path))
    # # policy.eval()
    # # demo(policy, args=args)
    # # print(f"path:{args.path}")
    #
    # args.net = "sac"
    # print(f'# {args.net}')
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentModSAC_None_None/exp_2021-11-17-00-14-23_cuda:1/actor.pth'
    # # policy = ActorSAC(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # # policy.load_state_dict(torch.load(args.path))
    # # policy.eval()
    # # demo(policy, args=args)
    # # print("1")
    # # print(f"path:{args.path}")
    #
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentModSAC_None_None/exp_2021-11-17-22-17-56_cuda:1/actor.pth'
    # policy = ActorSAC(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print("2")
    # print(f"path:{args.path}")
    #
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentModSAC_None_None/bad-exp_2021-11-17-22-18-26_cuda:1/actor.pth'
    # # policy = ActorSAC(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # # policy.load_state_dict(torch.load(args.path))
    # # policy.eval()
    # # demo(policy, args=args)
    # # print("3")
    # # print(f"path:{args.path}")
    #
    # args.net = "d3qn"
    # print(f'# {args.net}')
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentD3QN_None_None/a21-0.6-exp_2021-11-10-13-03-45_cuda:1/actor.pth'
    # args.action_space = [[a, b] for a in [-1., 0., 1.] for b in [-0.6, -0.3, -0.1, 0., 0.1, 0.3, 0.6]]
    # policy = QNetTwinDuel(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=len(args.action_space))
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")
    #
    # args.net = "discreteppo"
    # print(f'# {args.net}')
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_discrete_clip/exp_2021-11-19-23-15-36_cuda:1/actor.pth'
    # args.action_space = [[a, b] for a in [-1., 0., 1.] for b in [-0.6, -0.3, -0.1, 0., 0.1, 0.3, 0.6]]
    # policy = ActorDiscretePPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=len(args.action_space))
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")
    #
    # args.net = "hybridppo-share-3"
    # print(f'# {args.net}')
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentHybridPPO2_None_clip/share-3/exp_2021-11-29-23-59-45_cuda:1/actor.pth'
    # policy = ActorHybridPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=[2, 3 ** 2],if_shared=True)
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")
    #
    # args.net = "hybridppo-share-3-sp"
    # print(f'# {args.net}')
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentHybridPPO2_None_clip/share-3/sp-exp_2021-11-30-00-00-15_cuda:1/actor.pth'
    # policy = ActorHybridPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=[2, 3 ** 2], if_shared=True)
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")

    # args.net = "hybridppo-2"
    # print(f'# {args.net}')
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentHybridPPO2_None_clip/3/exp_2021-11-30-00-00-15_cuda:1/actor.pth'
    # policy = ActorHybridPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=[2, 3 ** 2],if_shared=True)
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")
    #
    # args.net = "sadppo"
    # print(f'# {args.net}')
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_discrete_action_dim_clip/exp_2021-11-22-16-27-08_cuda:1/actor.pth'
    # policy = ActorSADPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # policy.set_sp_a_num([3, 11])
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")
    #
    # args.net = "rnnppo"
    # print(f'# {args.net}:stored-state')
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentRNNPPO2_None_clip-rnnppo/store_state/exp_2021-11-18-23-41-47_cuda:1/actor.pth'
    # args.hidden_state_dim = 32
    # policy = CarlaRNNPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM,
    #                      hidden_state_dim=args.hidden_state_dim)
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")

    # args.net = "rnnppo"
    # print(f'# {args.net}:zero-state')
    # args.hidden_state_dim = 32
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentRNNPPO2_None_clip-rnnppo/zero-state/exp_2021-11-18-23-42-17_cuda:1/actor.pth'
    # policy = CarlaRNNPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM,
    #                      hidden_state_dim=args.hidden_state_dim)
    # policy.load_state_dict(torch.load(args.path))
    # policy.eval()
    # demo(policy, args=args)
    # print(f"path:{args.path}")
