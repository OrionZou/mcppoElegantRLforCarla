import os
import time
import carla
import gym
import numpy as np
import gym_carla_feature
from gym_carla_feature.start_env.misc import set_carla_transform

os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["DISPLAY"] = "localhost:12.0"
RENDER = False


def rule_demo(args):
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
    # one_eval(args,params)
    eval(args, params)


def eval(args, params):
    debug = args.debug if hasattr(args, 'debug') else False
    test_num = args.test_num if hasattr(args, 'test_num') else 1000
    env = gym.make('carla-v2', params=params)
    from gym_carla_feature.start_env.navigation.behavior_agent import BehaviorAgent
    result = {}
    result['return'] = []
    result['avg_r'] = []
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
    for ep_i in range(test_num):
        ss_time = time.time()
        obs = env.reset()
        return_ = 0
        total_time = 0
        total_steps = 0
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
        agent = BehaviorAgent(env.ego, behavior='normal')
        # agent = BehaviorAgent(env.ego, behavior='cautious')
        # agent = BehaviorAgent(env.ego, behavior='aggressive')
        # agent.set_destination(set_carla_transform(env.route_begin).location,
        #                       set_carla_transform(env.route_dest).location,
        #                       clean=True)
        agent.set_global_plan(env.routeplanner.get_waypoints_queue(), clean=True)
        agent.update_information(env.ego, params['desired_speed'] * 3.6)
        for i in range(1, env.max_step + 1):

            # # top view
            if RENDER:
                spectator = env.world.get_spectator()
                transform = env.ego.get_transform()
                spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40),
                                                        carla.Rotation(pitch=-90)))
            act_control = agent.run_step(debug=debug)
            start_time = time.time()
            obs, r, done, info = env.step(act_control)
            agent.update_information(env.ego, params['desired_speed'] * 3.6)
            if agent.is_empty_plan():
                temp_deque = env.routeplanner.get_waypoints_queue()
                tem_transform = agent._local_planner.waypoints_queue[-1][0].transform
                while not tem_transform == temp_deque[0][0].transform:
                    temp_deque.popleft()
                agent.set_global_plan(temp_deque, clean=False)
            avg_v += info['velocity']
            velocity_lon += info['velocity_lon']
            velocity_lat += abs(info['velocity_lat'])
            end_time = time.time()
            curr_time = end_time - start_time
            # print(f"run time:{curr_time}")
            total_time += curr_time
            total_steps += 1

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
            time_info += info['time']

            ego_location = env.ego.get_transform().location
            ep_info['position'].append([ego_location.x, ego_location.y])
            ep_info['reward'].append(r)
            ep_info['velocity'].append(info['velocity'])
            ep_info['velocity_lon'].append(info['velocity_lon'])
            ep_info['velocity_lat'].append(info['velocity_lat'])
            ep_info['delta_steer'].append(info['delta_steer'])
            ep_info['lon_action'].append(info['lon_action'])
            ep_info['steer'].append(info['steer'])
            ep_info['a0'].append(info['a0'])
            ep_info['a1'].append(info['a1'])
            ep_info['yaw_angle'].append(info['yaw_angle'])
            ep_info['lat_distance'].append(info['lat_distance'])
            ep_info['delta_yaw'].append(info['delta_yaw'])
            ep_info['delta_a_lon'].append(info['acc_lon'])
            ep_info['delta_a_lat'].append(info['acc_lat'])
            ep_info['delta_jerk_lon'].append(info['jerk_lon'])
            ep_info['delta_jerk_lat'].append(info['jerk_lat'])
            return_ += r
            if done:
                break
        # print("*"*60)
        # print(f"ep{ep_i} used time:{time.time()-ss_time}s")
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
        result['avg_r'].append(return_ / i)
        result['avg_v'].append(avg_v / i)
        result['velocity_lat'].append(velocity_lat / i)
        result['velocity_lon'].append(velocity_lon / i)
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
        result['time'].append(time_info)
    print(f'test {test_num} episode finished !!!')
    save_path = f'/home/zgy/repos/ray_elegantrl/veh_control_logs/eval/pid'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for k, vl in ep_info.items():
        np.save(f'{save_path}/ep-{k}', np.array(vl))
    safe_data = []
    comfort_data = []
    for k, vl in result.items():
        print(f'{k}: <mean>{np.array(vl).mean(axis=0)} | <std>{np.array(vl).std(axis=0)}')
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


def one_eval(args, params):
    debug = args.debug if hasattr(args, 'debug') else False
    test_num = args.test_num if hasattr(args, 'test_num') else 1000
    env = gym.make('carla-v2', params=params)
    from gym_carla_feature.start_env.navigation.behavior_agent import BehaviorAgent

    ss_time = time.time()
    obs = env.reset()
    return_ = 0
    total_time = 0
    total_steps = 0
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

    agent = BehaviorAgent(env.ego, behavior='normal')
    # agent = BehaviorAgent(env.ego, behavior='cautious')
    # agent = BehaviorAgent(env.ego, behavior='aggressive')
    # agent.set_destination(set_carla_transform(env.route_begin).location,
    #                       set_carla_transform(env.route_dest).location,
    #                       clean=True)
    agent.set_global_plan(env.routeplanner.get_waypoints_queue(), clean=True)
    agent.update_information(env.ego, params['desired_speed'] * 3.6)
    for i in range(1, env.max_step + 1):

        # # top view
        if RENDER:
            spectator = env.world.get_spectator()
            transform = env.ego.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40),
                                                    carla.Rotation(pitch=-90)))
        act_control = agent.run_step(debug=debug)
        start_time = time.time()
        obs, r, done, info = env.step(act_control)
        agent.update_information(env.ego, params['desired_speed'] * 3.6)
        if agent.is_empty_plan():
            temp_deque = env.routeplanner.get_waypoints_queue()
            tem_transform = agent._local_planner.waypoints_queue[-1][0].transform
            while not tem_transform == temp_deque[0][0].transform:
                temp_deque.popleft()
            agent.set_global_plan(temp_deque, clean=False)

        end_time = time.time()
        curr_time = end_time - start_time
        # print(f"run time:{curr_time}")
        total_time += curr_time
        total_steps += 1
        print("-" * 60)
        print(env.r_info)
        print("step :", i,
              "\n return:", r,
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
        avg_v += info['velocity']
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
        return_ += r
        if done:
            break
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Carla RL')
    # parser.add_argument('--demo_type', type=str, default='ppo2')
    parser.add_argument('--debug', default=False, action="store_true", )
    # parser.add_argument('--policy_type', type=str, default=None)
    # parser.add_argument('--objective_type', type=str, default='clip')
    # parser.add_argument('--if_ir', default=False, action="store_true", )
    # parser.add_argument('--lambda_entropy', type=float, default=0.01)
    # parser.add_argument('--ratio_clip', type=float, default=0.25)
    # parser.add_argument('--lambda_gae_adv', type=float, default=0.97)
    # parser.add_argument('--if_dest_end', default=False, action="store_true", )
    # parser.add_argument('--discrete_steer', nargs='+', type=float, default=1)
    parser.add_argument('--reward_type', type=int, default=12)
    parser.add_argument('--port', nargs='+', type=int, default=[2050])
    parser.add_argument('--desired_speed', type=float, default=12)
    parser.add_argument('--max_step', type=int, default=800)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--test_num', type=int, default=1)
    # parser.add_argument('--town', type=str, default='Town03')
    # parser.add_argument('--task_mode', type=str, default='urbanroad')
    parser.add_argument('--noise_std', type=float, default=0.01)
    parser.add_argument('--noise_interval', type=int, default=1)
    parser.add_argument('--if_noise_dt', default=False, action="store_true", )
    args = parser.parse_args()

    noise_std_list = [0.0]
    args.if_noise_dt = False
    # noise_std_list = [0., 0.01, 0.05, 0.1]
    for noise_std in noise_std_list:
        args.noise_std = noise_std
        print('*' * 10, noise_std, '*' * 10)
        rule_demo(args)

    # v_list = [9, 11, 13, 14]
    # for v in v_list:
    #     args.desired_speed = v
    #     print('*' * 10, v, '*' * 10)
    #     rule_demo(args)
