from ray_elegantrl.interaction import make_env
from ray_elegantrl.agent import AgentPPO2, AgentModSAC, AgentSafePPO2
import torch
import numpy as np
import os

os.environ["DISPLAY"] = "localhost:10.0"


# os.environ["SDL_VIDEODRIVER"] = "dummy"

# for cpu policy
def eval_policy(policy, env_dict, seed):
    result = {'step_num': [],
              'delta_yaw': []}
    env = make_env(env_dict, seed)
    # while True:
    for _ in range(100):
        state = env.reset()
        delta_yaw = 0
        res = {}
        res['reward'] = 0
        for i in range(env_dict['max_step']):
            ego_trans = env.ego.get_transform()
            ego_x, ego_y = ego_trans.location.x, ego_trans.location.y
            ego_yaw = (ego_trans.rotation.yaw) / 180 * np.pi
            next_s, reward, done, info = env.step(safe_exploit_policy(state, policy))
            ego_trans = env.ego.get_transform()
            ego_x, ego_y = ego_trans.location.x, ego_trans.location.y
            ego_yaw2 = (ego_trans.rotation.yaw) / 180 * np.pi
            delta_yaw += abs(ego_yaw2 - ego_yaw)
            res['reward'] += reward
            if done:
                break
            state = next_s
        print("step num:", i, " delta_yaw(per step):", delta_yaw / i)
        result['step_num'].append(i)
        result['delta_yaw'].append(delta_yaw / i)
        print(res)
    print("------!RESULT!-------")
    print("step num:", np.array(result['step_num']).mean(),
          " delta_yaw(per step):", np.array(result['delta_yaw']).mean())


def safe_exploit_policy(state, policy):
    safe_actions = [0., 0.2, 0.4, 0.6, 0.8, 1.]
    state = torch.as_tensor((state,), dtype=torch.float32).detach_()
    action_tensor = policy[0](state)
    action = action_tensor[0].detach().numpy()
    a_prob = policy[1].get_a_prob(torch.cat([state, action_tensor], dim=1))[0].detach().numpy()
    # steer_clip = rd.choice(a_prob.shape[0], p=a_prob)
    steer_clip = safe_actions[a_prob.argmax(axis=0)]
    if abs(action[1]) > steer_clip:
        action[1] = action[1] * steer_clip
    print(action, steer_clip)
    return action


def one_eval_policy(policy, env_dict, seed):
    result = {'step_num': [],
              'delta_yaw': []}
    env = make_env(env_dict, seed)
    # while True:

    state = env.reset()
    delta_yaw = 0
    res = {}
    res['reward'] = 0
    for i in range(env_dict['max_step']):
        action = policy(torch.as_tensor((state,), dtype=torch.float32).detach_())
        ego_trans = env.ego.get_transform()
        ego_x, ego_y = ego_trans.location.x, ego_trans.location.y
        ego_yaw = (ego_trans.rotation.yaw) / 180 * np.pi
        next_s, reward, done, info = env.step(action.detach().numpy()[0])
        ego_trans = env.ego.get_transform()
        ego_x, ego_y = ego_trans.location.x, ego_trans.location.y
        ego_yaw2 = (ego_trans.rotation.yaw) / 180 * np.pi
        delta_yaw += abs(ego_yaw2 - ego_yaw)
        res['reward'] += reward
        if done:
            break
        state = next_s
    print("step num:", i, " delta_yaw(per step):", delta_yaw / i)
    print(res)
    result['step_num'].append(i)
    result['delta_yaw'].append(delta_yaw / i)


if __name__ == '__main__':
    from gym_carla_feature.start_env.config import params

    params['port'] = 2120
    params['number_of_vehicles'] = 0
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    params['continuous_accel_range'] = 1.
    params['continuous_steer_range'] = 1.
    params['desired_speed'] = 20
    params['max_waypt'] = 12
    params['out_lane_thres'] = 3
    params['sampling_radius'] = 3
    params['discrete_steer'] = [-1.0, -0.8, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4,
                                0.8, 1.]
    params['action_type'] = 1
    params['reward_type'] = 4
    params['render'] = True
    params['autopilot'] = False
    params['town'] = 'Town07'
    params['task_mode'] = 'mountainroad'
    params['if_dest_end'] = False
    env = {
        'id': 'carla-v2',
        # 'state_dim': 307,
        # 'state_dim': 307 - 46,
        'state_dim': 45,
        'action_dim': 2,
        # 'action_dim': [1,11],
        'action_type': 1,
        'reward_dim': 2,
        'target_reward': 700,
        'max_step': 1000,
        'params_name': {'params': params}
    }
    from ray_elegantrl.configs.configs_ppo import config

    config['agent']['class_name'] = AgentSafePPO2
    config['agent']['ratio_clip'] = 0.25
    config['agent']['lambda_entropy'] = 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    # config['agent']['objective_type'] = 'kl'
    config['agent']['beta'] = 3.
    # config['agent']['policy_type'] = 'g'
    config['agent']['net_dim'] = 2 ** 8
    agent = config['agent']['class_name'](config['agent'])
    device = torch.device("cpu")
    agent.init(config['agent']['net_dim'], env['state_dim'], env['action_dim'], env['reward_dim'], device=device)
    PATH = '/home/zgy/repos/ray_elegantrl/gym_carla_feature/start_env/demo/logs/carla-v2_Town07_mountainroad_s45_a2_r2_tr700_ms1000/AgentSafePPO2_None_clip/exp_2021-07-11-14-04-06_cuda:0'
    agent.load_model(PATH)
    eval_policy(agent.act, env, seed=0)
