import sys
print(sys.path.append('/home/zgy/repos/ray_elegantrl'))

import ray
from ray_elegantrl.interaction import beginer
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"


def demo_senv_exp_ppo_feature():
    from ray_elegantrl.configs.configs_ppo import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = 0
    params['port'] = 2000
    params['max_step'] = 1000
    params['max_waypt'] = 12
    params['render'] = False
    params['autopilot'] = False
    params['town'] = 'Town07'
    params['task_mode'] = 'mountain_road'
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    # params['obs_space_type'] = ['orgin_state', 'waypoint', 'othervehs']
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)

    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        # 'state_dim': 70,
        'action_dim': 2,
        'action_type': params['action_type'],
        'reward_dim': 1,
        'target_reward': 700,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['ratio_clip'] = 0.25
    config['agent']['lambda_entropy'] = 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    # config['agent']['policy_type'] = 'mg'
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 1
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = 4 * env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 2
    config['evaluator']['eval_times'] = 6
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo_senv_exp_ppomg_feature():
    from ray_elegantrl.configs.configs_ppo import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = 0
    params['port'] = 2000
    params['max_step'] = 1000
    params['max_waypt'] = 12
    params['render'] = False
    params['autopilot'] = False
    params['town'] = 'Town07'
    params['task_mode'] = 'mountain_road'
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        'action_dim': 2,
        'action_type': params['action_type'],
        'reward_dim': 1,
        'target_reward': 1400,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['ratio_clip'] = 0.25
    config['agent']['lambda_entropy'] = 0.1
    config['agent']['lambda_gae_adv'] = 0.97
    config['agent']['policy_type'] = 'mg'
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 1
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = 4 * env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 2
    config['evaluator']['eval_times'] = 6
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo_senv_exp_sac_feature():
    from ray_elegantrl.configs.configs_modsac import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = 0
    params['port'] = 2000
    params['max_waypt'] = 12
    params['max_step'] = 1000
    params['render'] = False
    params['autopilot'] = False
    params['task_mode'] = 'fixed_route'
    params['desired_speed'] = 20
    params['out_lane_thres'] = 3
    params['sampling_radius'] = 3
    params['obs_space_type'] = ['orgin_state', 'waypoint', ]
    # params['obs_space_type'] = ['orgin_state', 'waypoint','othervehs']
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        # 'state_dim': 70,
        'action_dim': 2,
        'action_type': params['action_type'],
        'reward_dim': 1,
        'target_reward': 700,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 1
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 0
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['buffer']['max_buf'] = 2 ** 20
    config['evaluator']['pre_eval_times'] = 4
    config['evaluator']['eval_times'] = 8
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo_senv_exp_hsac_feature():
    from ray_elegantrl.agent import AgentHybrid2SAC, AgentHybridSAC
    from ray_elegantrl.configs.configs_modsac import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = 0
    params['action_type'] = 0
    params['port'] = 2000
    params['max_waypt'] = 12
    params['max_step'] = 1000
    params['render'] = False
    params['autopilot'] = False
    params['task_mode'] = 'fixed_route'
    params['desired_speed'] = 20
    params['out_lane_thres'] = 3
    params['sampling_radius'] = 3
    params['obs_space_type'] = ['orgin_state', 'waypoint', ]
    # params['obs_space_type'] = ['orgin_state', 'waypoint','othervehs']
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        # 'state_dim': 70,
        'action_dim': [1, 11],
        'action_type': params['action_type'],
        # choose -1 discrete action space | 1 continuous action space | 0 hybird action space |
        'reward_dim': 1,
        'target_reward': 700,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['class_name'] = AgentHybrid2SAC
    # config['agent']['class_name'] = AgentHybridSAC
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 1
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 0
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['buffer']['max_buf'] = 2 ** 20
    config['evaluator']['pre_eval_times'] = 4
    config['evaluator']['eval_times'] = 8
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_ppo_feature(args=None):
    from ray_elegantrl.agent import AgentPPO2
    from ray_elegantrl.configs.configs_ppo import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = args.number_of_vehicles if hasattr(args, 'number_of_vehicles') else 0
    port = args.port[0] if hasattr(args, 'port') and len(args.port) == 1 else 2000
    params['port'] = args.port if hasattr(args, 'port') and len(args.port) > 1 \
        else [port, port + 4, port + 8, port + 12]
    params['max_waypt'] = 12
    params['max_step'] = 1000
    params['render'] = False
    params['autopilot'] = False
    params['desired_speed'] = 20
    params['out_lane_thres'] = 3
    params['sampling_radius'] = 3
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    params['town'] = args.town if hasattr(args, 'town') else 'Town07'
    params['task_mode'] = args.task_mode if hasattr(args, 'task_mode') else 'mountainroad'
    params['reward_type'] = args.reward_type if hasattr(args, 'reward_type') else 1
    params['if_dest_end'] = args.if_dest_end if hasattr(args, 'if_dest_end') else False
    # params['town'] = 'Town03'
    # params['task_mode'] = 'fixedroute'
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        'action_dim': 2,
        'action_type': params['action_type'],
        'reward_dim': 1,
        'target_reward': 700,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['class_name'] = AgentPPO2
    config['agent']['lambda_entropy'] = args.lambda_entropy if hasattr(args, 'lambda_entropy') else 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    config['agent']['objective_type'] = args.objective_type if hasattr(args, 'objective_type') else 'clip'
    config['agent']['beta'] = config.beta if hasattr(args, 'beta') else 3.
    config['agent']['ratio_clip'] = config.ratio_clip if hasattr(args, 'ratio_clip') else 0.25
    config['agent']['policy_type'] = args.policy_type if hasattr(args, 'policy_type') else None
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 4
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['evaluator']['break_step']=args.break_step if hasattr(args, 'break_step') else 2e6
    config['env'] = env
    config['gpu_id'] = args.gpu_id if hasattr(args, 'gpu_id') else 0
    config['if_cwd_time'] = args.if_cwd_time if hasattr(args, 'if_cwd_time') else True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_mixppo_feature():
    from ray_elegantrl.agent import AgentMixPPO2
    from ray_elegantrl.configs.configs_ppo import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = 0
    # port = 2200
    # params['port'] = [port, port + 4, port + 8, port + 12]
    params['port'] = [2016, 2116, 2216, 2020]
    params['max_waypt'] = 12
    params['max_step'] = 1000
    params['render'] = False
    params['autopilot'] = False
    params['desired_speed'] = 20
    params['out_lane_thres'] = 3
    params['sampling_radius'] = 3
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    params['town'] = 'Town07'
    params['task_mode'] = 'mountainroad'
    params['reward_type'] = 1
    params['if_dest_end'] = False
    # params['town'] = 'Town03'
    # params['task_mode'] = 'fixedroute'

    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        'action_dim': 2,
        'action_type': params['action_type'],
        'reward_dim': 1,
        'target_reward': 700,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['class_name'] = AgentMixPPO2
    config['agent']['ratio_clip'] = 0.25
    config['agent']['lambda_entropy'] = 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    # config['agent']['objective_type'] = 'kl'
    config['agent']['beta'] = 3.
    config['agent']['policy_type'] = 'beta'
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 4
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_hppo_feature():
    from ray_elegantrl.agent import AgentHybridPPO
    from ray_elegantrl.configs.configs_ppo import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = 0
    port = 2100
    params['port'] = [port, port + 4, port + 8, port + 12]
    # params['port'] = [2016, 2116, 2216, 2020]
    params['max_waypt'] = 12
    params['max_step'] = 1000
    params['render'] = False
    params['autopilot'] = False
    params['desired_speed'] = 20
    params['out_lane_thres'] = 3
    params['sampling_radius'] = 3
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    params['discrete_steer'] = [-0.8, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4,
                                0.8]
    params['action_type'] = 0
    params['town'] = 'Town07'
    params['task_mode'] = 'mountainroad'
    params['reward_type'] = 3
    # params['town'] = 'Town03'
    # params['task_mode'] = 'fixedroute'
    params['if_dest_end'] = False
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        'action_dim': [1, len(params['discrete_steer'])],
        'action_type': params['action_type'],
        'reward_dim': 1,
        'target_reward': 700,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['class_name'] = AgentHybridPPO
    config['agent']['ratio_clip'] = 0.25
    config['agent']['lambda_entropy'] = 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    # config['agent']['objective_type'] = 'kl'
    config['agent']['beta'] = 3.
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 4
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_ppomo_feature():
    from ray_elegantrl.agent import AgentPPOMO2
    from ray_elegantrl.configs.configs_ppo import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = 0
    port = 2000
    params['port'] = [port, port + 4, port + 8, port + 12]
    # params['port'] = [2016, 2116, 2216, 2020]
    params['max_waypt'] = 12
    params['max_step'] = 1000
    params['render'] = False
    params['autopilot'] = False
    params['desired_speed'] = 20
    params['out_lane_thres'] = 3
    params['sampling_radius'] = 3
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    params['town'] = 'Town07'
    params['task_mode'] = 'mountainroad'
    params['reward_type'] = 4
    # params['town'] = 'Town03'
    # params['task_mode'] = 'fixedroute'
    params['if_dest_end'] = False
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        'action_dim': 2,
        'action_type': params['action_type'],
        'reward_dim': 2,
        'target_reward': 700,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['class_name'] = AgentPPOMO2
    config['agent']['ratio_clip'] = 0.25
    config['agent']['lambda_entropy'] = 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    config['agent']['if_auto_weights'] = True
    config['agent']['weights'] = [1., 1.]
    config['agent']['pid_Ki'] = 0.01
    config['agent']['pid_Kp'] = 1
    config['agent']['pid_Kd'] = 4
    # config['agent']['objective_type'] = 'kl'
    config['agent']['beta'] = 3.
    # config['agent']['policy_type'] = 'mg'
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 4
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = [0.99, 0.96]
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_ppors_feature():
    from ray_elegantrl.configs.configs_ppo import config
    from ray_elegantrl.agent import AgentPPO2RS
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = 0
    # port = 2200
    # params['port'] = [port, port + 4, port + 8, port + 12]
    params['port'] = [2016, 2116, 2216, 2020]
    params['max_waypt'] = 12
    params['max_step'] = 1000
    params['render'] = False
    params['autopilot'] = False
    params['desired_speed'] = 20
    params['out_lane_thres'] = 3
    params['sampling_radius'] = 3
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    params['town'] = 'Town07'
    params['task_mode'] = 'mountainroad'
    # params['town'] = 'Town03'
    # params['task_mode'] = 'fixedroute'
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        'action_dim': 2,
        'action_type': params['action_type'],
        'reward_dim': 2,
        'target_reward': 800,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['class_name'] = AgentPPO2RS
    config['agent']['ratio_clip'] = 0.25
    config['agent']['lambda_entropy'] = 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    # config['agent']['objective_type'] = 'kl'
    config['agent']['beta'] = 3.
    # config['agent']['policy_type'] = 'mg'
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 4
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = [0.99, 0.95]
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_pposafe_feature():
    from ray_elegantrl.configs.configs_ppo import config
    from ray_elegantrl.agent import AgentSafePPO2
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = 0
    port = 2200
    params['port'] = [port, port + 4, port + 8, port + 12]
    # params['port'] = [2016, 2116, 2216, 2020]
    params['max_waypt'] = 12
    params['max_step'] = 1000
    params['render'] = False
    params['autopilot'] = False
    params['desired_speed'] = 20
    params['out_lane_thres'] = 3
    params['sampling_radius'] = 3
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    params['town'] = 'Town07'
    params['task_mode'] = 'mountainroad'
    params['if_dest_end'] = False
    params['action_type'] = 1
    params['reward_type'] = 4
    # params['town'] = 'Town03'
    # params['task_mode'] = 'fixedroute'
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        'action_dim': 2,
        'action_type': params['action_type'],
        'reward_dim': 2,
        'target_reward': 700,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['class_name'] = AgentSafePPO2
    config['agent']['ratio_clip'] = 0.25
    config['agent']['lambda_entropy'] = 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    # config['agent']['objective_type'] = 'kl'
    config['agent']['beta'] = 3.
    # config['agent']['policy_type'] = 'mg'
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 4
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = [0.99, 0.95]
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_sac_feature(args=None):
    from ray_elegantrl.configs.configs_modsac import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = args.number_of_vehicles if hasattr(args, 'number_of_vehicles') else 0
    port = args.port[0] if hasattr(args, 'port') and len(args.port) == 1 else 2000
    params['port'] = args.port if hasattr(args, 'port') and len(args.port) > 1 \
        else [port, port + 4, port + 8, port + 12]
    params['max_waypt'] = 12
    params['max_step'] = 1000
    params['render'] = False
    params['autopilot'] = False
    params['desired_speed'] = 20
    params['out_lane_thres'] = 3
    params['sampling_radius'] = 3
    params['town'] = args.town if hasattr(args, 'town') else 'Town07'
    params['task_mode'] = args.task_mode if hasattr(args, 'task_mode') else 'mountainroad'
    params['reward_type'] = args.reward_type if hasattr(args, 'reward_type') else 1
    params['if_dest_end'] = args.if_dest_end if hasattr(args, 'if_dest_end') else False
    # params['town'] = 'Town03'
    # params['task_mode'] = 'fixedroute'
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        'action_dim': 2,
        'action_type': params['action_type'],
        'reward_dim': 1,
        'target_reward': 700,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['net_dim'] = 2 ** 8
    config['agent']['policy_type'] = args.policy_type if hasattr(args, 'policy_type') else None
    config['interactor']['rollout_num'] = 4
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 0
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['buffer']['max_buf'] = 2 ** 20
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = args.if_cwd_time if hasattr(args, 'if_cwd_time') else True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_sacrs_feature():
    from ray_elegantrl.agent import AgentSACRS
    from ray_elegantrl.configs.configs_modsac import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = 0
    port = 2200
    params['port'] = [port, port + 4, port + 8, port + 12]
    params['max_waypt'] = 12
    params['max_step'] = 1000
    params['render'] = False
    params['autopilot'] = False
    params['task_mode'] = 'fixed_route'
    params['desired_speed'] = 20
    params['out_lane_thres'] = 3
    params['sampling_radius'] = 3
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        'action_dim': 2,
        'action_type': params['action_type'],
        'reward_dim': 2,
        'target_reward': 800,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['class_name'] = AgentSACRS
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 4
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 0
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['buffer']['max_buf'] = 2 ** 20
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_ppomg_feature():
    from ray_elegantrl.agent import AgentPPO2
    from ray_elegantrl.configs.configs_ppo import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = 0
    port = 2200
    params['port'] = [port, port + 4, port + 8, port + 12]
    # params['port'] = [2216, 2116, 2016, 2020]
    # params['port'] = [2016, 2020, 2116, 2120]
    params['max_waypt'] = 12
    params['max_step'] = 1000
    params['render'] = False
    params['autopilot'] = False
    params['desired_speed'] = 20
    params['sampling_radius'] = 3
    params['out_lane_thres'] = 3
    params['town'] = 'Town07'
    params['task_mode'] = 'mountainroad'
    params['if_dest_end'] = False
    # params['town'] = 'Town03'
    # params['task_mode'] = 'fixedroute'
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': 45,
        'action_dim': 2,
        'action_type': params['action_type'],
        'reward_dim': 1,
        'target_reward': 700,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['class_name'] = AgentPPO2
    config['agent']['ratio_clip'] = 0.25
    config['agent']['lambda_entropy'] = 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    config['agent']['policy_type'] = 'mg'
    config['agent']['objective_type'] = 'kl'
    config['agent']['beta'] = 3.
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 4
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Carla RL')
    parser.add_argument('--demo_type', type=str, default='ppo2')
    parser.add_argument('--debug', default=False, action="store_true", )
    parser.add_argument('--port', nargs='+', type=int, default=2000)
    parser.add_argument('--reward_type', type=int, default=1)
    parser.add_argument('--policy_type', type=str, default=None)
    parser.add_argument('--objective_type', type=str, default='clip')
    parser.add_argument('--lambda_entropy', type=float, default=0.01)
    parser.add_argument('--break_step', type=int, default=2000000)
    args = parser.parse_args()
    ray.init(local_mode=True) if args.debug else ray.init()
    if args.demo_type == 'ppo2':
        demo_menv_exp_ppo_feature(args)
    elif args.demo_type == 'sac':
        demo_menv_exp_sac_feature(args)
    # demo_menv_exp_mixppo_feature()
    # demo_menv_exp_hppo_feature()
    # demo_menv_exp_ppomo_feature()
    # demo_menv_exp_ppors_feature()
    # demo_menv_exp_pposafe_feature()
    # demo_menv_exp_ppomg_feature()
    # demo_menv_exp_sac_feature()
    # demo_menv_exp_sacrs_feature()
