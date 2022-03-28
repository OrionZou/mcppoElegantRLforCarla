import sys

print(sys.path.append('/home/zgy/repos/ray_elegantrl'))

import ray
from ray_elegantrl.interaction import beginer
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

MAX_WAYPT = 12
# MAX_STEP = 1000
# NUM_EPISODE = 1
MAX_STEP = 200
NUM_EPISODE = 5
RENDER = False
AUTOPILOT = False
DESIRED_V = 12
OUTLANE = 3
SAMPLING_RADIUS = 3
OBS_SPAVE_TYPE = ['orgin_state', 'waypoint']
TOWN = 'Town07'
TASK_MODE = 'mountainroad'
# TOWN = 'Town03'
# TASK_MODE = 'urbanroad'
REWARD_TYPE = 12
IF_DEST_END = False

ENV_ID = "carla-v2"
STATE_DIM = 50
ACTION_DIM = 2
REWARD_DIM = 1
TARGET_RETURN = 200


def demo_menv_exp_ppo_feature(args=None):
    from ray_elegantrl.agent import AgentPPO2, AgentRNNPPO2, AgentPPO2CMAES
    from ray_elegantrl.configs.configs_ppo import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = args.number_of_vehicles if hasattr(args, 'number_of_vehicles') else 0
    port = args.port[0] if hasattr(args, 'port') and len(args.port) == 1 else 2000
    params['port'] = args.port if hasattr(args, 'port') and len(args.port) > 1 \
        else [port, port + 4, port + 8, port + 12]
    params['max_waypt'] = MAX_WAYPT
    params['max_step'] = MAX_STEP
    params['render'] = RENDER
    params['autopilot'] = AUTOPILOT
    params['desired_speed'] = DESIRED_V
    params['out_lane_thres'] = OUTLANE
    params['sampling_radius'] = SAMPLING_RADIUS
    params['obs_space_type'] = OBS_SPAVE_TYPE
    params['town'] = args.town if hasattr(args, 'town') else TOWN
    params['task_mode'] = args.task_mode if hasattr(args, 'task_mode') else TASK_MODE
    params['reward_type'] = args.reward_type if hasattr(args, 'reward_type') else REWARD_TYPE
    params['if_dest_end'] = args.if_dest_end if hasattr(args, 'if_dest_end') else IF_DEST_END
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'action_type': params['action_type'],
        'reward_dim': REWARD_DIM,
        'target_return': TARGET_RETURN,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    if args.demo_type == 'rnnppo2':
        config['agent']['class_name'] = AgentRNNPPO2
    elif args.demo_type == 'ppo2cmaes':
        config['agent']['class_name'] = AgentPPO2CMAES
    else:
        config['agent']['class_name'] = AgentPPO2
    config['agent']['lambda_entropy'] = args.lambda_entropy if hasattr(args, 'lambda_entropy') else 0.01
    config['agent']['lambda_gae_adv'] = args.lambda_gae_adv if hasattr(args, 'lambda_gae_adv') else 0.97
    config['agent']['objective_type'] = args.objective_type if hasattr(args, 'objective_type') else 'clip'
    config['agent']['sp_a_num'] = args.sp_a_num if hasattr(args, 'sp_a_num') else [3, 3]
    config['agent']['if_ir'] = args.if_ir if hasattr(args, 'if_ir') else False
    config['agent']['beta'] = args.beta if hasattr(args, 'beta') else 3.
    config['agent']['ratio_clip'] = args.ratio_clip if hasattr(args, 'ratio_clip') else 0.25
    config['agent']['policy_type'] = args.policy_type if hasattr(args, 'policy_type') else None
    config['agent']['net_dim'] = 2 ** 8
    config['agent']['if_rnn'] = args.if_rnn if hasattr(args, 'if_rnn') else False
    config['agent']['infer_by_sequence'] = args.infer_by_sequence if hasattr(args, 'infer_by_sequence') else False
    config['agent']['if_store_state'] = (not args.if_zero_state) if hasattr(args, 'if_zero_state') else True
    config['agent']['hidden_state_dim'] = args.hidden_state_dim if hasattr(args, 'hidden_state_dim') else 256
    config['interactor']['rollout_num'] = args.rollout_num if hasattr(args, 'rollout_num') else 4
    config['trainer']['batch_size'] = args.batch_size if hasattr(args, 'batch_size') else 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = args.gamma if hasattr(args, 'gamma') else 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num'] * NUM_EPISODE
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['evaluator']['eval_gap'] = args.eval_gap if hasattr(args, 'eval_gap') else 10000
    config['evaluator']['break_step'] = args.break_step if hasattr(args, 'break_step') else 2e6
    config['env'] = env
    config['gpu_id'] = args.gpu_id if hasattr(args, 'gpu_id') else 0
    config['if_cwd_time'] = args.if_cwd_time if hasattr(args, 'if_cwd_time') else True
    config['random_seed'] = args.random_seed if hasattr(args, 'random_seed') else 0
    beginer(config)


def demo_menv_exp_contriantedppo_feature(args=None):
    from ray_elegantrl.agent import AgentConstriantPPO2
    from ray_elegantrl.configs.configs_constrianedppo import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = args.number_of_vehicles if hasattr(args, 'number_of_vehicles') else 0
    port = args.port[0] if hasattr(args, 'port') and len(args.port) == 1 else 2000
    params['port'] = args.port if hasattr(args, 'port') and len(args.port) > 1 \
        else [port, port + 4, port + 8, port + 12]
    params['max_waypt'] = MAX_WAYPT
    params['max_step'] = MAX_STEP
    params['render'] = RENDER
    params['autopilot'] = AUTOPILOT
    params['desired_speed'] = DESIRED_V
    params['out_lane_thres'] = OUTLANE
    params['sampling_radius'] = SAMPLING_RADIUS
    params['obs_space_type'] = OBS_SPAVE_TYPE
    params['town'] = args.town if hasattr(args, 'town') else TOWN
    params['task_mode'] = args.task_mode if hasattr(args, 'task_mode') else TASK_MODE
    params['reward_type'] = args.reward_type if hasattr(args, 'reward_type') else 14
    params['if_dest_end'] = args.if_dest_end if hasattr(args, 'if_dest_end') else IF_DEST_END
    config['agent']['cost_threshold'] = args.cost_threshold if hasattr(args, 'cost_threshold') else [0.2, 0.01]
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'action_type': params['action_type'],
        'reward_dim': len(config['agent']['cost_threshold']) + 1,
        'target_return': TARGET_RETURN,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['class_name'] = AgentConstriantPPO2
    config['agent']['if_critic_shared'] = args.if_critic_shared if hasattr(args, 'if_critic_shared') else True
    config['agent']['if_auto_weights'] = args.if_auto_weights if hasattr(args, 'if_auto_weights') else True
    config['agent']['weights'] = args.weights if hasattr(args, 'weights') else [1.] + [0.] * len(
        config['agent']['cost_threshold'])
    config['agent']['cost_threshold'] = args.cost_threshold if hasattr(args, 'cost_threshold') else [0.2, 0.01]
    config['agent']['pid_Ki'] = 0.01  # [0.01, 0.01]
    config['agent']['pid_Kp'] = 0.25  # [0.25, 0.25]
    config['agent']['pid_Kd'] = 4.  # [4, 4]
    config['agent']['lambda_entropy'] = args.lambda_entropy if hasattr(args, 'lambda_entropy') else 0.01
    config['agent']['lambda_gae_adv'] = args.lambda_gae_adv if hasattr(args, 'lambda_gae_adv') else 0.97
    config['agent']['objective_type'] = args.objective_type if hasattr(args, 'objective_type') else 'clip'

    config['agent']['beta'] = args.beta if hasattr(args, 'beta') else 3.
    config['agent']['ratio_clip'] = args.ratio_clip if hasattr(args, 'ratio_clip') else 0.25
    config['agent']['policy_type'] = args.policy_type if hasattr(args, 'policy_type') else None
    config['agent']['net_dim'] = 2 ** 8
    config['agent']['if_rnn'] = args.if_rnn if hasattr(args, 'if_rnn') else False
    config['agent']['infer_by_sequence'] = args.infer_by_sequence if hasattr(args, 'infer_by_sequence') else False
    config['agent']['if_store_state'] = (not args.if_zero_state) if hasattr(args, 'if_zero_state') else True
    config['agent']['hidden_state_dim'] = args.hidden_state_dim if hasattr(args, 'hidden_state_dim') else 256
    config['interactor']['rollout_num'] = args.rollout_num if hasattr(args, 'rollout_num') else 4
    config['interactor']['env_horizon'] = args.env_horizon if hasattr(args, 'env_horizon') else 200
    config['trainer']['batch_size'] = args.batch_size if hasattr(args, 'batch_size') else 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = args.gamma if hasattr(args, 'gamma') else [0.99] + [0.95] * len(
        config['agent']['cost_threshold'])
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num'] * NUM_EPISODE
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['evaluator']['eval_gap'] = args.eval_gap if hasattr(args, 'eval_gap') else 10000
    config['evaluator']['break_step'] = args.break_step if hasattr(args, 'break_step') else 2e6
    config['env'] = env
    config['gpu_id'] = args.gpu_id if hasattr(args, 'gpu_id') else 0
    config['if_cwd_time'] = args.if_cwd_time if hasattr(args, 'if_cwd_time') else True
    config['random_seed'] = args.random_seed if hasattr(args, 'random_seed') else 0
    beginer(config)


def demo_menv_exp_discreteppo_feature(args=None):
    from ray_elegantrl.agent import AgentPPO2
    from ray_elegantrl.configs.configs_ppo import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = args.number_of_vehicles if hasattr(args, 'number_of_vehicles') else 0
    port = args.port[0] if hasattr(args, 'port') and len(args.port) == 1 else 2000
    params['port'] = args.port if hasattr(args, 'port') and len(args.port) > 1 \
        else [port, port + 4, port + 8, port + 12]
    params['max_waypt'] = MAX_WAYPT
    params['max_step'] = MAX_STEP
    params['render'] = RENDER
    params['autopilot'] = AUTOPILOT
    params['desired_speed'] = DESIRED_V
    params['out_lane_thres'] = OUTLANE
    params['sampling_radius'] = SAMPLING_RADIUS
    params['obs_space_type'] = OBS_SPAVE_TYPE
    params['town'] = args.town if hasattr(args, 'town') else TOWN
    params['task_mode'] = args.task_mode if hasattr(args, 'task_mode') else TASK_MODE
    params['reward_type'] = args.reward_type if hasattr(args, 'reward_type') else REWARD_TYPE
    params['if_dest_end'] = args.if_dest_end if hasattr(args, 'if_dest_end') else IF_DEST_END

    params['obs_space_type'] = ['orgin_state', 'waypoint']
    params['action_type'] = -1
    params['discrete_acc'] = [-1, 0.0, 1]
    params['discrete_steer'] = args.discrete_steer if hasattr(args, 'discrete_steer') and len(
        args.discrete_steer) > 1 else [-0.2, 0.0, 0.2]
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    ACTION_DIM = len(params['discrete_acc']) * len(params['discrete_steer'])
    env = {
        'id': 'carla-v2',
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'action_type': params['action_type'],
        'reward_dim': REWARD_DIM,
        'target_return': TARGET_RETURN,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['class_name'] = AgentPPO2
    config['agent']['lambda_entropy'] = args.lambda_entropy if hasattr(args, 'lambda_entropy') else 0.01
    config['agent']['lambda_gae_adv'] = args.lambda_gae_adv if hasattr(args, 'lambda_gae_adv') else 0.97
    config['agent']['objective_type'] = args.objective_type if hasattr(args, 'objective_type') else 'clip'
    config['agent']['if_ir'] = args.if_ir if hasattr(args, 'if_ir') else False
    config['agent']['beta'] = args.beta if hasattr(args, 'beta') else 3.
    config['agent']['ratio_clip'] = args.ratio_clip if hasattr(args, 'ratio_clip') else 0.25
    config['agent']['policy_type'] = 'discrete'
    config['agent']['net_dim'] = 2 ** 8
    config['agent']['if_rnn'] = args.if_rnn if hasattr(args, 'if_rnn') else False
    config['agent']['if_store_state'] = (not args.if_zero_state) if hasattr(args, 'if_zero_state') else True
    config['agent']['hidden_state_dim'] = args.hidden_state_dim if hasattr(args, 'hidden_state_dim') else 256
    config['interactor']['rollout_num'] = args.rollout_num if hasattr(args, 'rollout_num') else 4
    config['trainer']['batch_size'] = args.batch_size if hasattr(args, 'batch_size') else 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = args.gamma if hasattr(args, 'gamma') else 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num'] * NUM_EPISODE
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['evaluator']['eval_gap'] = args.eval_gap if hasattr(args, 'eval_gap') else 10000
    config['evaluator']['break_step'] = args.break_step if hasattr(args, 'break_step') else 2e6
    config['env'] = env
    config['gpu_id'] = args.gpu_id if hasattr(args, 'gpu_id') else 0
    config['if_cwd_time'] = args.if_cwd_time if hasattr(args, 'if_cwd_time') else True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_hybridppo2_feature(args=None):
    from ray_elegantrl.agent import AgentHybridPPO2
    from ray_elegantrl.configs.configs_ppo import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = args.number_of_vehicles if hasattr(args, 'number_of_vehicles') else 0
    port = args.port[0] if hasattr(args, 'port') and len(args.port) == 1 else 2000
    params['port'] = args.port if hasattr(args, 'port') and len(args.port) > 1 \
        else [port, port + 4, port + 8, port + 12]
    params['max_waypt'] = MAX_WAYPT
    params['max_step'] = MAX_STEP
    params['render'] = RENDER
    params['autopilot'] = AUTOPILOT
    params['desired_speed'] = DESIRED_V
    params['out_lane_thres'] = OUTLANE
    params['sampling_radius'] = SAMPLING_RADIUS
    params['obs_space_type'] = OBS_SPAVE_TYPE
    params['town'] = args.town if hasattr(args, 'town') else TOWN
    params['task_mode'] = args.task_mode if hasattr(args, 'task_mode') else TASK_MODE
    params['reward_type'] = args.reward_type if hasattr(args, 'reward_type') else REWARD_TYPE
    params['if_dest_end'] = args.if_dest_end if hasattr(args, 'if_dest_end') else IF_DEST_END
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM + 1,
        'action_type': params['action_type'],
        'reward_dim': REWARD_DIM,
        'target_return': TARGET_RETURN,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['class_name'] = AgentHybridPPO2
    config['agent']['lambda_entropy'] = args.lambda_entropy if hasattr(args, 'lambda_entropy') else 0.01
    config['agent']['lambda_gae_adv'] = args.lambda_gae_adv if hasattr(args, 'lambda_gae_adv') else 0.97
    config['agent']['objective_type'] = args.objective_type if hasattr(args, 'objective_type') else 'clip'
    config['agent']['beta'] = args.beta if hasattr(args, 'beta') else 3.
    config['agent']['ratio_clip'] = args.ratio_clip if hasattr(args, 'ratio_clip') else 0.25
    config['agent']['policy_type'] = args.policy_type if hasattr(args, 'policy_type') else None
    config['agent']['discrete_degree'] = args.discrete_degree if hasattr(args, 'discrete_degree') else 3
    config['agent']['if_share'] = args.if_rnn if hasattr(args, 'if_share') else True
    config['agent']['if_sp_action_loss'] = args.if_sp_action_loss if hasattr(args, 'if_sp_action_loss') else False
    config['agent']['net_dim'] = 2 ** 8
    config['agent']['if_rnn'] = args.if_rnn if hasattr(args, 'if_rnn') else False
    config['agent']['if_store_state'] = (not args.if_zero_state) if hasattr(args, 'if_zero_state') else True
    config['agent']['hidden_state_dim'] = args.hidden_state_dim if hasattr(args, 'hidden_state_dim') else 256
    config['interactor']['rollout_num'] = args.rollout_num if hasattr(args, 'rollout_num') else 4
    config['trainer']['batch_size'] = args.batch_size if hasattr(args, 'batch_size') else 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = args.gamma if hasattr(args, 'gamma') else 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num'] * NUM_EPISODE
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['evaluator']['eval_gap'] = args.eval_gap if hasattr(args, 'eval_gap') else 10000
    config['evaluator']['break_step'] = args.break_step if hasattr(args, 'break_step') else 2e6
    config['env'] = env
    config['gpu_id'] = args.gpu_id if hasattr(args, 'gpu_id') else 0
    config['if_cwd_time'] = args.if_cwd_time if hasattr(args, 'if_cwd_time') else True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_hierarchicalppo2_feature(args=None):
    from ray_elegantrl.agent import AgentHierarchicalPPO2, AgentHCPPO2
    from ray_elegantrl.configs.configs_ppo import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = args.number_of_vehicles if hasattr(args, 'number_of_vehicles') else 0
    port = args.port[0] if hasattr(args, 'port') and len(args.port) == 1 else 2000
    params['port'] = args.port if hasattr(args, 'port') and len(args.port) > 1 \
        else [port, port + 4, port + 8, port + 12]
    params['max_waypt'] = MAX_WAYPT
    params['max_step'] = MAX_STEP
    params['render'] = RENDER
    params['autopilot'] = AUTOPILOT
    params['desired_speed'] = DESIRED_V
    params['out_lane_thres'] = OUTLANE
    params['sampling_radius'] = SAMPLING_RADIUS
    params['obs_space_type'] = OBS_SPAVE_TYPE
    params['town'] = args.town if hasattr(args, 'town') else TOWN
    params['task_mode'] = args.task_mode if hasattr(args, 'task_mode') else TASK_MODE
    params['reward_type'] = args.reward_type if hasattr(args, 'reward_type') else REWARD_TYPE
    params['if_dest_end'] = args.if_dest_end if hasattr(args, 'if_dest_end') else IF_DEST_END
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM + 1,
        'action_type': params['action_type'],
        'reward_dim': REWARD_DIM,
        'target_return': TARGET_RETURN,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    if args.demo_type in ['hcppo2']:
        config['agent']['class_name'] = AgentHCPPO2
    else:
        config['agent']['class_name'] = AgentHierarchicalPPO2
    config['agent']['lambda_entropy'] = args.lambda_entropy if hasattr(args, 'lambda_entropy') else 0.01
    config['agent']['lambda_gae_adv'] = args.lambda_gae_adv if hasattr(args, 'lambda_gae_adv') else 0.97
    config['agent']['objective_type'] = args.objective_type if hasattr(args, 'objective_type') else 'clip'
    config['agent']['beta'] = args.beta if hasattr(args, 'beta') else 3.
    config['agent']['ratio_clip'] = args.ratio_clip if hasattr(args, 'ratio_clip') else 0.25
    config['agent']['policy_type'] = args.policy_type if hasattr(args, 'policy_type') else None
    config['agent']['discrete_degree'] = args.discrete_degree if hasattr(args, 'discrete_degree') else 3
    config['agent']['train_model'] = args.train_model if hasattr(args, 'train_model') else 'discrete'
    config['agent']['save_path'] = args.hppo_save_path if hasattr(args, 'hppo_save_path') else None
    config['agent']['net_dim'] = 2 ** 8
    config['agent']['if_rnn'] = args.if_rnn if hasattr(args, 'if_rnn') else False
    config['agent']['if_store_state'] = (not args.if_zero_state) if hasattr(args, 'if_zero_state') else True
    config['agent']['hidden_state_dim'] = args.hidden_state_dim if hasattr(args, 'hidden_state_dim') else 256
    config['interactor']['rollout_num'] = args.rollout_num if hasattr(args, 'rollout_num') else 4
    config['trainer']['batch_size'] = args.batch_size if hasattr(args, 'batch_size') else 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = args.gamma if hasattr(args, 'gamma') else 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num'] * NUM_EPISODE
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['evaluator']['eval_gap'] = args.eval_gap if hasattr(args, 'eval_gap') else 10000
    config['evaluator']['break_step'] = args.break_step if hasattr(args, 'break_step') else 2e6
    config['env'] = env
    config['gpu_id'] = args.gpu_id if hasattr(args, 'gpu_id') else 0
    config['if_cwd_time'] = args.if_cwd_time if hasattr(args, 'if_cwd_time') else True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_sac_feature(args=None):
    from ray_elegantrl.configs.configs_modsac import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = args.number_of_vehicles if hasattr(args, 'number_of_vehicles') else 0
    port = args.port[0] if hasattr(args, 'port') and len(args.port) == 1 else 2000
    params['port'] = args.port if hasattr(args, 'port') and len(args.port) > 1 \
        else [port, port + 4, port + 8, port + 12]
    params['max_waypt'] = MAX_WAYPT
    params['max_step'] = MAX_STEP
    params['render'] = RENDER
    params['autopilot'] = AUTOPILOT
    params['desired_speed'] = DESIRED_V
    params['out_lane_thres'] = OUTLANE
    params['sampling_radius'] = SAMPLING_RADIUS
    params['obs_space_type'] = OBS_SPAVE_TYPE
    params['town'] = args.town if hasattr(args, 'town') else TOWN
    params['task_mode'] = args.task_mode if hasattr(args, 'task_mode') else TASK_MODE
    params['reward_type'] = args.reward_type if hasattr(args, 'reward_type') else REWARD_TYPE
    params['if_dest_end'] = args.if_dest_end if hasattr(args, 'if_dest_end') else IF_DEST_END
    # params['town'] = 'Town03'
    # params['task_mode'] = 'fixedroute'
    params['obs_space_type'] = ['orgin_state', 'waypoint']
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    env = {
        'id': 'carla-v2',
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'action_type': params['action_type'],
        'reward_dim': REWARD_DIM,
        'target_return': TARGET_RETURN,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['net_dim'] = 2 ** 8
    config['agent']['policy_type'] = args.policy_type if hasattr(args, 'policy_type') else None
    config['agent']['learning_rate'] = args.learning_rate if hasattr(args, 'learning_rate') else 1e-4
    config['interactor']['rollout_num'] = args.rollout_num if hasattr(args, 'rollout_num') else 4
    config['trainer']['batch_size'] = args.batch_size if hasattr(args, 'batch_size') else 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 1
    config['interactor']['gamma'] = 0.99
    config['interactor']['reward_scale'] = 2 ** args.reward_scale if hasattr(args, 'reward_scale') else 1

    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num'] * NUM_EPISODE
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['buffer']['max_buf'] = args.max_buf if hasattr(args, 'max_buf') else 2 ** 20
    config['evaluator']['eval_gap'] = args.eval_gap if hasattr(args, 'eval_gap') else 10000
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['env'] = env
    config['gpu_id'] = args.gpu_id if hasattr(args, 'gpu_id') else 0
    config['if_cwd_time'] = args.if_cwd_time if hasattr(args, 'if_cwd_time') else True
    config['random_seed'] = 0
    beginer(config)


def demo_menv_exp_d3qn_feature(args=None):
    from ray_elegantrl.configs.configs_d3qn import config
    from gym_carla_feature.start_env.config import params
    params['number_of_vehicles'] = args.number_of_vehicles if hasattr(args, 'number_of_vehicles') else 0
    port = args.port[0] if hasattr(args, 'port') and len(args.port) == 1 else 2000
    params['port'] = args.port if hasattr(args, 'port') and len(args.port) > 1 \
        else [port, port + 4, port + 8, port + 12]
    params['max_waypt'] = MAX_WAYPT
    params['max_step'] = MAX_STEP
    params['render'] = RENDER
    params['autopilot'] = AUTOPILOT
    params['desired_speed'] = DESIRED_V
    params['out_lane_thres'] = OUTLANE
    params['sampling_radius'] = SAMPLING_RADIUS
    params['obs_space_type'] = OBS_SPAVE_TYPE
    params['town'] = args.town if hasattr(args, 'town') else TOWN
    params['task_mode'] = args.task_mode if hasattr(args, 'task_mode') else TASK_MODE
    params['reward_type'] = args.reward_type if hasattr(args, 'reward_type') else REWARD_TYPE
    params['if_dest_end'] = args.if_dest_end if hasattr(args, 'if_dest_end') else IF_DEST_END

    params['obs_space_type'] = ['orgin_state', 'waypoint']
    params['action_type'] = -1
    params['discrete_acc'] = [-1, 0.0, 1]
    params['discrete_steer'] = args.discrete_steer if hasattr(args, 'discrete_steer') and len(
        args.discrete_steer) > 1 else [-0.2, 0.0, 0.2]
    from gym_carla_feature.start_env.misc import write_yaml
    write_yaml(params)
    ACTION_DIM = len(params['discrete_acc']) * len(params['discrete_steer'])
    env = {
        'id': 'carla-v2',
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'action_type': params['action_type'],
        'reward_dim': REWARD_DIM,
        'target_return': TARGET_RETURN,
        'max_step': params['max_step'],
        'params_name': {'params': params}
    }
    config['agent']['net_dim'] = 2 ** 8
    config['agent']['policy_type'] = args.policy_type if hasattr(args, 'policy_type') else None
    config['interactor']['rollout_num'] = args.rollout_num if hasattr(args, 'rollout_num') else 4
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 0
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num'] * NUM_EPISODE
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['buffer']['max_buf'] = 2 ** 20
    config['evaluator']['eval_gap'] = args.eval_gap if hasattr(args, 'eval_gap') else 10000
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['env'] = env
    config['gpu_id'] = args.gpu_id if hasattr(args, 'gpu_id') else 0
    config['if_cwd_time'] = args.if_cwd_time if hasattr(args, 'if_cwd_time') else True
    config['random_seed'] = 0
    beginer(config)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Carla RL')
    parser.add_argument('--debug', default=False, action="store_true", )
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--demo_type', type=str, default='ppo2')
    parser.add_argument('--port', nargs='+', type=int, default=2000)
    parser.add_argument('--reward_type', type=int, default=1)
    parser.add_argument('--policy_type', type=str, default=None)
    parser.add_argument('--objective_type', type=str, default='clip')
    parser.add_argument('--hppo_save_path', type=str, default=None)
    parser.add_argument('--train_model', type=str, default='discrete')  # mix discrete continues
    parser.add_argument('--if_ir', default=False, action="store_true", )
    parser.add_argument('--if_rnn', default=False, action="store_true", )
    parser.add_argument('--infer_by_sequence', default=False, action="store_true", )
    parser.add_argument('--if_zero_state', default=False, action="store_true", )
    parser.add_argument('--if_critic_shared', default=False, action="store_true", )
    parser.add_argument('--if_share', default=True, action="store_false", )
    parser.add_argument('--hidden_state_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=3.)
    parser.add_argument('--lambda_entropy', type=float, default=0.01)
    parser.add_argument('--ratio_clip', type=float, default=0.25)
    parser.add_argument('--lambda_gae_adv', type=float, default=0.97)
    parser.add_argument('--if_dest_end', default=False, action="store_true", )
    parser.add_argument('--if_sp_action_loss', default=False, action="store_true", )
    parser.add_argument('--discrete_steer', nargs='+', type=float, default=[-0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6])
    parser.add_argument('--cost_threshold', nargs='+', type=float, default=[0.1, 0.01])
    parser.add_argument('--gamma', nargs='+', type=float, default=0.99)
    parser.add_argument('--weights', nargs='+', type=float, default=[1, 0, 0])
    parser.add_argument('--sp_a_num', nargs='+', type=float, default=[3, 11])
    parser.add_argument('--discrete_degree', type=int, default=3)
    parser.add_argument('--gpu_id', type=int, default=1)
    # parser.add_argument('--eval_gap', type=int, default=0)
    parser.add_argument('--eval_gap', type=int, default=20000)
    parser.add_argument('--reward_scale', type=float, default=1.)
    parser.add_argument('--rollout_num', type=int, default=4)
    parser.add_argument('--max_buf', type=int, default=20)
    parser.add_argument('--break_step', type=int, default=2000000)
    # parser.add_argument('--has', type=int, default=0)
    args = parser.parse_args()
    ray.init(local_mode=True) if args.debug else ray.init()
    if args.demo_type in ['ppo2', 'ppo2cmaes']:
        demo_menv_exp_ppo_feature(args)
    elif args.demo_type == 'discreteppo2':
        demo_menv_exp_discreteppo_feature(args)
    elif args.demo_type == 'ppo2-rnd':
        args.demo_type = 'ppo2'
        args.if_ir = True
        demo_menv_exp_ppo_feature(args)
    elif args.demo_type == 'rnnppo2':
        args.if_rnn = True
        demo_menv_exp_ppo_feature(args)
    elif args.demo_type == 'sac':
        args.max_buf = 2 ** args.max_buf
        demo_menv_exp_sac_feature(args)
    elif args.demo_type == 'd3qn':
        demo_menv_exp_d3qn_feature(args)
    elif args.demo_type == 'hybridppo2':
        demo_menv_exp_hybridppo2_feature(args)
    elif args.demo_type in ['hierarchicalppo2', 'hcppo2']:
        demo_menv_exp_hierarchicalppo2_feature(args)
    elif args.demo_type == 'cppo':
        demo_menv_exp_contriantedppo_feature(args)
