import sys
print(sys.path.append('/home/zgy/repos/ray_elegantrl'))
from ray_elegantrl.interaction import beginer
import ray


def demo_d3qn():
    from ray_elegantrl.configs.configs_d3qn import config
    env = {
        'id': 'LunarLander-v2',
        'state_dim': 8,
        'action_dim': 4,
        'if_discrete_action': True,
        'reward_dim': 1,
        'target_reward': 0,
        'max_step': 500,
    }

    config['interactor']['rollout_num'] = 2
    config['interactor']['reward_scale'] = 1.
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 1
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['interactor']['gamma'] = 0.99
    config['evaluator']['break_step'] = int(1e6)
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['buffer']['max_buf'] = 2 ** 20
    config['env'] = env
    config['gpu_id'] = 1
    config['if_cwd_time'] = False
    config['random_seed'] = 0
    beginer(config)


def demo_sac():
    from ray_elegantrl.configs.configs_modsac import config
    from ray_elegantrl.agent import AgentSAC
    env = {
        'id': 'Hopper-v2',
        'state_dim': 11,
        'action_dim': 3,
        'reward_dim': 1,
        'if_discrete_action': False,
        'target_reward': 0,
        'max_step': 1000,
    }
    # config['agent']['class_name']=AgentSAC
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 1
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 0
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['interactor']['reward_scale'] = 2 * 0
    config['buffer']['max_buf'] = 2 ** 20
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 2
    config['env'] = env
    config['gpu_id'] = 0
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo_ppocma():
    from ray_elegantrl.agent import AgentPPO2CMA
    from ray_elegantrl.configs.configs_ppo import config
    env = {
        'id': 'Hopper-v2',
        'state_dim': 11,
        'action_dim': 3,
        'reward_dim': 1,
        'action_type': 1,
        'target_reward': 3600,
        'max_step': 1000,
    }
    config['agent']['class_name'] = AgentPPO2CMA
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


def demo_mujoco_swimmer_ppo(args=None):
    from ray_elegantrl.agent import AgentPPO2, AgentPPO,AgentPPO2CMAES
    from ray_elegantrl.configs.configs_ppo import config

    import gym
    env = gym.make(args.env_id)
    state_dim = env.reset().shape[0]
    action_dim = env.action_space.sample().shape[0]
    env.close()

    env = {
        'id': args.env_id,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'action_type': 1,
        'reward_dim': 1,
        'max_step': 1000,
        'target_reward': 100,
    }
    if args.demo_type == 'ppo2':
        config['agent']['class_name'] = AgentPPO2
    elif args.demo_type == 'ppo':
        config['agent']['class_name'] = AgentPPO
    elif args.demo_type == 'ppo2cmaes':
        config['agent']['class_name'] = AgentPPO2CMAES
    else:
        config['agent']['class_name'] = AgentPPO2

    config['agent']['lambda_entropy'] = args.lambda_entropy if hasattr(args, 'lambda_entropy') else 0.01
    config['agent']['lambda_gae_adv'] = 0.97
    config['agent']['objective_type'] = args.objective_type if hasattr(args, 'objective_type') else 'clip'
    config['agent']['beta'] = config.beta if hasattr(args, 'beta') else 3.
    config['agent']['ratio_clip'] = config.ratio_clip if hasattr(args, 'ratio_clip') else 0.25
    config['agent']['policy_type'] = args.policy_type if hasattr(args, 'policy_type') else None
    config['agent']['sp_a_num'] = args.sp_a_num if hasattr(args, 'sp_a_num') else [3, 3]
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 1
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = 0.99

    config['trainer']['sample_step'] = 4 * env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 4
    config['evaluator']['eval_gap'] = args.eval_gap if hasattr(args, 'eval_gap') else 20000
    config['evaluator']['break_step'] = args.break_step if hasattr(args, 'break_step') else 2e6
    config['env'] = env
    config['gpu_id'] = args.gpu_id if hasattr(args, 'gpu_id') else 0
    config['if_cwd_time'] = args.if_cwd_time if hasattr(args, 'if_cwd_time') else True
    config['random_seed'] = 0
    beginer(config)


def demo_mujoco_swimmer_sac(args=None):
    from ray_elegantrl.agent import AgentSAC,AgentModSAC
    from ray_elegantrl.configs.configs_modsac import config

    import gym
    env = gym.make(args.env_id)
    state_dim = env.reset().shape[0]
    action_dim = env.action_space.sample().shape[0]
    env.close()

    env = {
        'id': args.env_id,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'action_type': 1,
        'reward_dim': 1,
        'max_step': 1000,
        'target_reward': 100,
    }
    config['agent']['class_name'] = AgentModSAC # AgentSAC obj act cri 爆炸
    config['random_seed']=0
    config['agent']['net_dim'] = 2 ** 8
    config['interactor']['rollout_num'] = 4
    config['trainer']['batch_size'] = 2 ** 8
    config['trainer']['policy_reuse'] = 2 ** 1
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['interactor']['reward_scale'] =args.reward_scale if hasattr(args, 'reward_scale') else 2**0
    config['buffer']['max_buf'] = 2 ** 20
    config['evaluator']['eval_gap'] = args.eval_gap if hasattr(args, 'eval_gap') else 20000
    config['evaluator']['pre_eval_times'] = 1
    config['evaluator']['eval_times'] = 4
    config['env'] = env
    config['gpu_id'] = 1
    config['if_cwd_time'] = True
    config['random_seed'] = 10086

    beginer(config)
if __name__ == '__main__':
    # demo_d3qn()
    # demo_sac()
    # demo_ppocma()
    import argparse

    parser = argparse.ArgumentParser(description='Mujoco RL')
    parser.add_argument('--env_id', type=str, default='Swimmer-v2')
    parser.add_argument('--demo_type', type=str, default='ppo2')
    parser.add_argument('--debug', default=False, action="store_true", )
    parser.add_argument('--policy_type', type=str, default=None)
    parser.add_argument('--objective_type', type=str, default='clip')
    parser.add_argument('--lambda_entropy', type=float, default=0.01)
    parser.add_argument('--sp_a_num', nargs='+', type=float, default=[3, 11])
    parser.add_argument('--reward_scale', type=float, default=2**-2)
    parser.add_argument('--break_step', type=int, default=2000000)
    parser.add_argument('--eval_gap', type=int, default=10000)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    ray.init(local_mode=True) if args.debug else ray.init()
    if args.demo_type in ['ppo2', 'ppo','ppo2cmaes']:
        args.reward_scale=2**0
        args.eval_gap = 100000
        demo_mujoco_swimmer_ppo(args)
    elif args.demo_type in ['sac']:
        args.reward_scale = 2 ** -2
        demo_mujoco_swimmer_sac(args)


