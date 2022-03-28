from ray_elegantrl.agent import *

# Default config for on policy | ppo
max_step = 1024
rollout_num = 4
config = {
    'gpu_id': 0,
    'cwd': None,
    'if_cwd_time': True,
    'random_seed': 0,
    'env': {
        'id': 'LunarLander-v2',
        'state_dim': 8,
        'action_dim': 4,
        'if_discrete_action': True,
        'reward_dim': 1,
        'target_return': 0,
        'max_step': max_step,
    },
    'agent': {
        'class_name': AgentD3QN,
        'net_dim': 2 ** 8,
        'explore_rate': 0.25,
        'learning_rate': 1e-4,
        'soft_update_tau': 2 ** -8,
        'policy_type': None,
        'objective_type': None,
        'if_rnn': False,
        'if_store_state': True,
    },
    'trainer': {
        'batch_size': 2 ** 8,
        'policy_reuse': 2 ** 0,
        'sample_step': max_step * rollout_num,
    },
    'interactor': {
        'horizon_step': max_step * rollout_num,
        'reward_scale': 2 ** 0,
        'gamma': 0.99,
        'rollout_num': rollout_num,
    },
    'buffer': {
        'max_buf': 2 ** 20,
        'if_on_policy': False,
        'if_per': False,
        'if_gpu': True,
        'if_discrete_action': True,
    },
    'evaluator': {
        'pre_eval_times': 2,  # for every rollout_worker 0 means cencle pre_eval
        'eval_times': 4,  # for every rollout_worker
        'eval_gap':8192,
        'if_save_model': True,
        'break_step': 2e6,
        'satisfy_return_stop': False,
    }

}
