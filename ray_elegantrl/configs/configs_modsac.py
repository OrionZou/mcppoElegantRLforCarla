from ray_elegantrl.agent import *

# Default config for offpolicy
max_step = 150
rollout_num = 4
config = {
    'gpu_id': 0,
    'cwd': None,
    'if_cwd_time': True,
    'random_seed': 0,
    'env': {
        'id': 'HopperPyBulletEnv-v0',
        'state_dim': 15,
        'action_dim': 2,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_return': 0,
        'max_step': max_step,
    },
    'agent': {
        'class_name': AgentModSAC,
        'net_dim': 2 ** 8,
        'if_use_dn': False,
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
        'max_buf': 2 ** 17,
        'if_on_policy': False,
        'if_per': False,
        'if_gpu': True,
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
