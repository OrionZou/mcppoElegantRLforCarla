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
        'id': 'HopperPyBulletEnv-v0',
        'state_dim': 15,
        'action_dim': 2,
        'action_type': 1, # choose -1 discrete action space | 1 continuous action space | 0 hybird action space |
        'reward_dim': 1,
        'target_return': 0,
        'max_step': max_step,
    },
    'agent': {
        'class_name': AgentPPO,
        'net_dim': 2 ** 8,
        'ratio_clip': 0.3,
        'c_dclip': 3,
        'lambda_entropy': 0.04,
        'lambda_gae_adv': 0.97,
        'total_iterations': 1000,
        'if_use_gae': True,
        'if_use_dn': False,
        'learning_rate': 1e-4,
        'soft_update_tau': 2 ** -8,
        'policy_type': None,
        'objective_type': 'clip',
        'if_ir': False,
        'if_rnn': True,
        'if_store_state': True,
        'hidden_state_dim': 256,
        'rnn_timestep':16,
        'beta': 3.,
    },
    'trainer': {
        'batch_size': 2 ** 8,
        'policy_reuse': 2 ** 4,
        'sample_step': max_step * rollout_num,
    },
    'interactor': {
        'horizon_step': max_step * rollout_num,
        'reward_scale': 2 ** 0,
        'gamma': 0.99,
        'rollout_num': rollout_num,
    },
    'buffer': {
        'max_buf': max_step * rollout_num,
        'if_on_policy': True,
        'if_per': False,
        'if_gpu': False,
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
