from gym_carla_feature.start_env.demo.eval_multpolicys_carla import *


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Carla RL')
    parser.add_argument('--reward_type', type=int, default=12)
    parser.add_argument('--port', nargs='+', type=int, default=[2060])
    parser.add_argument('--desired_speed', type=float, default=12)
    parser.add_argument('--max_step', type=int, default=800)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--net', type=str, default="ppo")
    parser.add_argument('--hidden_state_dim', type=int, default=128)
    # parser.add_argument('--town', type=str, default='Town03')
    # parser.add_argument('--task_mode', type=str, default='urbanroad')
    parser.add_argument('--action_space', type=int, default=None)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--if_noise_dt', default=False, action="store_true", )
    args = parser.parse_args()


    args.net = "cppo"
    args.reward_type=12
    print(f'# {args.net}')
    args.town='Town07'
    args.task_mode='mountainroad'
    args.max_step=800
    args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/constriant/AgentConstriantPPO2_None_clip/r17/exp_2021-12-12-13-26-49_cuda:1/model01/0001089126_164.3576'
    #
    # # args.town = 'Town07'
    # # args.task_mode = 'mountainroad'
    # # args.max_step = 4000
    # # args.path='/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentConstriantPPO2_None_clip/exp_2021-12-12-19-46-35_cuda:1/model01/0001063958_817.3339'
    #
    # # args.town = 'Town03'
    # # args.task_mode = 'urbanroad'
    # # args.max_step = 4000
    # # args.path='/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentConstriantPPO2_None_clip/exp_2021-12-12-21-56-36_cuda:1/model01/0001812759_844.4043'
    #
    policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    demo(policy, args=args)
    print(f"path:{args.path}")

    args.net = "ppo"
    args.reward_type=12
    print(f'# {args.net}-ppo')

    args.town='Town07'
    args.task_mode='mountainroad'
    args.max_step=800
    args.path = './veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_None_clip/exp_2021-12-03-21-56-07_cuda:1/model/0001871988_168.5951'
    #
    # # args.town = 'Town07'
    # # args.task_mode = 'mountainroad'
    # # args.max_step = 4000
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentPPO2_None_clip/exp_2021-12-09-23-36-27_cuda:1/model00/0001728393_865.4932'
    #
    # # args.town = 'Town03'
    # # args.task_mode = 'urbanroad'
    # # args.max_step = 4000
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentPPO2_None_clip/exp_2021-12-12-04-40-01_cuda:1/model00/0001985593_879.641'
    #
    policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    demo(policy, args=args)
    print(f"path:{args.path}")


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

    # args.net = "sac"
    # print(f'# {args.net}')
    #
    # args.town='Town07'
    # args.task_mode='mountainroad'
    # args.max_step=800
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentModSAC_None_None_RS5/exp_2021-12-06-20-53-45_cuda:1/model00/0001679059_170.2191'
    #
    # # args.town = 'Town07'
    # # args.task_mode = 'mountainroad'
    # # args.max_step = 4000
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentModSAC_None_None/exp_2021-12-10-11-12-56_cuda:1/model00/0001768920_837.4113'
    # #
    # # args.town = 'Town03'
    # # args.task_mode = 'urbanroad'
    # # args.max_step = 4000
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentModSAC_None_None/exp_2021-12-11-11-49-06_cuda:1/model00/0001071567_817.4923'
    #
    # policy = ActorSAC(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # demo(policy, args=args)
    # print(f"path:{args.path}")


    # args.net = "d3qn"
    # print(f'# {args.net}')
    #
    # args.town='Town07'
    # args.task_mode='mountainroad'
    # args.max_step=800
    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/discrete/AgentD3QN_None_None/exp_2021-12-04-01-01-20_cuda:1/model/0001568374_168.0686'
    #
    # # args.town = 'Town07'
    # # args.task_mode = 'mountainroad'
    # # args.max_step = 4000
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V2/AgentD3QN_None_None/exp_2021-12-09-23-37-27_cuda:1/model00/0001916828_719.6427'
    #
    # # args.town = 'Town03'
    # # args.task_mode = 'urbanroad'
    # # args.max_step = 4000
    # # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town-03-V1/AgentD3QN_None_None/exp_2021-12-11-11-49-36_cuda:1/model00/0001810156_840.9174'
    #
    # args.action_space = [[a, b] for a in [-1., 0., 1.] for b in [-0.6, -0.3, -0.1, 0., 0.1, 0.3, 0.6]]
    # policy = QNetTwinDuel(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=len(args.action_space))
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
    # args.town = 'Town07'
    # args.task_mode = 'mountainroad'
    # args.max_step = 800

    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_discrete_action_dim_clip/exp_2021-11-22-16-27-08_cuda:1'
    # policy = ActorSADPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # policy.set_sp_a_num([3, 11])
    # demo(policy, args=args)
    # print(f"path:{args.path}")

    # args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_s50_a2_r1_tr200_ms200_False/AgentPPO2_discrete_action_dim_clip/exp_2022-01-01-22-18-09_cuda:1/model00/0001088391_167.6529'
    # policy = ActorSADPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    # policy.set_sp_a_num([3, 50])
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