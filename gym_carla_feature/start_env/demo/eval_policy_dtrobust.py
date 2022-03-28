from gym_carla_feature.start_env.demo.eval_multpolicys_carla import *

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Carla RL')
    parser.add_argument('--reward_type', type=int, default=12)
    parser.add_argument('--port', nargs='+', type=int, default=[2050])
    parser.add_argument('--desired_speed', type=float, default=12)
    parser.add_argument('--max_step', type=int, default=800)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--net', type=str, default="ppo")
    parser.add_argument('--hidden_state_dim', type=int, default=128)
    parser.add_argument('--action_space', type=int, default=None)
    parser.add_argument('--noise_std', type=float, default=0)
    parser.add_argument('--if_noise_dt', default=False, action="store_true", )

    args = parser.parse_args()
    args.test_num = 1
    dt_list = [0.02, 0.05, 0.1, 0.2,  0.3,0.4,0.5]
    args.if_noise_dt = False
    for dt in dt_list:
        args.dt = dt
        args.max_step = int((0.05 / dt) * 800)
        print('*' * 10, dt, '*' * 10)

        args.net = "ppo"
        print(f'# {args.net}')
        args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_None_clip/exp_2021-12-03-21-56-07_cuda:1/model/0001871988_168.5951'
        policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
        demo(policy, args)

        args.net = "cppo"
        print(f'# {args.net}')
        args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/constriant/AgentConstriantPPO2_None_clip/r17/exp_2021-12-12-13-26-49_cuda:1/model01/0001085126_164.667'
        policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
        demo(policy, args)

        args.net = "discreteppo"
        print(f'# {args.net}')
        args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/discrete/AgentPPO2_discrete_clip/exp_2021-12-05-14-43-25_cuda:0/model/0001654295_167.8884'
        args.action_space = [[a, b] for a in [-1., 0., 1.] for b in [-0.6, -0.3, -0.1, 0., 0.1, 0.3, 0.6]]
        policy = ActorDiscretePPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=len(args.action_space))
        demo(policy, args)

    noise_std_list = [0.]
    args.if_noise_dt = True

    # args.if_noise_dt = True
    # noise_std_list = [0., 0.01, 0.05, 0.1]
    # # noise_std_list = [0.05]
    # for noise_std in noise_std_list:
    #     args.noise_std = noise_std
    #     print('*' * 10, noise_std, '*' * 10)
    #
    #     args.net = "ppo"
    #     print(f'# {args.net}')
    #     args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/continous/AgentPPO2_None_clip/exp_2021-12-03-21-56-07_cuda:1/model/0001871988_168.5951'
    #     policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    #     demo(policy, args)
    #
    #     args.net = "cppo"
    #     print(f'# {args.net}')
    #     args.path = '/home/zgy/repos/ray_elegantrl/veh_control_logs/veh_control/town07-V1/constriant/AgentConstriantPPO2_None_clip/r17/exp_2021-12-12-13-26-49_cuda:1/model01/0001085126_164.667'
    #     policy = ActorPPO(mid_dim=2 ** 8, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    #     demo(policy, args)
