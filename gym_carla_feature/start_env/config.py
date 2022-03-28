params = {
    'number_of_vehicles': 0,  # number of npc vehicles
    # 'number_of_walkers': 0,  # number of npc walkers
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 4,  # the number of past steps to draw
    'dt': 0.2,  # time interval between two frames
    'if_noise_dt': False,
    'noise_std': 0.01,
    'noise_interval':10,
    'action_type': 1,  # choose -1 discrete action space | 1 continuous action space | 0 hybird action space |
    'discrete_acc': [-1.0, 0.0, 1.0],  # discrete value of normalized accelerations
    # 'discrete_steer': [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
    #                    0.6, 0.7, 0.8, 0.9, 1.],  # discrete value of steering angles
    # 'discrete_steer': [-1.0, -0.8, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4,
    #                   0.8, 1.],  # discrete value of steering angles during delta t
    'discrete_steer': [-0.2, 0.0, 0.2],
    'continuous_accel_range': 1.,  # continuous acceleration range
    'continuous_steer_range': 1.,  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': [2000, 2004, 2008, 2012, 2016, 2020, 2024, 2028, 2032, 2036],  # connection port for mp
    'town': 'Town03',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03),fixed_route (only for Town03)]
    'max_step': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.5,  # bin size of lidar sensor (meter) 0.125
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'if_dest_end': False,
    'reward_type': 0,
    'desired_speed': 8,  # desired speed (m/s)
    'sampling_radius': 5,
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'obs_space_type': ['orgin_state', 'waypoint', 'brideye', 'lidar', 'camera'],
    'render': False,
    'autopilot': False,
}
