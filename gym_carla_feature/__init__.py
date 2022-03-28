from gym.envs.registration import register

register(
    id='carla-v0',
    entry_point='gym_carla_feature.env_a:CarlaEnv',
    reward_threshold=1000,
)

register(
    id='carla-r-v0',
    entry_point='gym_carla_feature.env_a:CarlaEnvMR',
    reward_threshold=1000,
)

register(
    id='carla-v1',
    entry_point='gym_carla_feature.review_env_b:CarlaEnv',
    reward_threshold=1000,
)
register(
    id='carla_vae-v1',
    entry_point='gym_carla_feature.scenarios.carla_vae_env:CarlaVAE',
    reward_threshold=1000,
)
register(
    id='carla_vae_mo-v1',
    entry_point='gym_carla_feature.scenarios.carla_vae_env:CarlaVAEMO',
    reward_threshold=1000,
)
register(
    id='carla-v2',
    entry_point='gym_carla_feature.start_env.env_2:CarlaEnv',
    reward_threshold=1000,
)