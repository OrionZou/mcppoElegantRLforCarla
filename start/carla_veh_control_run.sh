#!/bin/bash

#source /usr/local/miniconda3/bin/activate rl-base
cd /home/zgy/repos/ray_elegantrl
export PYTHONPATH=.
mkdir -p ./log

#
reward_type=12
nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2100 --random_seed 0  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_ppo_0.log &
sleep 30s
#reward_type=13
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2018 --random_seed 1  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_ppo_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2400 --random_seed 12 --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_ppo_1.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2500 --random_seed 123 --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_ppo_2.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2600 --random_seed 1234 --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_ppo_3.log &
#sleep 30s

#reward_type=13
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'rnnppo2' --port 2500 --batch_size 16 --policy_type mg --reward_type $reward_type --hidden_state_dim 32 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_rnnppomg_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'rnnppo2' --port 2500 --batch_size 16 --reward_type $reward_type --hidden_state_dim 32 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_rnnppo_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'rnnppo2' --port 2100 --batch_size 16 --reward_type $reward_type --if_zero_state --hidden_state_dim 32 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_rnnppo_1.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'rnnppo2' --port 2118 --batch_size 16 --reward_type $reward_type --infer_by_sequence --hidden_state_dim 32 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_rnnppo_2.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'rnnppo2' --port 2018 --batch_size 16 --reward_type $reward_type --infer_by_sequence --if_zero_state --hidden_state_dim 32 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_rnnppo_3.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'rnnppo2' --port 2400 --batch_size 16 --reward_type $reward_type --hidden_state_dim 64 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_rnnppo_2.log &
#sleep 30s

#reward_type=14
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2000 --if_critic_shared  --cost_threshold 0.1 0.02  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_0.log &
#sleep 30s
#reward_type=14
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2000 --cost_threshold 10 0.05  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_1.log &
#sleep 30s
#reward_type=14
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2100 --cost_threshold 0.02 0.05  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_2.log &
#sleep 30s
#reward_type=14
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2200 --cost_threshold 10 10  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_3.log &
#sleep 30s
#reward_type=14
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2300 --cost_threshold 0.02 10  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_4.log &
#sleep 30s


#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2000 --cost_threshold 10 0.03  --reward_type $reward_type --policy_type mg --if_rnn --hidden_state_dim 32 --batch_size 16 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_1.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2100 --cost_threshold 0.07 0.03  --reward_type $reward_type --policy_type mg --if_rnn --hidden_state_dim 32 --batch_size 16 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_2.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2200 --cost_threshold 10 10  --reward_type $reward_type --policy_type mg --if_rnn --hidden_state_dim 32 --batch_size 16 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_3.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2300 --cost_threshold 0.07 10  --reward_type $reward_type --policy_type mg --if_rnn --hidden_state_dim 32 --batch_size 16 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_4.log &
#sleep 30s

#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2400 --cost_threshold 0.07 0.03  --reward_type $reward_type --policy_type mg  --hidden_state_dim 32 --batch_size 16 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_5.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2500 --cost_threshold 0.07 0.03  --reward_type $reward_type --hidden_state_dim 32 --batch_size 16 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_6.log &
#sleep 30s

#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2300 --cost_threshold 0.1 0.1  --reward_type $reward_type --policy_type mg  --hidden_state_dim 32 --batch_size 16 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_5.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2200 --cost_threshold 0.1 0.1  --reward_type $reward_type --hidden_state_dim 32 --batch_size 16 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_6.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2300 --eval_gap 0 --cost_threshold 0.1 0.01  --reward_type $reward_type --hidden_state_dim 32 --batch_size 16 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_6.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2200 --cost_threshold 10 0.01  --reward_type $reward_type --hidden_state_dim 32 --batch_size 16 --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_10.log &
#sleep 30s

#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2000 --eval_gap 0 --cost_threshold 0.1 0.01  --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_12.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2100 --eval_gap 0 --cost_threshold 0.1 0.05  --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_12.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2400 --eval_gap 0 --cost_threshold 0.1 0.03  --reward_type $reward_type --objective_type auto_kl --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_12.log &
#sleep 30s
#reward_type=16
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2000 --eval_gap 0 --gamma 0.99 0.95 0.95 0.95 --cost_threshold 0.1 0.01 0.1  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_13.log &
#sleep 30s
#reward_type=17
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2100 --gamma 0.99 0.95 0.95 --cost_threshold 0.1 0.1  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_14.log &
#sleep 30s
#reward_type=18
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2400 --gamma 0.99 0.95 0.95 --cost_threshold 0.01 0.1  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_15.log &
#sleep 30s
#reward_type=17
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2300 --gamma 0.99 0.95 0.95 0.95 --weight 1 0. 0. 0. --cost_threshold 0.02 0.05 0.12  --reward_type $reward_type  --objective_type clip --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_18.log &
#sleep 30s
#reward_type=17
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2200 --eval_gap 0 --gamma 0.99 0.95 0.95 0.95 --weight 1 0. 0. 0. --cost_threshold 0.02 0.05 0.12  --reward_type $reward_type --policy_type mg  --objective_type clip --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_15.log &
#sleep 30s
reward_type=17
nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2000 --eval_gap 0 --gamma 0.99 0.95 0.95 0.95 --weight 1 0. 0. 0. --cost_threshold 0.02 0.05 0.12  --reward_type $reward_type  --objective_type clip --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_15.log &
sleep 30s
#reward_type=16
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2300 --eval_gap 0 --gamma 0.99 0.95 0.95 0.95 --weight 1 0. 0. 0. --cost_threshold 0.1 0.01 0.12  --reward_type $reward_type  --objective_type clip --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_16.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2000 2007 2011 2015 --eval_gap 0 --cost_threshold 0.1 0.05  --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_12.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2300 --eval_gap 0 --cost_threshold 0.1 0.05  --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_10.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2518 --eval_gap 0 --cost_threshold 0.1 0.1  --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_11.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2100 --cost_threshold 10 0.01  --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_7.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2200 --cost_threshold 0.1 10  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_8.log &
#sleep 30s
#reward_type=15
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2300 --cost_threshold 10 10   --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_9.log &
#sleep 30s

#reward_type=14
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2500 --cost_threshold 10 0.02  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_5.log &
#sleep 30s
#reward_type=14
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'cppo' --port 2300 --cost_threshold 10 0.05  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_cppo_6.log &
#sleep 30s
#
#reward_type=12 Failed
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'hcppo2'  --port 2500 --train_model mix --discrete_degree 3 --batch_size 256 --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_hcppo2_0.log &
#sleep 30s

#reward_type=12
#save_apth='/home/zgy/repos/ray_elegantrl/veh_control_logs/carla-v2_Town07_mountainroad_goodresult/s50_r1_12_ep200_dt0.2_False/AgentPPO2_None_clip/exp_2021-11-10-13-02-45_cuda:1/'
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'hierarchicalppo2' --hppo_save_path $save_apth --port 2500 --train_model discrete --discrete_degree 3 --batch_size 256 --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_hierarchicalppo2_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'hybridppo2' --port 2200 --discrete_degree 3 --if_share --batch_size 256 --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_hybridppo2_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'hybridppo2' --port 2400 --discrete_degree 2 --batch_size 256 --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_hybridppo2_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'hybridppo2' --port 2400 --discrete_degree 3 --if_share --batch_size 256 --reward_type $reward_type --if_sp_action_loss  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_hybridppo2_2.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'hybridppo2' --port 2300 --discrete_degree 2 --batch_size 256 --reward_type $reward_type --if_sp_action_loss --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_hybridppo2_3.log &
#sleep 30s

#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'discreteppo2' --port 2200 --batch_size 256 --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_discreteppo2_0.log &
#sleep 30s
#
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2300 --batch_size 256 --policy_type discrete_action_dim --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_sadppo2_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2000 --batch_size 256 --policy_type discrete_action_dim --sp_a_num 3 20 --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_sadppo2_0.log &
#sleep 30s
#
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2100 --batch_size 256 --policy_type discrete_action_dim --sp_a_num 3 50 --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_sadppo2_1.log &
#sleep 30s
#
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2200 --batch_size 256 --policy_type discrete_action_dim --sp_a_num 3 100 --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_sadppo2_2.log &
#sleep 30s

#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2300 --batch_size 256 --policy_type discrete_action_dim --sp_a_num 10 50 --reward_type $reward_type  --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_sadppo2_3.log &
#sleep 30s

#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2200  --reward_type $reward_type --policy_type mg --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_ppo2beta_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2018 --random_seed 1 --reward_type $reward_type --policy_type mg --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_ppo2mg_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2118 --random_seed 12 --reward_type $reward_type --policy_type mg --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_ppo2mg_1.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2200 --random_seed 123 --reward_type $reward_type --policy_type mg --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_ppo2mg_2.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2100  --reward_type $reward_type  --policy_type beta --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_ppo2mg_0.log &
#sleep 30s

#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2cmaes' --port 2400  --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_pppo2cmaes_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2cmaes' --port 2500 --random_seed 1 --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_pppo2cmaes_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2cmaes' --port 2118  --random_seed 12 --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_pppo2cmaes_3.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2cmaes' --port 2200 --random_seed 123 --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_pppo2cmaes_1.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2cmaes' --port 2400 --random_seed 1234 --reward_type $reward_type --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_pppo2cmaes_2.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'ppo2' --port 2400 --reward_type $reward_type --policy_type beta --lambda_entropy 0.01 --ratio_clip 0.2> ./log/exp_carla_ppo2beta2_0.log &
#sleep 30s

#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'sac' --port 2200 --reward_scale 5 --reward_type $reward_type --batch_size 256 > ./log/exp_carla_sac_1.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'sac' --port 2500 --reward_scale 5 --reward_type $reward_type --batch_size 256 > ./log/exp_carla_sac_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'sac' --port 2400 --reward_scale 0 --reward_type $reward_type --learning_rate 0.0001  --batch_size 256 > ./log/exp_carla_sac_0.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'sac' --port 2500 --reward_scale 4  --reward_type $reward_type --learning_rate 0.0001 --batch_size 256 > ./log/exp_carla_sac_1.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'sac' --port 2018 --reward_scale 4  --reward_type $reward_type --learning_rate 0.0004 --batch_size 256 > ./log/exp_carla_sac_1.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'sac' --port 2200 --reward_scale 3 --reward_type $reward_type --batch_size 256 > ./log/exp_carla_sac_2.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'sac' --port 2300 --reward_scale 1 --reward_type $reward_type --batch_size 512  > ./log/exp_carla_sac_4.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'sac' --port 2218 --reward_scale 1 --reward_type $reward_type --max_buf 18 --batch_size 256  > ./log/exp_carla_sac_5.log &
#sleep 30s

#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'd3qn' --port 2100 --reward_type $reward_type  --discrete_steer -0.6 -0.3 -0.1 0.0 0.1 0.3 0.6 > ./log/exp_carla1.log &
#sleep 30s
#reward_type=12
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'd3qn' --port 2100 --reward_type $reward_type  --discrete_steer -0.6 0.0 0.6 > ./log/exp_carla_d3qn_0.log &
#sleep 30s
#reward_type=11
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'd3qn' --port 2200 --reward_type $reward_type  --discrete_steer -0.6 0.0 0.6 > ./log/exp_carla_d3qn_0.log &
#sleep 30s
#reward_type=7
#nohup python gym_carla_feature/start_env/demo/veh_control.py --demo_type 'd3qn' --port 2006 2220 2120 2116 --reward_type $reward_type  --discrete_steer -0.2 0.0 0.2 > ./log/exp_carla1.log &
#sleep 30s