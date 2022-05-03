# mcppoElegantRLforCarla
There is the implementation of multi-constraint proximal policy optimization (MCPPO), which is modified based on a DRL framework -- [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)(小雅). The carla environment is modified based on [gym-carla](https://github.com/cjy1992/gym-carla). 

![Image text](https://github.com/GyChou/mcppoElegantRLforCarla/blob/main/images/town07-part.gif)

## Get started

### OS and Carla

Our experimental environment is based on Ubuntu18.04 and Carla0.9.11
Before training in Carla, you need
- install Carla, the tutorial: https://carla.readthedocs.io/en/latest/start_quickstart/
- install Docker, the tutorial: https://docs.docker.com/engine/install/ubuntu/
- install Nvidia-docker2 (for running env by GPU), the tutorial: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
- start carla server, the tutorial: https://carla.readthedocs.io/en/latest/build_docker/#nvidia-docker2 or start with ```python ./gym_carla_feature/start_env/demo/startServer.py```

### Python environment
Please see requirements.txt
```
pip install -r requirements.txt
```
### Training

After runing carla server, you can start a quickstart shell
```
bash ./start/carla_veh_control_run.sh
```
or start with 
```
python ./gym_carla_feature/start_env/demo/veh_control.py
```
You also use ray_elegantrl to try other envs such as
```
python ./ray_elegantrl/demo.py
```
### Evaluation
The tensorboard file and model file will be saved in ```./veh_control_logs```

## Citation
<!-- ```
@inproceedings{zou2022mcppo,
 title={Multi-Constraint Deep Reinforcement Learning for Smooth Action Control},
 author={Guangyuan Zou, Ying He, F. Richard Yu, Longquan Chen, Longquan Chen, Weike Pan, Zhong Ming},
 booktitle={the 31st International Joint Conference on Artificial Intelligence (IJCAI2022)},
}
``` -->
 Waiting update
<!-- The arxiv link to the paper:  -->


## Thanks 
Thanks for [小雅](https://github.com/AI4Finance-Foundation/ElegantRL) and [Yinyan Zeng](https://github.com/Yonv1943), the author. I learned a lot from this repository. 
