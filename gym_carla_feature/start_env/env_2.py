import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import carla
import pygame
import random
import math
from skimage.transform import resize
from collections import deque
from gym_carla_feature.start_env.render import BirdeyeRender
from gym_carla_feature.start_env.route_planner import RoutePlanner
from gym_carla_feature.start_env.misc import write_yaml, read_yaml, compute_distance, is_within_distance_ahead, \
    get_pos, get_preview_lane_dis, display_to_rgb, rgb_to_display_surface, vec_is_within_distance_ahead, compute_angle, \
    draw_waypoints
from gym_carla_feature.start_env.config import params as defalt_config


class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params=defalt_config):
        # Parameters
        self.display_size = params['display_size']  # rendering screen size
        self.max_past_step = params['max_past_step']
        self.number_of_vehicles = params['number_of_vehicles']
        # self.number_of_walkers = params['number_of_walkers']
        self.init_dt = self.dt = params['dt']
        self.task_mode = params['task_mode']
        self.max_step = params['max_step']
        self.max_waypt = params['max_waypt']
        self.obs_range = params['obs_range']
        self.lidar_bin = params['lidar_bin']
        self.d_behind = params['d_behind']
        self.obs_size = int(self.obs_range / self.lidar_bin)
        self.out_lane_thres = params['out_lane_thres']
        self.sampling_radius = params['sampling_radius']
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.display_route = params['display_route']
        self.is_render = params['render']
        self.autopilot = params['autopilot']
        self.if_dest_end = params['if_dest_end']
        self.reward_type = params['reward_type']

        self.if_noise_dt = params['if_noise_dt']
        # if self.if_noise_dt:
        #     self.dt = self.init_dt = self.init_dt / 2
        self.noise_std = params['noise_std']
        self.noise_interval = params['noise_interval']

        # action space
        self.action_type = params['action_type']
        self.discrete_act = [params['discrete_acc'], params['discrete_steer']]  # acc, steer
        self.continuous_act_range = [params['continuous_accel_range'], params['continuous_steer_range']]
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.action_type < 0:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        elif self.action_type > 0:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # acc, steer
        else:
            # todo for hybrid action space
            self.action_space = spaces.Dict({
                'continuous_acc': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'discrete_steer': spaces.Discrete(self.n_steer)
            })

        # observation space
        self.obs_space_type = params['obs_space_type']
        observation_space_dict = {}
        if 'orgin_state' in self.obs_space_type:
            observation_space_dict = {
                'orgin_state': spaces.Box(low=-1000, high=1000, shape=(14,), dtype=np.float32)
            }
        if 'waypoint' in self.obs_space_type:
            # 5 is max waypoints interal
            observation_space_dict.update(
                {'waypoint': spaces.Box(low=-self.sampling_radius * self.max_waypt,
                                        high=self.sampling_radius * self.max_waypt,
                                        shape=(self.max_waypt * 3,), dtype=np.float32)})
        if 'othervehs' in self.obs_space_type:
            # 5 is max waypoints interal
            observation_space_dict.update(
                {'othervehs': spaces.Box(low=-100,
                                         high=100,
                                         shape=(5 * 5,), dtype=np.float32)})
        if 'brideye' in self.obs_space_type:
            observation_space_dict.update(
                {'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8)})
        if 'lidar' in self.obs_space_type:
            observation_space_dict.update(
                {'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8)})
        if 'camera' in self.obs_space_type:
            observation_space_dict.update(
                {'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8)})
        self.observation_space = spaces.Dict(observation_space_dict)

        # Connect to carla server and get world object
        self.set_client_port(params['port'])
        print(f'connecting to Carla server port:{self.port}...')
        self.client = carla.Client('localhost', self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(params['town'])
        print('Carla server connected!')

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt
        self.settings.max_substeps = 10
        self.settings.max_substep_delta_time = self.settings.fixed_delta_seconds / self.settings.max_substeps

        # Get trafficmanager
        self.traffic_manager = self.client.get_trafficmanager(self.port + 3)
        # Disable sync mode
        self._set_synchronous_mode(True)

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Get vehicle spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())

        # Destination
        self.route_wps = None
        if params['task_mode'] == 'roundabout':
            self.route_dest = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
            self.route_begin = [52.1 + np.random.uniform(-5, 5), -4.2, 178.66]  # random
            # self.route_begin=[52.1,-4.2, 178.66] # static
        elif params['task_mode'] == 'urbanroad':
            self.route_begin = self.vehicle_spawn_points[81]
            self.route_wps = [self.vehicle_spawn_points[166],
                              self.vehicle_spawn_points[196], ]
            self.route_dest = self.vehicle_spawn_points[205]
        elif params['task_mode'] == 'mountainroad':
            # self.route_begin = self.vehicle_spawn_points[41]
            # self.route_dest = self.vehicle_spawn_points[38]
            self.route_begin = self.vehicle_spawn_points[83]
            self.route_wps = [self.vehicle_spawn_points[72],
                              self.vehicle_spawn_points[38],
                              self.vehicle_spawn_points[107],
                              self.vehicle_spawn_points[27], ]
            self.route_dest = self.vehicle_spawn_points[98]
        elif params['task_mode'] == 'mountainroad2':
            # self.route_begin = self.vehicle_spawn_points[41]
            # self.route_dest = self.vehicle_spawn_points[38]
            self.route_begin = self.vehicle_spawn_points[83]
            self.route_wps = [self.vehicle_spawn_points[72],
                              self.vehicle_spawn_points[38],
                              self.vehicle_spawn_points[107]]
            self.route_dest = self.vehicle_spawn_points[27]
        else:
            self.route_begin = self.vehicle_spawn_points[0]
            self.route_dest = self.vehicle_spawn_points[1]

        # Get walker spawn points
        # self.walker_spawn_points = []
        # for i in range(self.number_of_walkers):
        #     spawn_point = carla.Transform()
        # loc = self.world.get_random_location_from_navigation()
        # if (loc != None):
        #     spawn_point.location = loc
        # self.walker_spawn_points.append(spawn_point)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Invasion sensor
        self.invasion_hist = []  # The invasion history
        self.invasion_hist_l = 1  # invasion history length
        self.lane_invasion_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')

        # Lidar sensor
        if 'lidar' in self.obs_space_type or self.is_render:
            self.lidar_data = None
            self.lidar_height = 2.1
            self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
            self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            self.lidar_bp.set_attribute('channels', '32')
            self.lidar_bp.set_attribute('range', '5000')

        # Camera sensor
        if 'camera' in self.obs_space_type or self.is_render:
            self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
            self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
            self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            # Modify the attributes of the blueprint to set image resolution and field of view.
            self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
            self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
            self.camera_bp.set_attribute('fov', '110')
            # Set the time in seconds between sensor captures
            self.camera_bp.set_attribute('sensor_tick', '0.02')

        self.total_record = {}
        # Record the time of total steps and resetting steps
        self.total_record['reset_step'] = 0
        self.total_record['total_step'] = 0

        # Record stop time step for reward avoiding ego vehicle don't move
        self.stop_step_thre = 30
        # todo
        self.vehicle_traffic = 1

        # Initialize the renderer

        if self.is_render or \
                'brideye' in self.obs_space_type or \
                'lidar' in self.obs_space_type or \
                'camera' in self.obs_space_type:
            self._init_birdeye_render()

    def set_client_port(self, ports):
        if isinstance(ports, list):
            pdict = read_yaml()
            self.port = pdict['port'].pop()
            write_yaml(pdict)
        else:
            self.port = ports

    def reset(self):
        # Clear sensor objects
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None

        # Delete sensors, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision',
                                'sensor.other.lane_invasion',
                                'sensor.lidar.ray_cast',
                                'sensor.camera.rgb',
                                'vehicle.*',
                                'controller.ai.walker',
                                'walker.*'])

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if spawn_point == self.route_begin:
                    continue
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1

        # Spawn pedestrians
        # random.shuffle(self.walker_spawn_points)
        # count = self.number_of_walkers
        # if count > 0:
        #     for spawn_point in self.walker_spawn_points:
        #         if self._try_spawn_random_walker_at(spawn_point):
        #             count -= 1
        #         if count <= 0:
        #             break
        # while count > 0:
        #     if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
        #         count -= 1

        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        # self.walker_polygons = []
        # walker_poly_dict = self._get_actor_polygons('walker.*')
        # self.walker_polygons.append(walker_poly_dict)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()
            if self.task_mode == 'random':
                self.route_begin = random.choice(self.vehicle_spawn_points)
            if self._try_spawn_ego_vehicle_at(self.route_begin):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []

        # Add lane invasion sensor
        self.lane_invasion_sensor = self.world.spawn_actor(self.lane_invasion_bp, carla.Transform(), attach_to=self.ego)
        self.lane_invasion_sensor.listen(lambda event: get_invasion_hist(event))

        def get_invasion_hist(event):
            for x in event.crossed_lane_markings:
                self.invasion_hist.append(x.type)
            while len(self.invasion_hist) > self.invasion_hist_l:
                self.invasion_hist.pop(0)

        self.invasion_hist = []

        if self.is_render or 'lidar' in self.obs_space_type:
            # Add lidar sensor
            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
            self.lidar_sensor.listen(lambda data: get_lidar_data(data))

            def get_lidar_data(data):
                self.lidar_data = data

        if self.is_render or 'camera' in self.obs_space_type:
            # Add camera sensor
            self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
            self.camera_sensor.listen(lambda data: get_camera_img(data))

            def get_camera_img(data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.camera_img = array

        # Update timesteps
        self.total_record['time_step'] = 0
        self.total_record['time'] = 0
        self.total_record['reset_step'] += 1

        # Record var for episode
        self.total_record['total_dis'] = 0
        # Reset stop veh steps count
        self.total_record['stop_step_time'] = 0
        self.total_record['reward_v_sum'] = 0.
        self.tmp_goal = None
        # init histroy actions
        self.his_act_deque = deque(maxlen=(3 * int(1 / self.dt) + 1))

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        start_time = 2  # s
        for _ in range(int(start_time / self.dt)):
            self.ego.apply_control(carla.VehicleControl(throttle=1., steer=0., brake=0.))
            self.world.tick()

        # Update route and set tmp goal
        self.routeplanner = RoutePlanner(self.ego, self.max_waypt, self.sampling_radius)
        if self.task_mode == 'random':
            self.route_dest = random.choice(self.vehicle_spawn_points)
            while self.route_dest == self.route_begin:
                self.route_dest = random.choice(self.vehicle_spawn_points)
        if self.route_wps is None:
            self.routeplanner.set_destination(self.route_dest.location)
            self.route_wps_copy = None
        else:
            self.route_wps_copy = self.route_wps.copy()
            self.route_wps_copy.reverse()
            self.routeplanner.set_destination(self.route_wps_copy.pop().location)
        self._receive_route()

        # Set ego information for render
        if self.is_render or \
                'brideye' in self.obs_space_type or \
                'lidar' in self.obs_space_type or \
                'camera' in self.obs_space_type:
            self.birdeye_render.set_hero(self.ego, self.ego.id)
        self.last_ego_transform = None
        return self._convert_obs(self._get_obs())

    def apply_action(self, action):
        # autopilot model
        if self.autopilot:
            return None
        # rule control
        if isinstance(action, carla.VehicleControl):
            self.last_control = self.ego.get_control()
            self.ego.apply_control(action)
            return action.throttle - action.brake, action.steer

        # rl control
        # Calculate acceleration and steering
        if self.action_type < 0:
            acc = self.discrete_act[0][action // self.n_steer]
            steer = self.discrete_act[1][action % self.n_steer]
        elif self.action_type > 0:
            acc = action[0]
            steer = action[1]
        else:
            acc = action[0]
            steer = self.discrete_act[1][int(action[1])]
        acc = acc * 3.0

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0., 1.) * self.continuous_act_range[0]
            # throttle = np.clip(acc, 0., 1.) * self.continuous_act_range[0]
            brake = 0.
        else:
            throttle = 0.
            brake = np.clip(-acc / 8, 0., 1.) * self.continuous_act_range[0]
            # brake = np.clip(-acc, 0., 1.) * self.continuous_act_range[0]

        # todo distrete
        # decimal = 1
        # steer = np.round_(steer, decimal)

        steer = steer * self.continuous_act_range[1]
        self.last_control = self.ego.get_control()
        self.ego.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake)))
        return throttle - brake, -steer

    def step(self, action):

        # Apply control
        action_lon, action_lat = self.apply_action(action)
        # if self.ego.is_at_traffic_light():
        #     self.ego.get_traffic_light().set_green_time(100)
        self.last_ego_transform = self.ego.get_transform()
        if self.if_noise_dt and self.total_record['time_step'] % self.noise_interval == 0:
            self.dt = np.clip(np.random.normal(self.init_dt, self.noise_std, 1), self.init_dt, 0.2)
            self.settings = self.world.get_settings()
            self.settings.fixed_delta_seconds = round(float(self.dt), 3)
            self.settings.max_substep_delta_time = self.settings.fixed_delta_seconds / self.settings.max_substeps
            self.world.apply_settings(self.settings)
        self.world.tick()

        # top view
        spectator = self.world.get_spectator()
        transform = self.ego.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40),
                                                carla.Rotation(pitch=-90)))
        draw_waypoints(self.world, self.waypoints[0:5])

        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        # walker_poly_dict = self._get_actor_polygons('walker.*')
        # self.walker_polygons.append(walker_poly_dict)
        # while len(self.walker_polygons) > self.max_past_step:
        #     self.walker_polygons.pop(0)

        # route planner
        self._receive_route()

        # Update timesteps
        # self.dt = self.dt * 2 if self.if_noise_dt else self.dt
        self.total_record['time_step'] += 1
        self.total_record['total_step'] += 1
        self.total_record['time'] += self.dt
        obs = self._get_obs()
        r_info = self._get_reward(obs)
        done = self._terminal(r_info)
        info = self._get_info(obs, r_info)
        # self.dt = self.dt / 2 if self.if_noise_dt else self.dt
        # if done: #debug
        #     print()
        return self._convert_obs(obs), self._convert_reward(done, r_info), done, info

    def update_his_deque(self, a_lon, a_lat, lateral_dis):
        # update history action
        act_control = self.ego.get_control()
        ego_trans = self.ego.get_transform()
        self.his_act_deque.append([act_control.throttle - act_control.brake,
                                   act_control.steer,
                                   ego_trans.rotation.yaw,
                                   a_lon,
                                   a_lat,
                                   abs(lateral_dis) / self.out_lane_thres,
                                   ])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        pass

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
          bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _init_birdeye_render(self):
        """Initialize the birdeye view renderer.
        """
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_size * 3, self.display_size),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
        birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)
        traffic_manager = self.client.get_trafficmanager(self.port + 3)
        traffic_manager.set_synchronous_mode(synchronous)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)

        if vehicle is not None:
            vehicle.set_autopilot(True, self.port + 3)
            # if ignore_trafficlights
            self.traffic_manager.ignore_lights_percentage(vehicle, 100)
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.
        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())  # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 0:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            if self.autopilot:
                vehicle.set_autopilot(True, self.port + 3)
            self.ego = vehicle
            return True

        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
          filt: the filter indicating what type of actors we'll look at.

        Returns:
          actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _receive_route(self):
        self.last_vehicle_traffic = self.vehicle_traffic
        if self.routeplanner.is_empty_queue() and self.route_wps_copy is not None:
            if len(self.route_wps_copy) > 0:
                self.routeplanner.set_destination(self.route_wps_copy.pop().location, clean=False)
            else:
                self.routeplanner.set_destination(self.route_dest.location, clean=False)
                self.route_wps_copy = None
        if self.tmp_goal is None:
            self.waypoints, self.current_waypoint, self.tmp_goal, self.vehicle_traffic, self.vehicle_front = self.routeplanner.run_step()
        else:
            # 如果临时目标不在车辆前方，就更新goal
            if is_within_distance_ahead(self.tmp_goal.transform, self.ego.get_transform(), max_distance=20):
                self.waypoints, self.current_waypoint, _, self.vehicle_traffic, self.vehicle_front = self.routeplanner.run_step()
            else:
                self.waypoints, self.current_waypoint, self.tmp_goal, self.vehicle_traffic, self.vehicle_front = self.routeplanner.run_step()

    def _get_obs(self):
        """Get the observations."""
        obs = {}
        # State observation
        ego_trans = self.ego.get_transform()
        ego_x, ego_y = ego_trans.location.x, ego_trans.location.y
        norm_ego_x = ego_trans.location.x / 100
        norm_ego_y = ego_trans.location.y / 100

        ego_yaw = (ego_trans.rotation.yaw) / 180 * np.pi

        ahead_wp_idx = 0
        waypoints = [[wp.transform.location.x, wp.transform.location.y, wp.transform.rotation.yaw] for wp in
                     self.waypoints]
        for i in range(3):
            if vec_is_within_distance_ahead(waypoints[i], np.array([ego_x, ego_y, ego_trans.rotation.yaw]),
                                            max_distance=20):
                ahead_wp_idx = i
                break
        lateral_dis, curr_wp_yaw_vector = get_preview_lane_dis(waypoints, ego_x, ego_y, ahead_wp_idx)
        ego_vector = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
        delta_yaw_wp = np.arcsin(np.clip(np.cross(curr_wp_yaw_vector, ego_vector), -1., 1.))
        _, curr_wp_yaw_vector2 = get_preview_lane_dis(waypoints, ego_x, ego_y, ahead_wp_idx + 1)
        delta_yaw_wp2 = np.arcsin(np.clip(np.cross(curr_wp_yaw_vector2, ego_vector), -1., 1.))

        v = self.ego.get_velocity()
        v = np.array([v.x, v.y])
        v_lon = np.dot(v, ego_vector)
        v_lat = np.cross(v, ego_vector)
        v_rate_la_lo = np.clip(v_lat / (v_lon + 1), 0, 10)
        a = self.ego.get_acceleration()
        a = np.array([a.x, a.y])
        a_lon = np.dot(a, ego_vector)
        a_lat = np.cross(a, ego_vector)

        control = self.ego.get_control()
        steer = control.steer

        if self.last_ego_transform is not None:
            delta_x = abs(ego_x - self.last_ego_transform.location.x)
            delta_y = abs(ego_y - self.last_ego_transform.location.y)
            radian = (self.last_ego_transform.rotation.yaw) / 180 * np.pi
            delta_yaw = np.arcsin(np.clip(np.cross(np.array([np.cos(radian), np.sin(radian)]), ego_vector), -1., 1.))
        else:
            delta_x = 0
            delta_y = 0
            delta_yaw = 0

        self.update_his_deque(a_lon=a_lon, a_lat=a_lat, lateral_dis=lateral_dis)
        state = np.array([lateral_dis,  # rough distance from lane center to ego
                          delta_yaw_wp,  # delta yaw between current ego and next waypoint
                          delta_yaw_wp2,  # delta yaw between current ego and next next waypoint
                          v_lon,  # longitudinal speed
                          v_lat,  # lateral speed
                          delta_x,  # distance from current ego position to last ego position in x axis
                          delta_y,  # distance from current ego position to last ego position in y axis
                          delta_yaw,

                          v_rate_la_lo,
                          # norm_ego_x,
                          # norm_ego_y,
                          a_lon,
                          a_lat,
                          steer,
                          norm_ego_x,
                          norm_ego_y,
                          #
                          # delta_x,  # distance from current ego position to last ego position in x axis
                          # delta_y,  # distance from current ego position to last ego position in y axis
                          # delta_yaw,  # distance from current ego position to last ego position in yawl
                          # self.total_record['time_step'] * self.dt,
                          # self.vehicle_traffic,  # todo
                          # self.vehicle_front
                          ])  # todo
        if 'orgin_state' in self.obs_space_type:
            obs.update({'orgin_state': state})

        if 'brideye' in self.obs_space_type or \
                'lidar' in self.obs_space_type or \
                'camera' in self.obs_space_type or \
                self.is_render:
            ## Birdeye rendering
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons
            # self.birdeye_render.walker_polygons = self.walker_polygons
            self.birdeye_render.waypoints = waypoints

            # birdeye view with roadmap and actors
            birdeye_render_types = ['roadmap', 'actors']
            if self.display_route:
                birdeye_render_types.append('waypoints')
            self.birdeye_render.render(self.display, birdeye_render_types)
            birdeye = pygame.surfarray.array3d(self.display)
            birdeye = birdeye[0:self.display_size, :, :]
            birdeye = display_to_rgb(birdeye, self.obs_size)
            # Add the waypoints to lidar image
            if self.display_route:
                wayptimg = (birdeye[:, :, 0] <= 10) * (birdeye[:, :, 1] <= 10) * (birdeye[:, :, 2] >= 240)
            else:
                wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
            wayptimg = np.expand_dims(wayptimg, axis=2)
            wayptimg = np.fliplr(wayptimg)
            if 'brideye' in self.obs_space_type:
                obs.update({'birdeye': birdeye.astype(np.uint8)})

            if self.is_render:
                # Display birdeye image
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, 0))

        if 'lidar' in self.obs_space_type or \
                self.is_render:
            ## Lidar image generation
            # Get point cloud data
            point_cloud = [[location.point.x, location.point.y, location.point.z] for location in
                           self.lidar_data]
            point_cloud = np.array(point_cloud)
            # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
            # and z is set to be two bins.
            x_bins = np.arange(-self.d_behind, self.obs_range - self.d_behind + self.lidar_bin, self.lidar_bin)
            y_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
            z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.27, 1]
            # Get lidar image according to the bins
            lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
            lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)  # red
            lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)  # green
            lidar = np.rot90(lidar, 2)
            # Add the waypoints to lidar image
            if self.display_route:
                wayptimg = (birdeye[:, :, 0] <= 10) * (birdeye[:, :, 1] <= 10) * (birdeye[:, :, 2] >= 240)
            else:
                wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
            wayptimg = np.expand_dims(wayptimg, axis=2)
            wayptimg = np.fliplr(wayptimg)

            # Get the final lidar image
            lidar = np.concatenate((lidar, wayptimg), axis=2)  # blue
            lidar = np.flip(lidar, axis=1)
            lidar = lidar * 255  # lidar final image

            if 'lidar' in self.obs_space_type:
                obs.update({'lidar': lidar.astype(np.uint8)})
            if self.is_render:
                # Display lidar image
                lidar_surface = rgb_to_display_surface(lidar, self.display_size)
                self.display.blit(lidar_surface, (self.display_size, 0))
        if 'camera' in self.obs_space_type or \
                self.is_render:
            camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
            if 'camera' in self.obs_space_type:
                obs.update({'camera': camera.astype(np.uint8)})
            if self.is_render:
                ## Display camera image
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size * 2, 0))

        dis_scale = self.sampling_radius
        waypoints = np.zeros((self.max_waypt, 3))
        # for i, wp in enumerate(self.waypoints):
        # # for i, wp in enumerate(self.waypoints[ahead_wp_idx:]):
        #     wp = wp.transform
        #     wp_yaw_vec = np.array([np.cos(wp.rotation.yaw / 180 * np.pi), np.sin(wp.rotation.yaw / 180 * np.pi)])
        #     wp_vec = np.array([ego_x - wp.location.x, ego_y - wp.location.y])
        #     waypoints[i] = np.array([np.dot(wp_vec, ego_vector) / dis_scale,
        #                              np.cross(wp_vec, ego_vector)/ dis_scale,
        #                              np.arccos(np.clip(np.dot(ego_vector, wp_yaw_vec), -1., 1.))])

        pre_wp = self.waypoints[0].transform
        pre_wp_vec = np.array([np.cos(pre_wp.rotation.yaw / 180 * np.pi), np.sin(pre_wp.rotation.yaw / 180 * np.pi)])
        waypoints[0] = np.array([[(ego_x - pre_wp.location.x) / dis_scale,
                                  (ego_y - pre_wp.location.y) / dis_scale,
                                  np.arccos(np.clip(np.dot(pre_wp_vec, ego_vector), -1., 1.))]])
        for i, wp in enumerate(self.waypoints[1:]):
            wp = wp.transform
            wp_vec = np.array([np.cos(wp.rotation.yaw / 180 * np.pi), np.sin(wp.rotation.yaw / 180 * np.pi)])
            waypoints[i + 1] = np.array(([(pre_wp.location.x - wp.location.x) / dis_scale,
                                          (pre_wp.location.y - wp.location.y) / dis_scale,
                                          np.arccos(np.clip(np.dot(pre_wp_vec, wp_vec), -1., 1.))]))
            pre_wp = wp
            pre_wp_vec = wp_vec

        if 'waypoint' in self.obs_space_type:
            obs.update({'waypoint': waypoints})

        if 'othervehs' in self.obs_space_type:
            near_veh_np = self._get_near_vehs()
            obs.update({'othervehs': near_veh_np})

        if self.is_render:
            # Display on pygame
            pygame.display.flip()

        return obs

    def _get_reward(self, obs):
        """Calculate the step reward."""
        ego_trans = self.ego.get_transform()
        ego_x, ego_y = ego_trans.location.x, ego_trans.location.y

        # reward for arrive destination
        r_dest = 0
        r_time_episode = 0
        if self.route_dest is not None and self.if_dest_end:  # If at destination
            if np.sqrt((ego_x - self.route_dest.location.x) ** 2 + (ego_y - self.route_dest.location.y) ** 2) < 10:
                r_dest = 1
                r_time_episode = 1 + self.max_step - self.total_record['time_step']

        # reward for template goal
        r_goal_rs = 0
        r_goal = 0
        if self.tmp_goal is not None:  # If at tmp goal

            goal = self.tmp_goal.transform.location
            last_dis = np.sqrt(
                (self.last_ego_transform.location.x - goal.x) ** 2 + (self.last_ego_transform.location.y - goal.y) ** 2)
            now_dis = np.sqrt((ego_x - goal.x) ** 2 + (ego_y - goal.y) ** 2)
            # tmp goal rs
            r_goal_rs = 0.99 * now_dis - last_dis

            if now_dis < 1:
                r_goal = 1
                self.tmp_goal = None

        # longitudinal speed
        lspeed_lon = obs['orgin_state'][3]
        self.total_record['reward_v_sum'] += lspeed_lon
        # cost for too fast
        # m1
        # r_speed_lon = (2. / self.desired_speed) * lspeed_lon - (lspeed_lon / self.desired_speed) ** 2
        # m2
        # ratio_cd=min(self.desired_speed, lspeed_lon) / self.desired_speed
        # r_speed_lon = ratio_cd -  4 * max(0,lspeed_lon - self.desired_speed) / ( self.desired_speed)
        # m3
        start_r = 0.1
        lspeed_lon=max(lspeed_lon,0)
        ratio_cd = min(self.desired_speed, lspeed_lon) / self.desired_speed
        ratio_cd2 = min(self.desired_speed * start_r, lspeed_lon) / (self.desired_speed * start_r)
        r_speed_lon = start_r * math.pow(ratio_cd2, 0.25) + \
                      (1 - start_r) * ratio_cd - max(0, lspeed_lon - self.desired_speed) / (self.desired_speed)
        # reward for collision
        c_collision = 0
        if len(self.collision_hist) > 0:
            c_collision = -1

        # reward for out of lane
        lateral_dis = obs['orgin_state'][0]

        c_out_route, c_out_lane, c_out = 0, 0, 0
        # c_out_route = -0.1 * np.round_(abs(lateral_dis) / self.out_lane_thres, 1)
        if abs(lateral_dis) > self.out_lane_thres and c_collision == 0:
            c_out = c_out_route = -1

        if len(self.invasion_hist) > 0 and c_collision == 0 and c_out_route == 0:
            c_out_lane = -1
            self.invasion_hist = []
            if self.reward_type == 3 or self.reward_type == 4 or self.reward_type == 6:
                c_out = c_out_lane

        # reward for direction
        delta_yaw = obs['orgin_state'][7]
        c_direction = -abs(delta_yaw) / np.pi

        if not self.autopilot:
            # c_action_lat = -(act_control.steer) ** 2  # ** 2
            # c_action_lon = -(max(act_control.throttle, act_control.brake)) ** 2  # ** 2
            # c_action_lat = -(act_control.steer - self.last_control.steer) ** 2
            # c_action_lon = -abs((3 * self.last_control.throttle - 8 * self.last_control.brake) - \
            #                     (3 * act_control.throttle - 8 * act_control.brake))
            # m1
            # cost for action :
            c_action_lon = 0.
            c_action_lat = 0.
            if len(self.his_act_deque) > 2:
                delta_lat = []
                delta_lon = []
                for i in range(1, len(self.his_act_deque)):
                    if self.his_act_deque[i - 1][0] * self.his_act_deque[i][0] >= 0:
                        delta_lon.append(0.)
                    else:
                        delta_lon.append((self.his_act_deque[i - 1][1] - self.his_act_deque[i][1]) ** 2)
                    if self.his_act_deque[i - 1][1] * self.his_act_deque[i][1] >= 0:
                        delta_lat.append(0.)
                    else:
                        delta_lat.append((self.his_act_deque[i - 1][1] - self.his_act_deque[i][1]) ** 2)
                    # delta_lat.append(compute_angle(self.his_act_deque[i - 1][2], self.his_act_deque[i][2]))
                c_action_lon = -np.sum(delta_lon)
                c_action_lat = -np.sum(delta_lat)

            # m1-2
            c_action_lon1_2 = 0.
            c_action_lat1_2 = 0.
            beta = 1
            if len(self.his_act_deque) > 2:
                delta_lat = []
                delta_lon = []
                for i in range(1, len(self.his_act_deque)):
                    delta_lon.append(math.fabs(self.his_act_deque[i - 1][1] - self.his_act_deque[i][1]) ** beta)
                    delta_lat.append(math.fabs(self.his_act_deque[i - 1][1] - self.his_act_deque[i][1]) ** beta)
                    # delta_lat.append(compute_angle(self.his_act_deque[i - 1][2], self.his_act_deque[i][2]))
                idx = len(self.his_act_deque) if len(self.his_act_deque) < (1 / self.dt) else int(1 / self.dt)
                c_action_lon1_2 = -np.mean(delta_lon[-idx:])
                c_action_lat1_2 = -np.mean(delta_lat[-idx:])

            # m2
            # if len(self.his_act_deque) > (2 * int(1 / self.dt) + 1):
            #
            #     delta_lat_last = []
            #     delta_lon_last = []
            #     for i in range(1, len(self.his_act_deque) - int(1 / self.dt)):
            #         if self.his_act_deque[i - 1][0] * self.his_act_deque[i][0] >= 0:
            #             delta_lon_last.append(0.)
            #         else:
            #             delta_lon_last.append((self.his_act_deque[i - 1][1] - self.his_act_deque[i][1]) ** 2)
            #         if self.his_act_deque[i - 1][1] * self.his_act_deque[i][1] >= 0:
            #             delta_lat_last.append(0.)
            #         else:
            #             delta_lat_last.append((self.his_act_deque[i - 1][1] - self.his_act_deque[i][1]) ** 2)

            #     delta_lat_curr = []
            #     delta_lon_curr = []
            #     for i in range(int(1 / self.dt) + 1, len(self.his_act_deque)):
            #         if self.his_act_deque[i - 1][0] * self.his_act_deque[i][0] >= 0:
            #             delta_lon_curr.append(0.)
            #         else:
            #             delta_lon_curr.append((self.his_act_deque[i - 1][1] - self.his_act_deque[i][1]) ** 2)
            #         if self.his_act_deque[i - 1][1] * self.his_act_deque[i][1] >= 0:
            #             delta_lat_curr.append(0.)
            #         else:
            #             delta_lat_curr.append((self.his_act_deque[i - 1][1] - self.his_act_deque[i][1]) ** 2)
            #
            #     c_action_lon = -(np.sum(delta_lon_curr) - np.sum(delta_lon_last))
            #     c_action_lat = -(np.sum(delta_lat_curr) - np.sum(delta_lat_last))

            c_acc_lon = 0
            c_acc_lat = 0
            c_jerk_lon = 0
            c_jerk_lat = 0
            c_lat_dis = 0
            beta = 2
            g = 9.8
            if len(self.his_act_deque) >= int(1 / self.dt):
                acc_lon = 0
                acc_lat = 0
                lat_dis = 0
                for i in range(len(self.his_act_deque) - int(1 / self.dt), len(self.his_act_deque)):
                    acc_lon += self.his_act_deque[i][3]
                    acc_lat += self.his_act_deque[i][4]
                    lat_dis += self.his_act_deque[i][5]
                acc_lon = acc_lon / int(1 / self.dt) / g
                acc_lat = acc_lat / int(1 / self.dt) / g
                c_lat_dis = -lat_dis / int(1 / self.dt)
                c_acc_lon = -math.pow(acc_lon, beta)
                c_acc_lat = -math.pow(acc_lat, beta)

            if len(self.his_act_deque) >= 2 * int(1 / self.dt):
                jerk_lon = 0
                jerk_lat = 0
                for i in range(len(self.his_act_deque) - 2 * int(1 / self.dt),
                               len(self.his_act_deque) - int(1 / self.dt)):
                    jerk_lon += self.his_act_deque[i][3]
                    jerk_lat += self.his_act_deque[i][4]
                jerk_lon = jerk_lon / int(1 / self.dt) / g - acc_lon
                jerk_lat = jerk_lat / int(1 / self.dt) / g - acc_lat
                c_jerk_lon = -math.pow(jerk_lon, beta)
                c_jerk_lat = -math.pow(jerk_lat, beta)

        # cost for stop illagel
        # update stop_step_time
        if (lspeed_lon < 1e-3) and ((self.vehicle_traffic >= 0) and (not self.vehicle_front)):
            self.total_record['stop_step_time'] += 1
        else:
            self.total_record['stop_step_time'] = 0

        c_stop = 0
        if self.total_record['stop_step_time'] > self.stop_step_thre:
            c_stop = -1
        elif self.total_record['stop_step_time'] > 0:
            c_stop = -0.02

        # break traffic rule
        # c_traffic_light = 0.
        # if (self.last_vehicle_traffic <= 0) and (self.current_waypoint.is_junction):
        #     c_traffic_light = -1.

        self.r_info = {
            'r_dest': r_dest,
            'r_time_episode': r_time_episode,
            'r_speed_lon': r_speed_lon,
            'r_goal': r_goal,
            'r_goal_rs': r_goal_rs,
            'c_collision': c_collision,
            'c_lat_dis': c_lat_dis,
            'c_out_route': c_out_route,
            'c_out_lane': c_out_lane,
            'c_out': c_out,
            'c_action_lat': c_action_lat,
            'c_action_lon': c_action_lon,
            'c_action_lat1_2': c_action_lat1_2,
            'c_action_lon1_2': c_action_lon1_2,
            'c_acc_lat': c_acc_lat,
            'c_acc_lon': c_acc_lon,
            'c_jerk_lat': c_jerk_lat,
            'c_jerk_lon': c_jerk_lon,
            'c_direction': c_direction,
            'c_stop': c_stop,
        }
        return self.r_info

    def _terminal(self, r_info):
        """
        Calculate whether to terminate the current episode.
        0 means finished
        1 means outlane
        2 means collision
        """

        # If reach maximum timestep
        if self.total_record['time_step'] >= self.max_step:
            return True
        target_step = self.init_dt * self.max_step
        # target_step = 2 * self.init_dt * self.max_step if self.if_noise_dt else self.init_dt * self.max_step
        if self.total_record['time'] > target_step:
            return True

        # If at destination
        if (not r_info['r_dest'] == 0) and self.if_dest_end:
            return True

        # If collides
        if r_info['c_collision'] == -1:
            return True

        # If out of lane or route
        if r_info['c_out'] == -1:
            return True

        #   If stop too long steps
        if r_info['c_stop'] == -1:
            return True

        return False

    def _convert_obs(self, obs):
        if 'othervehs' in self.obs_space_type:
            obs = np.hstack((obs['orgin_state'], obs['waypoint'].flatten(), obs['othervehs'].flatten()))
        else:
            obs = np.hstack((obs['orgin_state'], obs['waypoint'].flatten()))
        if np.isnan(obs).any():
            # print(obs)
            return np.nan_to_num(obs)
        else:
            return obs

    def _convert_reward(self, done, r_info):
        if done:
            v_ratio = (self.total_record['reward_v_sum'] / self.total_record['time_step']) / self.desired_speed
            r_v_evg = v_ratio ** 2 if v_ratio <= 1 else 2 - v_ratio ** 2
        else:
            r_v_evg = 0.
        # default: consider action penalty and out lane penalty in reward
        r_task_list = [1. * r_info['r_speed_lon'],
                       # 10. * r_v_evg
                       # 1. * r_info['r_goal'],
                       15. * r_info['c_collision'],
                       5. * r_info['c_out_route'],
                       5. * r_info['c_stop'],
                       0.1 * r_info['c_out_lane'],
                       0.1 * r_info['c_action_lat'],
                       0.1 * r_info['c_action_lon'],
                       ]
        if self.reward_type == 1:  # not consider out lane penalty in reward
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           ]
            return sum(r_task_list)
        elif self.reward_type == 2:  # consider out lane penalty in reward 1, other term in reward 0
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           ]
            c_safe_list = [r_info['c_out_lane']
                           ]
            return np.array([sum(r_task_list), sum(c_safe_list)])
        elif self.reward_type == 3:  # consider out lane terminal
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           1. * r_info['c_out_lane'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           ]
            return sum(r_task_list)
        elif self.reward_type == 4:  # consider out lane penalty in reward 1, other term in reward 0
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           ]
            c_safe_list = [r_info['c_out_lane']
                           ]
            return np.array([sum(r_task_list), sum(c_safe_list)])
        elif self.reward_type == 5:  # consider out lane terminal
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           ]
            return sum(r_task_list)
        elif self.reward_type == 6:  # consider out lane terminal
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           5. * r_info['c_out_lane'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           ]
            return sum(r_task_list)
        elif self.reward_type == 7:  # not consider c_action vs self.reward_type==1
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           ]
            return sum(r_task_list)
        elif self.reward_type == 8:  # not consider c_action vs self.reward_type==1
            r_task_list = [1. * r_info['r_time_episode'],
                           1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           ]
            return sum(r_task_list)
        elif self.reward_type == 9:  # not consider c_action vs self.reward_type==1
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           ]
            return sum(r_task_list)
        elif self.reward_type == 10:  # not consider c_action vs self.reward_type==1
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           0.2 * r_info['c_lat_dis'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           0.1 * r_info['c_acc_lat'],
                           # 0.1 * r_info['c_acc_lon'],
                           0.1 * r_info['c_jerk_lat'],
                           # 0.1 * r_info['c_jerk_lon'],
                           ]
            return sum(r_task_list)
        elif self.reward_type == 11:  # not consider c_action vs self.reward_type==1
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           0.5 * r_info['c_lat_dis'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           0.1 * r_info['c_acc_lat'],
                           0.1 * r_info['c_acc_lon'],
                           0.1 * r_info['c_jerk_lat'],
                           0.1 * r_info['c_jerk_lon'],
                           ]
            return sum(r_task_list)
        elif self.reward_type == 12:  # not consider c_action vs self.reward_type==1
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           0.5 * r_info['c_lat_dis'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           0.5 * r_info['c_acc_lat'],
                           0.5 * r_info['c_acc_lon'],
                           0.5 * r_info['c_jerk_lat'],
                           0.5 * r_info['c_jerk_lon'],
                           ]
            return sum(r_task_list)
        elif self.reward_type == 13:  # not consider c_action vs self.reward_type==1
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           0.5 * r_info['c_lat_dis'],
                           15. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           0.2 * r_info['c_action_lat1_2'],
                           0.2 * r_info['c_action_lon1_2'],
                           0.5 * r_info['c_acc_lat'],
                           0.5 * r_info['c_acc_lon'],
                           0.5 * r_info['c_jerk_lat'],
                           0.5 * r_info['c_jerk_lon'],
                           ]
            return sum(r_task_list)
        elif self.reward_type == 14:  # consider out lane penalty in reward 1, other term in reward 0
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           0.5 * r_info['c_lat_dis'],
                           5. * r_info['c_out_route'],
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           0.5 * r_info['c_acc_lat'],
                           0.5 * r_info['c_acc_lon'],
                           0.5 * r_info['c_jerk_lat'],
                           0.5 * r_info['c_jerk_lon'],
                           ]
            c_action_lat_list = [r_info['c_action_lat1_2']]
            c_acc_lat_list = [(1 / self.dt) * r_info['c_acc_lat']]
            result = np.array([sum(r_task_list), sum(c_action_lat_list), sum(c_acc_lat_list)])
            if np.isnan(result).any():
                return np.nan_to_num(result)
            return result
        elif self.reward_type == 15:  # consider out lane penalty in reward 1, other term in reward 0
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           0.5 * r_info['c_lat_dis'],
                           15. * r_info['c_out_route'],  # 5->15
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           0.5 * r_info['c_acc_lat'],
                           0.5 * r_info['c_acc_lon'],
                           0.5 * r_info['c_jerk_lat'],
                           0.5 * r_info['c_jerk_lon'],
                           ]
            c_action_lat_list = [(1 / self.dt) * r_info['c_action_lat1_2']]
            c_acc_lat_list = [r_info['c_acc_lat']]
            result = np.array([sum(r_task_list), sum(c_action_lat_list), sum(c_acc_lat_list)])
            if np.isnan(result).any():
                return np.nan_to_num(result)
            return result
        elif self.reward_type == 16:  # consider out lane penalty in reward 1, other term in reward 0
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           0.5 * r_info['c_lat_dis'],
                           15. * r_info['c_out_route'],  # 5->15
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           0.5 * r_info['c_acc_lat'],
                           0.5 * r_info['c_acc_lon'],
                           0.5 * r_info['c_jerk_lat'],
                           0.5 * r_info['c_jerk_lon'],
                           ]
            c_action_lat_list = [(1 / self.dt) * r_info['c_action_lat1_2']]
            c_acc_lat_list = [r_info['c_acc_lat']]
            c_lat_dis = [r_info['c_lat_dis']]
            result = np.array([sum(r_task_list), sum(c_action_lat_list), sum(c_acc_lat_list), sum(c_lat_dis)])
            if np.isnan(result).any():
                return np.nan_to_num(result)
            return result
        elif self.reward_type == 17:  # consider out lane penalty in reward 1, other term in reward 0
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           0.5 * r_info['c_lat_dis'],
                           15. * r_info['c_out_route'],  # 5->15
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           0.5 * r_info['c_acc_lat'],
                           0.5 * r_info['c_acc_lon'],
                           0.5 * r_info['c_jerk_lat'],
                           0.5 * r_info['c_jerk_lon'],
                           ]
            c_action_lat_list = [r_info['c_action_lat1_2']]
            c_acc_lat_list = [(1 / self.dt) * r_info['c_acc_lat']]
            c_lat_dis = [r_info['c_lat_dis']]
            result = np.array([sum(r_task_list), sum(c_action_lat_list), sum(c_acc_lat_list), sum(c_lat_dis)])
            if np.isnan(result).any():
                return np.nan_to_num(result)
            return result
        elif self.reward_type == 18:  # consider out lane penalty in reward 1, other term in reward 0
            r_task_list = [1. * r_info['r_speed_lon'],
                           15. * r_info['c_collision'],
                           0.5 * r_info['c_lat_dis'],
                           15. * r_info['c_out_route'],  # 5->15
                           5. * r_info['c_stop'],
                           0.1 * r_info['c_action_lat'],
                           0.1 * r_info['c_action_lon'],
                           0.5 * r_info['c_acc_lat'],
                           0.5 * r_info['c_acc_lon'],
                           0.5 * r_info['c_jerk_lat'],
                           0.5 * r_info['c_jerk_lon'],
                           ]
            c_action_lat_list = [(1 / self.dt) * r_info['c_action_lat1_2']]
            c_acc_lat_list = [r_info['c_acc_lat']]
            c_lat_dis = [r_info['c_lat_dis']]
            result = np.array([sum(r_task_list), sum(c_acc_lat_list), sum(c_lat_dis)])
            if np.isnan(result).any():
                return np.nan_to_num(result)
            return result
        else:
            return sum(r_task_list)

    def _get_info(self, obs, r_info):

        info = {}
        # control = self.ego.get_control()
        # info['steer']=control.steer
        delta_steer = 0
        steer = 0
        lon_action = 0
        if len(self.his_act_deque) >= int(1 / self.dt):
            for i in range(len(self.his_act_deque) - int(1 / self.dt), len(self.his_act_deque)):
                delta_steer += (math.fabs(self.his_act_deque[i - 1][1] - self.his_act_deque[i][1]))
                steer += self.his_act_deque[i][1]
                lon_action += self.his_act_deque[i][0]
            delta_steer = delta_steer / int(1 / self.dt)
            steer = steer / int(1 / self.dt)
            lon_action = lon_action / int(1 / self.dt)
        info['delta_steer'] = delta_steer
        info['lon_action'] = lon_action
        info['steer'] = steer
        info['a0'] = self.his_act_deque[-1][0]
        info['a1'] = self.his_act_deque[-1][1]
        # info['delta_steer'] = (self.his_act_deque[-1][1] - self.his_act_deque[-2][1]) if len(
        #     self.his_act_deque) > 1 else 0
        # info['delta_action_steer'] = (self.his_act_deque[-1][6] - self.his_act_deque[-2][6]) if len(
        #     self.his_act_deque) > 1 else 0
        v = self.ego.get_velocity()
        info['velocity'] = np.sqrt(v.x ** 2 + v.y ** 2)
        info['velocity_lon'] = obs['orgin_state'][3]
        info['velocity_lat'] = obs['orgin_state'][4]
        next_x, next_y = get_pos(self.ego)
        self.total_record['total_dis'] += np.sqrt((next_x - self.last_ego_transform.location.x) ** 2 +
                                                  (next_y - self.last_ego_transform.location.y) ** 2)
        info['lat_distance'] = (obs['orgin_state'][0])
        info['delta_yaw'] = (obs['orgin_state'][7])
        info['yaw_angle'] = (obs['orgin_state'][1])

        g = 9.8
        acc_lon = 0
        acc_lat = 0
        if len(self.his_act_deque) >= int(1 / self.dt):

            for i in range(len(self.his_act_deque) - int(1 / self.dt), len(self.his_act_deque)):
                acc_lon += self.his_act_deque[i][3]
                acc_lat += self.his_act_deque[i][4]
            acc_lon = acc_lon / int(1 / self.dt) / g
            acc_lat = acc_lat / int(1 / self.dt) / g
        jerk_lon = 0
        jerk_lat = 0
        if len(self.his_act_deque) >= 2 * int(1 / self.dt):
            for i in range(len(self.his_act_deque) - 2 * int(1 / self.dt),
                           len(self.his_act_deque) - int(1 / self.dt)):
                jerk_lon += self.his_act_deque[i][3]
                jerk_lat += self.his_act_deque[i][4]
            jerk_lon = jerk_lon / int(1 / self.dt) / g - acc_lon
            jerk_lat = jerk_lat / int(1 / self.dt) / g - acc_lat

        info['jerk_lat'] = (jerk_lat)
        info['jerk_lon'] = (jerk_lon)
        info['acc_lon'] = (acc_lon)
        info['acc_lat'] = (acc_lat)

        info['distance'] = self.total_record['total_dis']
        info['standingStep'] = 1 if info['velocity'] < 1e-3 else 0
        info['outlane'] = 1 if r_info['c_out_lane'] == -1 else 0
        info['outroute'] = 1 if r_info['c_out_route'] == -1 else 0
        # info['lane_skewness'] = abs(obs['orgin_state'][0]) / self.out_lane_thres
        info['collision'] = 1 if r_info['c_collision'] == -1 else 0
        info['finish'] = 1 if r_info['r_dest'] == 1 else 0
        info['time'] = self.dt

        return info

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                    actor.destroy()

    def _get_near_vehs(self):
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        ego_transform = self.ego.get_transform()
        ego_v = self.ego.get_velocity()
        ego_vector = np.array([np.cos(ego_transform.rotation.yaw), np.sin(ego_transform.rotation.yaw)])
        # near_veh_list = []
        near_vehs = np.zeros((5, 5))
        i = 0
        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self.ego.id:
                continue
            tarveh_transform = target_vehicle.get_transform()
            tarveh_v = target_vehicle.get_velocity()
            if is_within_distance_ahead(tarveh_transform,
                                        ego_transform,
                                        self.sampling_radius * 7):
                # near_veh_list.append(target_vehicle.id)
                # near_veh_list.append([target_vehicle.id,
                #                       tarveh_transform.location.x,
                #                       tarveh_transform.location.y,
                #                       tarveh_transform.rotation.yaw,
                #                       tarveh_v.x,
                #                       tarveh_v.y])
                tar_yaw_vector = np.array(
                    [np.cos(tarveh_transform.rotation.yaw), np.sin(tarveh_transform.rotation.yaw)])
                near_vehs[i, :] = np.array([ego_transform.location.x - tarveh_transform.location.x,
                                            ego_transform.location.y - tarveh_transform.location.y,
                                            np.arcsin(np.clip(np.cross(tar_yaw_vector, ego_vector), -1., 1.)),
                                            ego_v.x - tarveh_v.x,
                                            ego_v.y - tarveh_v.y])
                i += 1
                if i == 5:
                    break
        return near_vehs
