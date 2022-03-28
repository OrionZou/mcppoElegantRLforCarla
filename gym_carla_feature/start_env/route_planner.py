#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This file is modified from <https://github.com/carla-simulator/carla>:
# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from enum import Enum
from collections import deque
import random
import numpy as np
import math
import carla
from copy import deepcopy
from gym_carla_feature.start_env.navigation.global_route_planner import GlobalRoutePlanner
from gym_carla_feature.start_env.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from gym_carla_feature.start_env.misc import compute_distance, is_within_distance_ahead, compute_magnitude_angle


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4


class RoutePlanner():
    def __init__(self, vehicle, buffer_size, sampling_radius=5):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._sampling_radius = sampling_radius
        self._min_interval_distance = np.sqrt((self._sampling_radius / 2) ** 2 + 2 ** 2)

        self._buffer_size = buffer_size
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        self._waypoints_queue = deque(maxlen=600)
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))
        self._target_road_option = RoadOption.LANEFOLLOW

        self._last_traffic_light = None
        self._proximity_tlight_threshold = 100.0  # meters
        self._proximity_vehicle_threshold = 10.0  # meters

        self._grp = None

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = retrieve_options(
                    next_waypoints, last_waypoint)

                road_option = road_options_list[1]
                # road_option = random.choice(road_options_list)

                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def set_destination(self, end_location, start_location=None, clean=True):
        """
        This method creates a list of waypoints from navigation's position to destination location
        based on the route returned by the global router.

            :param start_location: initial position
            :param end_location: final position
            :param clean: boolean to clean the waypoint queue
        """
        if start_location is None:
            if clean:
                start_waypoint = self._current_waypoint
            else:
                start_waypoint = self._waypoints_queue[-1][0]
        else:
            start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        if self._grp is None:
            wld = self._vehicle.get_world()
            dao = GlobalRoutePlannerDAO(
                wld.get_map(), sampling_resolution=self._sampling_radius)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp
        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)
        if clean:
            self._waypoints_queue.clear()
        else:
            wpset = zip(*route).__next__()
            for i in range(len(route)):
                if self._waypoints_queue[-1][0] not in wpset:
                    route = route[i:]
        for wp, rp in route:
            self._waypoints_queue.append((wp, rp))

    def get_waypoints_queue(self):
        res = deque(maxlen=600)
        for ele in self._waypoint_buffer:
            res.append(ele)
        for ele in self._waypoints_queue:
            res.append(ele)
        return res

    def run_step(self):
        """
        :return:
        """
        waypoints = self._get_waypoints()
        red_light, vehicle_front, light_time, _ = self._get_hazard()
        # todo some trafficlight get_elapsed_time=0
        # return waypoints, self._current_waypoint, self._waypoint_buffer[2][0], light_time, vehicle_front
        return waypoints, self._current_waypoint, self._waypoint_buffer[2][0], 0., vehicle_front

    def is_empty_queue(self):
        if len(self._waypoints_queue) < 5 * self._buffer_size:
            return True
        else:
            return False

    def _get_waypoints(self):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """
        # not enough waypoints in the horizon? => add more!
        to_add_wp_num = 3 * self._buffer_size - len(self._waypoints_queue)
        if to_add_wp_num > 0:
            self._compute_next_waypoints(k=to_add_wp_num)
        vehicle_transform = self._vehicle.get_transform()
        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # purge the queue of obsolete waypoints
        out_num = 0
        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if not is_within_distance_ahead(waypoint.transform, vehicle_transform, max_distance=20):
                out_num += 1
            else:
                break
        if out_num > 1:
            for _ in range(out_num - 1):
                self._waypoint_buffer.popleft()
        # Buffering the waypoints
        while len(self._waypoint_buffer) < self._buffer_size:
            if self._waypoints_queue:
                ele = self._waypoints_queue.popleft()
                if len(self._waypoint_buffer) > 0:
                    if compute_distance(ele[0].transform.location,
                                        self._waypoint_buffer[-1][0].transform.location) < 1.:
                        # if ele[0].transform == self._waypoint_buffer[-1][0].transform:
                        continue
                self._waypoint_buffer.append(ele)
            else:
                break
        waypoints = []
        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            waypoints.append(waypoint)
        return waypoints

    def _get_hazard(self):
        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, target_vehicle = self._is_vehicle_hazard(vehicle_list)

        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        # light_state = self._vehicle.is_at_traffic_light()
        # traffic_light = self._vehicle.get_traffic_light()
        if light_state:
            traffic_time = traffic_light.get_elapsed_time() / traffic_light.get_red_time() - 1
        else:
            if traffic_light is not None:
                if traffic_light.state == carla.TrafficLightState.Green:
                    traffic_time = 1 - traffic_light.get_elapsed_time() / traffic_light.get_green_time()
                else:
                    traffic_time = 0
            else:
                traffic_time = 1
        # todo: traffic_light.get_red_time() = 2,
        return light_state, vehicle_state, traffic_time, target_vehicle

    # todo
    def _around_vehicle(self, vehicle_list):
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.is_junction:
                pass
            else:
                if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id:
                    continue
                else:
                    if ego_vehicle_waypoint.lane_id == target_vehicle_waypoint.lane_id:
                        pass
                    elif ego_vehicle_waypoint.lane_id > target_vehicle_waypoint.lane_id:  # 右
                        pass
                    else:  # 左
                        pass

            if is_within_distance_ahead(target_vehicle.get_transform(),
                                        self._vehicle.get_transform(),
                                        self._proximity_vehicle_threshold):
                return (True, target_vehicle)

        return (False, None)

    def _is_vehicle_hazard(self, vehicle_list):
        """
        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(target_vehicle.get_transform(),
                                        self._vehicle.get_transform(),
                                        self._proximity_vehicle_threshold):
                return (True, target_vehicle)

        return (False, None)

    def _is_light_red(self, lights_list):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_location = self._get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if is_within_distance_ahead(object_waypoint.transform,
                                        self._vehicle.get_transform(),
                                        self._proximity_tlight_threshold):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return (True, traffic_light)
                else:
                    return (False, traffic_light)

        return (False, None)

    def _get_trafficlight_trigger_location(self, traffic_light):  # pylint: disable=no-self-use
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """

        def rotate_point(point, radians):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)


def retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
         candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
         RoadOption.STRAIGHT
         RoadOption.LEFT
         RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
